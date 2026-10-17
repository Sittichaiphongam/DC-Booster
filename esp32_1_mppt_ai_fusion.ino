// ESP32 #1: AI replay for display + measured-power discrete P&O control.
// Receiver mapping must be output_value / 128. No PWM is generated.
// AI power is diagnostic only until calibrated for the physical system.
// Assumes ESP32 #2 applies UART commands; MEAS does not acknowledge channel.
#include <Arduino.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "model.h"
#include "scaler_params.h"
#include "test_sequences.h"


// ======================================
// TENSOR ARENA & TFLITE
// ======================================
constexpr int kTensorArenaSize = 32 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// ======================================
// REPLAY LIST
// ======================================
const int NUM_REPLAY_SAMPLES = 10;
const int REPLAY_INDEX[NUM_REPLAY_SAMPLES] = {
  0, 21, 24, 27, 30, 34, 39, 44, 149, 72
};

int replay_pos = 0;

double sum_abs_err_temp = 0;
double sum_abs_err_irr = 0;
int eval_count = 0;


float feedbackV = 0, feedbackI = 0;
int commandedChannel = 3;
int searchDirection = 1;
bool controlStarted = false;
bool baselineReady = false;
float previousPower = 0.0f;
const float POWER_ABS_TOLERANCE_W = 0.01f;
const float POWER_REL_TOLERANCE = 0.02f;

// Call only once per control interval with fresh feedback.
int updateChannelMPPT(float power) {
  if (!controlStarted) {
    controlStarted = true;
    Serial.printf("[CHANNEL MPPT] Initialize Ch=%d; waiting for baseline\n",
                  commandedChannel);
    return commandedChannel;
  }

  if (baselineReady) {
    const float deltaPower = power - previousPower;
    const float tolerance = fmaxf(POWER_ABS_TOLERANCE_W,
                                 POWER_REL_TOLERANCE * fabsf(previousPower));
    if (deltaPower < -tolerance) searchDirection = -searchDirection;
    Serial.printf("[CHANNEL MPPT] Assumed measured Ch=%d | P=%.4f W"
                  " | DeltaP=%.4f W | Tolerance=%.4f W\n",
                  commandedChannel, power, deltaPower, tolerance);
  } else {
    baselineReady = true;
    Serial.printf("[CHANNEL MPPT] Baseline Ch=%d | P=%.4f W\n",
                  commandedChannel, power);
  }
  previousPower = power;

  if (commandedChannel == 7 && searchDirection > 0) searchDirection = -1;
  if (commandedChannel == 0 && searchDirection < 0) searchDirection = 1;
  commandedChannel += searchDirection;
  Serial.printf("[CHANNEL MPPT] Next Ch=%d | Direction=%d\n",
                commandedChannel, searchDirection);
  return commandedChannel;
}

bool feedbackReady = false, previousReady = false;
uint32_t feedbackAt = 0, lastCycle = 0, lastWaitLog = 0;
bool firstCycle = true;
char feedbackLine[96];
size_t feedbackLength = 0;
bool feedbackOverflow = false;
uint32_t feedbackByteAt = 0;
const uint32_t FEEDBACK_MAX_AGE_MS = 3000;
const uint32_t REPLAY_INTERVAL_MS = 30000;

void readFeedback();
bool parseFeedbackFloat(char* text, float &value);
void processFeedback(char* line);

bool parseFeedbackFloat(char* text, float &value) {
  char* end;
  value = strtof(text, &end);
  return end != text && *end == '\0' && isfinite(value);
}

void processFeedback(char* line) {
  if (strncmp(line, "MEAS,", 5) != 0) return;
  char* vText = line + 5;
  char* comma = strchr(vText, ',');
  if (!comma) return;
  *comma = '\0';
  float v, i;
  if (!parseFeedbackFloat(vText, v) ||
      !parseFeedbackFloat(comma + 1, i) || v <= 0 || i < 0) return;
  feedbackV = v;
  feedbackI = i;
  feedbackAt = millis();
  feedbackReady = true;
}

void readFeedback() {
  int budget = 256;
  while (Serial2.available() && budget-- > 0) {
    char c = Serial2.read();
    feedbackByteAt = millis();
    if (c == '\n') {
      if (!feedbackOverflow && feedbackLength) {
        feedbackLine[feedbackLength] = '\0';
        processFeedback(feedbackLine);
      }
      feedbackLength = 0;
      feedbackOverflow = false;
    } else if (c != '\r') {
      if (!feedbackOverflow && feedbackLength < sizeof(feedbackLine) - 1)
        feedbackLine[feedbackLength++] = c;
      else feedbackOverflow = true;
    }
  }
  if ((feedbackLength || feedbackOverflow) &&
      millis() - feedbackByteAt > 1000) {
    feedbackLength = 0;
    feedbackOverflow = false;
  }
}

float inverseScale(float scaledVal, int featureIdx) {
  float mn = FEATURE_MIN[featureIdx];
  float mx = FEATURE_MAX[featureIdx];
  return scaledVal * (mx - mn) + mn;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // เริ่มต้น Serial2 สำหรับส่งข้อมูลไปยัง ESP32 ตัวที่ 2
  Serial2.begin(115200, SERIAL_8N1, 16, 17);

  Serial.print("NUM_TEST_SAMPLES   = ");
  Serial.println(NUM_TEST_SAMPLES);
  Serial.print("NUM_REPLAY_SAMPLES = ");
  Serial.println(NUM_REPLAY_SAMPLES);
  Serial.print("TS_SEQ_LEN         = ");
  Serial.println(TS_SEQ_LEN);
  Serial.print("TS_NUM_FEATURES    = ");
  Serial.println(TS_NUM_FEATURES);

  if (TS_SEQ_LEN != SEQ_LEN || TS_NUM_FEATURES != NUM_FEATURES) {
    Serial.println(
      "ERROR: test sequence shape mismatch!"
    );
  }

  if (TS_SEQ_LEN != SEQ_LEN || TS_NUM_FEATURES != NUM_FEATURES) {
    while (true) delay(1000);
  }
  for (int n = 0; n < NUM_REPLAY_SAMPLES; n++) {
    if (REPLAY_INDEX[n] < 0 || REPLAY_INDEX[n] >= NUM_TEST_SAMPLES) {
      Serial.println("Replay index out of range");
      while (true) delay(1000);
    }
  }

  const tflite::Model* tflite_model = tflite::GetModel(g_model);

  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (true) delay(1000);
  }

  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddReshape();
  resolver.AddShape();
  resolver.AddStridedSlice();
  resolver.AddPack();

  static tflite::MicroInterpreter static_interpreter(
    tflite_model, resolver, tensor_arena, kTensorArenaSize
  );

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    while (true) delay(1000);
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  if (input_tensor->type != kTfLiteFloat32 ||
      output_tensor->type != kTfLiteFloat32 ||
      input_tensor->bytes != TS_SEQ_LEN * TS_NUM_FEATURES * sizeof(float) ||
      output_tensor->bytes < 2 * sizeof(float)) {
    Serial.println("Unsupported tensor type or shape");
    while (true) delay(1000);
  }

  Serial.println(
    "Ready: AI replay for display; discrete P&O controls channel every 30 s.\n"
  );
}

void loop() {
  readFeedback();
  uint32_t now = millis();
  if (!feedbackReady || now - feedbackAt > FEEDBACK_MAX_AGE_MS) {
    if (now - lastWaitLog >= 3000) {
      lastWaitLog = now;
      Serial.println("WAIT: fresh MEAS,Vin,Iin from ESP32 #2; no command sent.");
    }
    // After an interruption, reacquire a baseline before comparison.
    baselineReady = false;
    delay(1);
    return;
  }
  if (!firstCycle && now - lastCycle < REPLAY_INTERVAL_MS) {
    delay(1);
    return;
  }
  firstCycle = false;
  lastCycle = now;
  int sample_idx = REPLAY_INDEX[replay_pos];

  // 1) โหลด Sequence เข้า Input Tensor
  const float* seq = TEST_X[sample_idx];

  for (int i = 0; i < TS_SEQ_LEN * TS_NUM_FEATURES; i++) {
    input_tensor->data.f[i] = seq[i];
  }

  // 2) Inference
  uint32_t start_us = micros();
  TfLiteStatus status = interpreter->Invoke();
  uint32_t end_us = micros();

  if (status != kTfLiteOk) {
    Serial.print("Invoke failed at sample index ");
    Serial.println(sample_idx);
    replay_pos = (replay_pos + 1) % NUM_REPLAY_SAMPLES;
    return;
  }

  float latency_ms = (end_us - start_us) / 1000.0;

  // 3) Inverse-scale Predictions
  float pred_temp = inverseScale(
    output_tensor->data.f[0], TARGET_0_IDX
  );
  float pred_irr = inverseScale(
    output_tensor->data.f[1], TARGET_1_IDX
  );

  if (!isfinite(pred_temp) || !isfinite(pred_irr)) {
    Serial.println("Invalid AI prediction; command skipped.");
    return;
  }
  if (pred_irr < 0) pred_irr = 0;

  // คำนวณ DC Power
  // ตรวจสอบว่าสูตรนี้ใช้สเกลเดียวกับฮาร์ดแวร์จริงก่อนนำไปใช้งาน
  float pred_dc_power =
    -196.5433103
    + 12.18301632 * pred_temp
    + 13104.75613 * pred_irr;

  if (pred_dc_power < 0) pred_dc_power = 0;

  // 4) คำนวณ Error เทียบกับค่าจริงของ sample เดียวกัน
  float actual_temp = TEST_Y_ACTUAL[sample_idx][0];
  float actual_irr = TEST_Y_ACTUAL[sample_idx][1];

  float err_temp = fabs(pred_temp - actual_temp);
  float err_irr = fabs(pred_irr - actual_irr);

  sum_abs_err_temp += err_temp;
  sum_abs_err_irr += err_irr;
  eval_count++;

  // Read again after inference and reject stale feedback.
  readFeedback();
  if (!feedbackReady || millis() - feedbackAt > FEEDBACK_MAX_AGE_MS) {
    Serial.println("Feedback expired; command skipped.");
    return;
  }
  const float measuredV = feedbackV;
  const float measuredI = feedbackI;
  const float measuredPower = measuredV * measuredI;
  if (!isfinite(measuredPower)) {
    Serial.println("Invalid measured power; command skipped.");
    baselineReady = false;
    return;
  }

  const int nextChannel = updateChannelMPPT(measuredPower);
  const int output_value = nextChannel * 128 + 64;
  const float output_ratio = output_value / 1023.0f;

  // 5) ส่งข้อมูลไปยัง ESP32 ตัวที่ 2
  // รูปแบบ: temp,irr,output_value
  Serial2.print(pred_temp, 2);
  Serial2.print(",");
  Serial2.print(pred_irr, 4);
  Serial2.print(",");
  Serial2.println(output_value);

  // 6) แสดงข้อมูลทาง Serial Monitor
  Serial.print("[replay_pos ");
  Serial.print(replay_pos);
  Serial.print(" / sample_idx ");
  Serial.print(sample_idx);
  Serial.print("] ");

  Serial.print("Lat=");
  Serial.print(latency_ms, 2);
  Serial.print("ms | ");

  Serial.print("Pred Temp=");
  Serial.print(pred_temp, 2);
  Serial.print(" | ");

  Serial.print("Pred Irr=");
  Serial.print(pred_irr, 4);

  Serial.print(" | AI power estimate (unused)=");
  Serial.print(pred_dc_power, 2);
  Serial.print(" W | ");

  Serial.print("Vin=");
  Serial.print(measuredV, 3);
  Serial.print(" | Iin=");
  Serial.print(measuredI, 4);
  Serial.print(" | Measured Pin=");
  Serial.print(measuredPower, 4);
  Serial.print(" W | Control=Discrete P&O | Commanded Channel=");
  Serial.print(nextChannel);
  Serial.print(" | Control (%)=");
 Serial.print(output_ratio * 100.0f, 2);
  Serial.print(" | Output Ratio: ");
  Serial.print(output_ratio, 4);
  Serial.print(" | Output Value: ");
  Serial.println(output_value);

  // แสดง Running MAE เมื่อครบ 1 รอบ
  replay_pos = (replay_pos + 1) % NUM_REPLAY_SAMPLES;

  if (replay_pos == 0) {
    Serial.println("--- Completed one replay cycle ---");
    Serial.print("Running MAE Module Temp: ");
    Serial.println(sum_abs_err_temp / eval_count, 4);
    Serial.print("Running MAE Irradiation: ");
    Serial.println(sum_abs_err_irr / eval_count, 4);
    Serial.println();
  }

  // Non-blocking 30-second scheduler keeps receiving feedback.
  delay(1);
}