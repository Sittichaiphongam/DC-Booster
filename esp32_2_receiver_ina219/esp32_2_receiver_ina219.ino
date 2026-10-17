// ESP32 #2: non-blocking UART, Wi-Fi reconnect, self-contained XY dashboard
#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>
#include <esp_system.h>
#include <string.h>
// Explicit prototypes: prevent Arduino from guessing function declarations.
void setup();
void loop();
int outputValueToChannel(int v);
void selectChannel(int ch);
bool parseNumber(const char* s, float& v);
void processLine(char* s);
void readUART();
String number(float v, int digits);
void handleData();
void handleRoot();
// Timers persist between loop iterations.
uint32_t lastReconnect = 0;
uint32_t lastRead = 0;
uint32_t lastPrint = 0;
bool wasConnected = false;

// ================== Wi-Fi settings ==================
const char* ssid = "Chopper";
const char* password = "12345678";

// ================== Web server and INA219 sensors ==================
WebServer server(80);
Adafruit_INA219 ina219_out(0x40), ina219_in(0x41);

// ================== Multiplexer control pins ==================
const int s0 = 25, s1 = 26, s2 = 27, s3 = 14;
int currentChannel = 0, g_output_value = 0;
float g_temp = 0, g_irr = 0, g_vin = 0, g_iin = 0, g_pin = 0, g_vout = 0, g_iout = 0, g_pout = 0;
bool inputReady = false, outputReady = false, uartReceived = false;
uint32_t sampleId = 0, lastSample = 0, lastUart = 0, lastByte = 0, badUart = 0;
String bootId;
char rx[128];
size_t rxLen = 0;
bool rxOverflow = false;

// ================== Convert 0–1023 into eight channels ==================
int outputValueToChannel(int v) {
  return constrain(v, 0, 1023) / 128;
}

// ================== Select the feedback resistor channel ==================
void selectChannel(int ch) {
  currentChannel = constrain(ch, 0, 7);
  digitalWrite(s0, currentChannel & 1);
  digitalWrite(s1, (currentChannel >> 1) & 1);
  digitalWrite(s2, (currentChannel >> 2) & 1);
  digitalWrite(s3, (currentChannel >> 3) & 1);
  Serial.printf("Auto-Switched to Channel: %d\n", currentChannel);
}

// ================== Validate received numbers ==================
bool parseNumber(const char* s, float& v) {
  char* end;
  v = strtof(s, &end);
  if (end == s || !isfinite(v)) return false;
  while (isspace((unsigned char)*end)) end++;
  return *end == '\0';
}
void processLine(char* s) {
  float t = 0, irr = 0, v = 0;
  char* a = strchr(s, ',');
  if (a) {
    *a = 0;
    char* b = strchr(a + 1, ',');
    if (!b) {
      badUart++;
      return;
    }
    *b = 0;
    if (!parseNumber(s, t) || !parseNumber(a + 1, irr) || !parseNumber(b + 1, v)) {
      badUart++;
      return;
    }
  } else if (!parseNumber(s, v)) {
    badUart++;
    return;
  }
  if (v < 0 || v > 1023 || floorf(v) != v) {
    badUart++;
    return;
  }
  if (a) {
    g_temp = t;
    g_irr = irr;
  }
  g_output_value = (int)v;
  int ch = outputValueToChannel(g_output_value);
  if (ch != currentChannel) selectChannel(ch);
  lastUart = millis();
  uartReceived = true;
}

// ================== Read UART without blocking the web server ==================
void readUART() {
  int budget = 256;
  while (Serial2.available() && budget-- > 0) {
    char c = Serial2.read();
    lastByte = millis();
    if (c == '\n') {
      if (!rxOverflow && rxLen) {
        rx[rxLen] = 0;
        processLine(rx);
      } else if (rxOverflow) badUart++;
      rxLen = 0;
      rxOverflow = false;
    } else if (c != '\r') {
      if (rxLen < sizeof(rx) - 1 && !rxOverflow) rx[rxLen++] = c;
      else rxOverflow = true;
    }
  }
  if ((rxLen || rxOverflow) && millis() - lastByte > 1000) {
    rxLen = 0;
    rxOverflow = false;
    badUart++;
  }
}
String number(float v, int digits) {
  return isfinite(v) ? String(v, digits) : String("null");
}

// ================== GET /data: readings and connection diagnostics ==================
void handleData() {
  float eff = g_pin > 0.001f ? 100.0f * g_pout / g_pin : 0;
  String j;
  j.reserve(800);
  j = "{\"channel\":" + String(currentChannel);
  j += ",\"output_value\":" + String(g_output_value);
  j += ",\"temp\":" + number(g_temp, 2) + ",\"irradiation\":" + number(g_irr, 4);
  j += ",\"vin\":" + number(g_vin, 3) + ",\"iin\":" + number(g_iin, 4) + ",\"pin\":" + number(g_pin, 4);
  j += ",\"vout\":" + number(g_vout, 3) + ",\"iout\":" + number(g_iout, 4) + ",\"pout\":" + number(g_pout, 4);
  j += ",\"efficiency\":" + number(eff, 1);
  j += ",\"sample_id\":" + String(sampleId) + ",\"boot_id\":\"" + bootId + "\"";
  j += ",\"uptime_ms\":" + String(millis()) + ",\"sample_age_ms\":" + String(millis() - lastSample);
  j += ",\"rssi\":" + String(WiFi.RSSI());
  j += ",\"uart_received\":" + String(uartReceived ? "true" : "false");
  j += ",\"uart_age_ms\":" + String(uartReceived ? String(millis() - lastUart) : String("null"));
  j += ",\"uart_errors\":" + String(badUart) + "}";
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "application/json", j);
}

// ================== GET /: dashboard stored in dashboard.h ==================
void handleRoot() {
  server.send(
    200,
    "text/plain; charset=utf-8",
    "ESP32 is online. Open WebDC HTML on your computer. Data: /data"
  );
}

// ================== Initialization ==================
void setup() {
  Serial.begin(115200);
  Serial2.begin(115200, SERIAL_8N1, 16, 17);
  Wire.begin(21, 22);
  Wire.setTimeOut(50);
  pinMode(s0, OUTPUT);
  pinMode(s1, OUTPUT);
  pinMode(s2, OUTPUT);
  pinMode(s3, OUTPUT);
  selectChannel(0);
  inputReady = ina219_in.begin();
  outputReady = ina219_out.begin();
  if (!inputReady) Serial.println("ERROR INA219 Input 0x41 missing");
  if (!outputReady) Serial.println("ERROR INA219 Output 0x40 missing");
  bootId = String(esp_random(), HEX);
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.setSleep(false);
  WiFi.begin(ssid, password);
  server.on("/", HTTP_GET, handleRoot);
  server.on("/data", HTTP_GET, handleData);
  server.begin();
  Serial.println("Receiver ready; Wi-Fi connects in background.");
}

// ================== Main loop: UART, HTTP, Wi-Fi reconnect and measurement ==================
void loop() {
  readUART();
  server.handleClient();
  uint32_t now = millis();
  bool connected = WiFi.status() == WL_CONNECTED;
  if (connected && !wasConnected) {
    Serial.print("Open http://");
    Serial.print(WiFi.localIP());
    Serial.println("/");
  }
  if (!connected && wasConnected) Serial.println("Wi-Fi lost; UART/control continue.");
  wasConnected = connected;
  if (!connected && now - lastReconnect >= 15000) {
    lastReconnect = now;
    Serial.println("Retry Wi-Fi...");
    WiFi.reconnect();
  }
  if (now - lastRead >= 1000) {
    lastRead = now;
    if (inputReady) {
      g_vin = ina219_in.getBusVoltage_V();
      g_iin = ina219_in.getCurrent_mA() / 1000.0f;
      g_pin = ina219_in.getPower_mW() / 1000.0f;
    } else g_vin = g_iin = g_pin = NAN;
    if (outputReady) {
      g_vout = ina219_out.getBusVoltage_V();
      g_iout = ina219_out.getCurrent_mA() / 1000.0f;
      g_pout = ina219_out.getPower_mW() / 1000.0f;
    } else g_vout = g_iout = g_pout = NAN;
    sampleId++;
    lastSample = millis();
    // Return fresh input measurements on TX2 (GPIO17) to ESP32 #1 RX2.
    if (inputReady && isfinite(g_vin) && isfinite(g_iin) && g_vin > 0 && g_iin >= 0) {
      Serial2.print("MEAS,");
      Serial2.print(g_vin, 4);
      Serial2.print(',');
      Serial2.println(g_iin, 6);
    }
  }
  if (now - lastPrint >= 5000) {
    lastPrint = now;

    const float eff =
      g_pin > 0.001f ? 100.0f * g_pout / g_pin : 0.0f;

    const float controlPercent =
      (g_output_value / 1023.0f) * 100.0f;

    Serial.printf(
      "Ch:%d | Temp:%.2f | Irr:%.4f"
      " | Vin:%.3f V | Iin:%.4f A"
      " | Pin:%.4f W | Pin(VI):%.4f W"
      " | Vout:%.3f V | Iout:%.4f A | Pout:%.4f W"
      " | Eff:%.1f %% | Control:%.2f %%"
      " | Output:%d | RSSI:%d | UART errors:%lu\n",
      currentChannel,
      g_temp,
      g_irr,
      g_vin,
      g_iin,
      g_pin,
      g_vin * g_iin,
      g_vout,
      g_iout,
      g_pout,
      eff,
      controlPercent,
      g_output_value,
      connected ? WiFi.RSSI() : 0,
      (unsigned long)badUart);
  }
  delay(1);
}