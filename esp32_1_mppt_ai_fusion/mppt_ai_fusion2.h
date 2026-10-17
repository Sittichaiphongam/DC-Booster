#ifndef MPPT_AI_FUSION_H
#define MPPT_AI_FUSION_H

#include <Arduino.h>
#include <math.h>

// ============================================================
// MPPT_AI_Fusion
// แปลงจาก MATLAB Simulink function block "AI-MPPT with Temporal
// Fusion (PAST + FUTURE)" มาเป็น C++ class สำหรับ Arduino/ESP32
//
// วิธีใช้:
//   MPPT_AI_Fusion mppt;
//   ...
//   float D_next = mppt.update(V, I, vo, io, D, P_pred);
//   // mppt.direction  -> +1 = เพิ่ม duty cycle, -1 = ลด duty cycle
// ============================================================
class MPPT_AI_Fusion {
public:
  // ผลลัพธ์ล่าสุด เปิดเป็น public ไว้เผื่อ debug/log ผ่าน Serial
  float ai_conf     = 0.8f;
  bool  converged    = false;
  bool  P_pred_ok    = false;
  float P_now        = 0.0f;
  float P_err        = 0.0f;
  int   direction    = 1;   // +1 = ขึ้น, -1 = ลง (มาจาก sign(dir) ก่อนคูณ step)

  float update(float V, float I, float vo, float io, float D, float P_pred) {
    if (!_init) initState(V, I);

    // ---- Input protection ----
    if (vo == 0.0f) vo = V;
    if (io == 0.0f) io = I;

    // ---- Adaptive filtering ----
    const bool is_tr = (fabsf(V - vo) > 0.5f) || (fabsf(I - io) > 0.1f);
    const float a = is_tr ? 0.68f : kAlphaV;

    _V_f = a * V + (1.0f - a) * _V_f;
    _I_f = a * I + (1.0f - a) * _I_f;
    P_now = _V_f * _I_f;

    // ---- Update rolling history (shift left, push new sample at end) ----
    pushHistory(P_now, _V_f);

    float P_diff[kHist - 1];
    float V_diff[kHist - 1];
    for (int k = 0; k < kHist - 1; k++) {
      P_diff[k] = _P_hist[k + 1] - _P_hist[k];
      V_diff[k] = _V_hist[k + 1] - _V_hist[k];
    }

    // ---- PAST: trend & gradient ----
    static const float weights[9] = {0.03f, 0.05f, 0.07f, 0.10f, 0.13f,
                                      0.15f, 0.17f, 0.20f, 0.10f};
    float trend_score = 0.0f;
    for (int k = 0; k < 9; k++) trend_score += weights[k] * sgn(P_diff[k]);

    float sum_g = 0.0f;
    int   cnt_g = 0;
    for (int k = 0; k < 9; k++) {
      if (fabsf(V_diff[k]) > 1e-3f) {
        sum_g += P_diff[k] / V_diff[k];
        cnt_g++;
      }
    }
    const bool  gradient_stable = (cnt_g > 0);
    const float avg_gradient    = gradient_stable ? (sum_g / cnt_g) : 0.0f;

    // ---- Consistency: last 4 diffs agreeing in sign with the newest diff ----
    float consistency = 0.0f;
    for (int k = 5; k <= 8; k++) {
      if (sgn(P_diff[k]) == sgn(P_diff[8])) consistency += 1.0f;
    }
    consistency /= 4.0f;

    // ---- FUTURE: AI prediction ----
    P_pred_ok = (P_pred > 1.0f) && (P_pred < 2.0f * kPmppStc);
    float Vmpp_pred;

    if (P_pred_ok) {
      P_err = (P_pred - P_now) / fmaxf(P_pred, 1.0f);
      const float qual = 1.0f - fminf(fabsf(P_err), 1.0f);
      ai_conf = 0.35f * qual + 0.65f * ai_conf;

      const float pred_dir = sgn(P_pred - P_now);
      _pred_mom = 0.7f * _pred_mom + 0.3f * pred_dir;

      // แก้: ให้ขยับเป็น "เปอร์เซ็นต์เล็กๆ ของ V_f" แทน ไม่ใช่ % ของ kVmppStc เต็มค่า
      const float gain = 0.02f + 0.03f * fminf(fabsf(P_err), 0.6f); // ~2-5%
      Vmpp_pred = _V_f + gain * _V_f * sgn(P_err);   // เปลี่ยนจาก kVmppStc -> _V_f

    } else {
      P_err = 0.0f;
      ai_conf = fmaxf(ai_conf * kAiDecay, 0.3f);
      Vmpp_pred = _V_f;
    }

    // ---- Fusion weights (past vs future) ----
    float w_p, w_f;
    if (ai_conf > 0.7f && P_pred_ok)      { w_p = 0.3f; w_f = 0.7f; }
    else if (ai_conf > 0.5f && P_pred_ok) { w_p = 0.5f; w_f = 0.5f; }
    else                                   { w_p = 0.7f; w_f = 0.3f; }

    float dir_past = (fabsf(avg_gradient) > 0.1f) ? sgn(avg_gradient)
                                                    : sgn(trend_score);
    if (dir_past == 0.0f) dir_past = 1.0f;

    const float dir_future = sgn(Vmpp_pred - _V_f);
    float dir = sgn(w_p * dir_past + w_f * dir_future);
    if (dir == 0.0f) dir = 1.0f;

    if (P_pred_ok && fabsf(P_err) < 0.03f) {
      dir = sgn(dir + 0.35f * sgn(P_pred - P_now));
    }
    if (dir == 0.0f) dir = 1.0f;
    direction = (dir >= 0.0f) ? 1 : -1;

    // ---- Convergence tracking ----
    const bool near_pred = P_pred_ok && fabsf(P_err) < 0.012f;
    const bool near_grad = gradient_stable && (fabsf(avg_gradient) < 0.5f);

    if (near_pred && near_grad) _converge_cnt = min(_converge_cnt + 1, 20);
    else                        _converge_cnt = max(_converge_cnt - 1, 0);
    converged = (_converge_cnt > 12) && (fabsf(avg_gradient) < 0.3f);

    // ---- Step size ----
    float step;
    if (ai_conf > kAiThr && P_pred_ok) {
      const float mag = fminf(fabsf(P_err) * kErrGain, 1.0f);
      step = kDMin + (kDMax - kDMin) * mag;
    } else {
      step = kDMin * 2.0f;
    }
    if (consistency > 0.7f) step *= 1.2f;
    step = fminf(step, kDMax);

    // ---- Momentum ----
    _momentum = kMomK * _momentum + (1.0f - kMomK) * step;

    float step_f;
    if (converged)       step_f = fmaxf(_momentum * 0.30f, kDMin * 3.0f);
    else if (near_pred)  step_f = _momentum * 0.45f;
    else                 step_f = _momentum;

    // ---- Duty update ----
    const float D_raw = D + dir * step_f;

    // Experimental software limits. The old ratio compared two estimates
    // of input voltage and forced the upper limit near kDutyMin.
    // No converter-physics limit is inferred from that ratio.
    float D_next = fminf(fmaxf(D_raw, kDutyMin), kDutyMax);

    // ---- Rate limit ----
    float rate;
    if (converged)   rate = 0.008f;
    else if (is_tr)  rate = kRateF;
    else             rate = kRateN;

    const float dD = D_next - D;
    if (fabsf(dD) > rate) D_next = D + sgn(dD) * rate;

    D_next = fminf(fmaxf(D_next, 0.0f), 1.0f);
    return D_next;
  }

private:
  static const int kHist = 10;

  // ---- constants (ตรงกับ MATLAB ทุกค่า) ----
  static constexpr float kVmppStc = 54.7f;
  static constexpr float kPmppStc = kVmppStc * 5.58f;
  static constexpr float kDutyMin = 0.05f;
  static constexpr float kDutyMax = 0.95f;
  static constexpr float kDMin    = 1.5e-4f;
  static constexpr float kDMax    = 0.02f;
  static constexpr float kAlphaV  = 0.82f;
  static constexpr float kAiThr   = 0.30f;
  static constexpr float kAiDecay = 0.96f;
  static constexpr float kErrGain = 10.0f;
  static constexpr float kMomK    = 0.55f;
  static constexpr float kRateN   = 0.05f;
  static constexpr float kRateF   = 0.10f;

  // ---- persistent state (เทียบเท่า `persistent` ใน MATLAB) ----
  bool  _init        = false;
  float _V_f = 0, _I_f = 0;
  float _P_hist[kHist] = {0};
  float _V_hist[kHist] = {0};
  float _momentum     = 0;
  int   _converge_cnt = 0;
  float _pred_mom     = 0;

  static float sgn(float x) {
    return (x > 0.0f) ? 1.0f : (x < 0.0f) ? -1.0f : 0.0f;
  }

  void initState(float V, float I) {
    _init = true;
    _V_f = V;
    _I_f = I;
    const float p0 = V * I;
    for (int i = 0; i < kHist; i++) {
      _P_hist[i] = p0;
      _V_hist[i] = V;
    }
  }

  void pushHistory(float P_now, float V_f) {
    for (int i = 0; i < kHist - 1; i++) {
      _P_hist[i] = _P_hist[i + 1];
      _V_hist[i] = _V_hist[i + 1];
    }
    _P_hist[kHist - 1] = P_now;
    _V_hist[kHist - 1] = V_f;
  }
};

#endif // MPPT_AI_FUSION_H
