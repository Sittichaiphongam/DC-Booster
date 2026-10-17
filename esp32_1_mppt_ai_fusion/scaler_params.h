#ifndef SCALER_PARAMS_H
#define SCALER_PARAMS_H

// Auto-generated from MinMaxScaler fit on columns: ['PLANT_ID', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
// Regenerate this file any time you retrain / change TARGETS / change the CSV.

#define NUM_FEATURES 4
#define SEQ_LEN 10

// Column order -- fill your Arduino raw_row[] in exactly this order
const char* FEATURE_NAMES[NUM_FEATURES] = {
  "PLANT_ID", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"
};

const float FEATURE_MIN[NUM_FEATURES] = {
  4135001.00000000f, 20.39850487f, 18.14041547f, 0.00000000f
};

const float FEATURE_MAX[NUM_FEATURES] = {
  4135001.00000000f, 35.25248613f, 65.54571367f, 1.22165185f
};

// Index of each model OUTPUT within FEATURE_MIN/FEATURE_MAX,
// in the same order as TARGETS / the model's output tensor
#define TARGET_0_IDX 2  // MODULE_TEMPERATURE
#define TARGET_1_IDX 3  // IRRADIATION

#endif // SCALER_PARAMS_H
