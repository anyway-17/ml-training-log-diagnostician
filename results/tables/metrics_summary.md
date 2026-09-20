# Evaluation Results

Total runs evaluated: **23**

## Detection (healthy vs. problem)

- Precision: **100.0%**
- Recall: **66.7%**
- False positive rate: **0.0%**
- Confusion: TP=12, FP=0, TN=5, FN=6

## Diagnosis accuracy (for correctly-flagged problem runs' specific type)

- Strict accuracy: **66.7%** (12/18)

## Breakdown by problem type

| Type | Correct | Total | Accuracy |
|---|---|---|---|
| healthy | 5 | 5 | 100.0% |
| lr_too_high | 3 | 3 | 100.0% |
| lr_too_low | 0 | 3 | 0.0% |
| overfitting | 3 | 3 | 100.0% |
| vanishing_gradients | 3 | 3 | 100.0% |
| label_noise | 3 | 3 | 100.0% |
| frozen_layer | 0 | 3 | 0.0% |

## Efficiency

- Avg tool calls per run: **3.0**
- Range: 2–6

## Per-run results

| run_id | true label | true type | detected? | diagnosis | correct? | confidence | calls |
|---|---|---|---|---|---|---|---|
| frozen_layer_01 | problem | frozen_layer | False | healthy | ❌ | high | 5 |
| frozen_layer_02 | problem | frozen_layer | False | healthy | ❌ | high | 6 |
| frozen_layer_03 | problem | frozen_layer | False | healthy | ❌ | high | 2 |
| healthy_01 | healthy | none | False | healthy | ✅ | high | 5 |
| healthy_02 | healthy | none | False | healthy | ✅ | high | 2 |
| healthy_03 | healthy | none | False | healthy | ✅ | high | 2 |
| healthy_04 | healthy | none | False | healthy | ✅ | high | 2 |
| healthy_05 | healthy | none | False | healthy | ✅ | high | 2 |
| label_noise_01 | problem | label_noise | True | label_noise | ✅ | medium | 3 |
| label_noise_02 | problem | label_noise | True | label_noise | ✅ | high | 2 |
| label_noise_03 | problem | label_noise | True | label_noise | ✅ | high | 2 |
| lr_high_01 | problem | lr_too_high | True | lr_too_high | ✅ | high | 2 |
| lr_high_02 | problem | lr_too_high | True | lr_too_high | ✅ | high | 5 |
| lr_high_03 | problem | lr_too_high | True | lr_too_high | ✅ | high | 5 |
| lr_low_01 | problem | lr_too_low | False | healthy | ❌ | high | 2 |
| lr_low_02 | problem | lr_too_low | False | healthy | ❌ | high | 2 |
| lr_low_03 | problem | lr_too_low | False | healthy | ❌ | high | 2 |
| overfit_01 | problem | overfitting | True | overfitting | ✅ | high | 2 |
| overfit_02 | problem | overfitting | True | overfitting | ✅ | high | 2 |
| overfit_03 | problem | overfitting | True | overfitting | ✅ | high | 2 |
| vanish_grad_01 | problem | vanishing_gradients | True | vanishing_gradients | ✅ | high | 6 |
| vanish_grad_02 | problem | vanishing_gradients | True | vanishing_gradients | ✅ | high | 2 |
| vanish_grad_03 | problem | vanishing_gradients | True | vanishing_gradients | ✅ | high | 3 |