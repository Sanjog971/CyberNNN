# Model card: CyberNN MLP

A short, honest description of the model: what it is, what it was trained on, how well it works, and where it should not be used.

## Model details

| Item | Value |
|---|---|
| Name | CyberNN MLP |
| Type | Feed-forward neural network (multilayer perceptron), binary classifier: BENIGN (0) or ATTACK (1) |
| Architecture | 66 inputs -> Linear 64 -> ReLU -> Linear 32 -> ReLU -> Linear 1 (one raw score; a sigmoid turns it into a 0 to 1 score) |
| Parameters | 6,401 |
| Framework | PyTorch 2.14.1 (CPU build), Python 3.12.10 |
| Trained on | 2026-10-15 (file time of the saved model) |
| Model file | `models/cybernn_mlp.pt` (28.0 KB) with `models/cybernn_mlp.json` (architecture, seed, best epoch, attack weight, validation-chosen threshold) |
| Preprocessing file | `models/preprocessor.json` (feature list, medians, means, scales) |
| Model SHA-256 | `b3f7e364ff9640e144203b10b6e5502669f58a0d540260e328e3c35681ac9c27` |
| Preprocessor SHA-256 | `9f65b81558b7f733b5f4fa7dac0969ae7a7eddf0501bb6c6c77d756e95ac2f0d` |

## Intended use

- Learning and demonstration: a worked example of building, evaluating and serving a flow-based intrusion detector.
- Triage practice: ranking flows in CIC-IDS2017-style CSV files so an analyst reviews the most suspicious first.

## Out of scope

- Production intrusion detection or any setting where a missed attack or a false alarm has real consequences.
- Automatic blocking, quarantining or any other action taken without a human decision.
- Traffic from other networks, other datasets, or flow files produced by a different extraction tool.
- Attack types other than FTP and SSH brute force. The model failed to detect any DDoS flow.

## Training data

- CIC-IDS2017, Tuesday file only (see `dataset.md`): 421,844 flows after removing duplicates, split 70/15/15 with seed 42.
- Training split: 295,290 flows (288,884 benign, 4,153 FTP-Patator, 2,253 SSH-Patator).
- Attack flows are about 2.2% of the data. The loss gives attacks about 45 times the weight of benign flows (45.1, the ratio of benign to attack flows in the training split).

## Inputs and preprocessing

- 66 numeric flow features: packet counts and lengths, inter-arrival times, TCP flag counts, header lengths, bulk and subflow statistics, initial window sizes, and active and idle times. The exact list is `feature_names` in `models/preprocessor.json`.
- `Destination Port` is excluded on purpose, and 10 columns that are constant in the training data are dropped.
- Missing, infinite and non-numeric values are replaced with training-set medians, then each value goes through a signed logarithm and is standardized with the training mean and standard deviation. All of these numbers come from the training split only.

## Training procedure

| Setting | Value |
|---|---|
| Loss | Binary cross-entropy with extra weight on attacks (`pos_weight` 45.1) |
| Optimizer | Adam, learning rate 0.001 |
| Batch size | 512 |
| Epochs | Up to 30, with early stopping (patience 5) on validation loss |
| Result | Stopped at epoch 29; best epoch 24; about 104 seconds on CPU |
| Seed | 42 (a single training run; variation between seeds was not measured) |

## Evaluation

All numbers below were produced by the scripts in this repository.

**Test split** (63,277 flows, 1,373 attacks), each model at the threshold chosen on validation data:

| Model | Threshold | Precision | Recall | F1 | False alarms | Missed attacks | Average precision |
|---|---|---|---|---|---|---|---|
| Logistic regression | 0.95 | 0.950 | 0.964 | 0.957 | 70 | 50 | 0.970 |
| Random forest | 0.28 | 0.997 | 0.999 | 0.998 | 4 | 2 | 1.000 |
| CyberNN MLP | 0.86 | 0.997 | 0.972 | 0.985 | 4 | 38 | 0.989 |

For the MLP at the default threshold 0.5, the test result is precision 0.914, recall 0.985 and F1 0.948, with 127 false alarms and 20 missed attacks. Recall by attack family at threshold 0.86: FTP-Patator 0.993 and SSH-Patator 0.934.

**Generalization tests** (all at fixed settings, no tuning on these data):

| Test | Result |
|---|---|
| Attack type held out of training, threshold 0.5 (validation data) | Unseen FTP-Patator: 271 of 890 detected, 1,540 false alarms. Unseen SSH-Patator: 2 of 483 detected |
| Monday, benign only (502,983 flows) | 99 false alarms at 0.86 (0.20 per 1,000), 986 at 0.5 (1.96 per 1,000) |
| Friday DDoS file (128,016 DDoS flows, 95,096 benign) | **0 DDoS flows detected** at 0.86 and at 0.5; 26 false alarms on the benign flows at 0.86 (0.27 per 1,000) |
| Raw Tuesday file with duplicates (445,909 flows) | Recall 0.730 and precision 0.993 at 0.86; the model trained on part of this file, so the numbers are not an honest estimate |

**Speed:** about 8 ms per 1,000 flows including preprocessing, on CPU.

**Why these metrics.** About 97 to 98% of flows are benign, so accuracy would reward a model that never alerts. Precision, recall, F1, the false-alarm rate and average precision are reported instead.

## Error analysis (validation data, threshold 0.86)

- 31 of 1,373 attacks were missed: 27 SSH-Patator and 4 FTP-Patator. Most of them score between 0.44 and 0.64.
- Caught attacks look like full brute-force sessions (seconds long, many packets each way). Most missed attacks are one-packet-each-way flows lasting microseconds with no payload bytes. This is evidence for the pattern, but the explanation (that those flows are labeled attacks yet look harmless) is a hypothesis.
- 9 false alarms: 5 on destination port 465 with nearly identical flows, 2 microsecond flows on port 389, and 2 others. On Monday, ports 443 and 465 account for 71 of 99 false alarms.

## Limitations and risks

- **It does not generalize to new attack families.** This is the most important limitation, and it was shown on real data (the DDoS result above).
- **Scores are ranking values, not probabilities.** The attack weighting pushes scores upward, which is why the best threshold is 0.86.
- **Training data is narrow:** two attack families from one day, with machine-generated benign traffic. False-alarm rates on a real network are unknown.
- **The dataset has reported errors** (see `dataset.md`), so some labels may be wrong.
- **An attacker who imitates benign flows can evade it.** Adversarial robustness was not tested.
- **Concept drift is not handled.** Traffic changes over time and the model would need retraining and fresh evaluation.
- **Fairness and privacy:** the flow records contain no payloads and, in this version of the dataset, no IP addresses. In a real deployment, flow data can still reveal user behavior and should be handled under the owner's policy.

## Recommendations

- Treat every score as a lead for a human analyst. Never connect the output to automatic blocking.
- Choose the threshold on data that reflects the real alert budget. At 0.5 the model catches more attacks but raises about 32 times more false alarms on the test split (127 against 4).
- Before any real use: retrain on traffic from the target network, train on several attack families, and evaluate on a held-out day.

## How to reproduce

Follow the quick start in the main `README.md`. In short:

```
python scripts\prepare_data.py
python scripts\build_preprocessor.py
python scripts\train_cybernn.py
python scripts\final_test.py
```

Retraining with the same seed should give the same model on the same machine; exact numbers can differ slightly on other hardware or library versions.

## Known issues

- The training function divides by the number of attack rows, so a training set with no attacks fails with an unclear error (it has not occurred in any run).
- The stored threshold prints as 0.8599999999999999 because of floating-point rounding; the dashboard rounds it to 0.86.
- `scripts/train_cybernn.py` keeps its own copy of the training loop that also exists in `src/cybernn/training.py`.
