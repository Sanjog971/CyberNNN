# Limitations and known failure cases

Ranked by how much they should affect your trust in CyberNN. Each item says what the evidence is and what would reduce the problem. "Hypothesis" marks explanations that were not verified.

## Ranked limitations

| # | Limitation | Evidence | What would reduce it |
|---|---|---|---|
| 1 | **It does not detect attack families it was not trained on** | 0 of 128,016 Friday DDoS flows detected at thresholds 0.86 and 0.5. With one attack type held out of training, the MLP found 271 of 890 FTP-Patator flows and 2 of 483 SSH-Patator flows | Train on several attack families; evaluate by holding out a whole day or family; try a model trained on benign traffic only |
| 2 | **Narrow, synthetic training data** | One day, two attack families, benign traffic generated from 25 simulated users | More days and families; data from a network you own, collected with permission |
| 3 | **The dataset has reported labeling and feature errors** | Published research (summaries consulted, not full texts); odd flows seen here: a benign-labeled flow with a perfect score, one FTP-Patator flow on port 80 | Audit labels; evaluate on a corrected version of the dataset |
| 4 | **Random splits are optimistic** | Flows from one attack session are near-identical, so a test flow can have a twin in training. On the raw Tuesday file, with duplicates, recall is 0.730 against 0.972 on the de-duplicated test split | Group or time-based splits; keep duplicates out of every split |
| 5 | **Many missed attacks are fragments** | 31 of 1,373 validation attacks missed; the typical miss is a one-packet-each-way flow lasting microseconds with no payload. Hypothesis: they are labeled attacks but look harmless | Label audit; features or context that separate fragments from harmless tiny flows |
| 6 | **A few benign behaviors cause most false alarms** | Monday: ports 443 (37) and 465 (34) account for 71 of 99 false alarms. Port 465 recurs on all three days | Add such benign sessions to training and check on fresh data; let analysts suppress alerts using the port shown in the dashboard |
| 7 | **Scores are ranking values and the threshold is a trade-off** | Best F1 threshold 0.86. On the test split, 0.5 gives 127 false alarms and 20 misses; 0.86 gives 4 false alarms and 38 misses | Choose the threshold against a real alert budget; calibrate scores if probabilities are needed |
| 8 | **One training run, no confidence intervals** | Seed 42 only; validation loss ranged from 0.032 to 0.059 between epochs; differences of a few flows are not meaningful | Repeat with several seeds and report the spread |
| 9 | **Hidden proxies for the port may exist** | The port is excluded, but other features may fingerprint the protocol. Not tested | Compare against a model trained with protocol-neutral features |
| 10 | **No test against evasion or drift** | Adversarial robustness and changing traffic over time were not evaluated | Adversarial testing; time-ordered evaluation; scheduled retraining |
| 11 | **Operational scope is small** | The dashboard is local and single-user (bound to localhost, no login); the API is loopback-only by default and uses one shared key. Neither keeps an audit log | Full identity/authorization, audit logging and monitoring before any shared deployment |

## Concrete failure cases

| Case | What happens | Example from this project |
|---|---|---|
| An unseen attack family | It is scored as normal | Friday DDoS: median score below 0.00005 |
| An attack that is a tiny fragment | It is missed | SSH-Patator misses: median duration 97 microseconds, 1 packet each way, no payload bytes |
| A full attack session scored very low | It is missed | One SSH-Patator session in the validation file (row index 44,864, zero-based; 20 packets forward, 18 backward, 2.1 s) scored 0.013; it deserves a manual look |
| Benign email-like sessions | They are flagged | Five benign flows on port 465 (about 15 packets forward, 21 backward, 2 s) scored 0.93 to 0.99 |
| Benign microsecond flows with extreme rates | They are flagged | Two flows on port 389 with rates up to about 7,000,000 bytes per second |
| A flow with the highest possible score labeled benign | A false alarm, or a mislabeled attack | One flow on port 137 (3 packets each way, about 3.2 s) scored 1.000000; it cannot be settled from these columns |
| Repeated identical flows | File-level metrics are skewed | The raw Tuesday file gives recall 0.730 against 0.972 after removing duplicates |

## Known code issues

- The training function divides by the number of attack rows, so a training set with no attacks fails with an unclear error. It has not occurred in any run.
- The stored threshold prints as 0.8599999999999999 because of floating-point rounding; the dashboard rounds it to 0.86.
- `scripts/train_cybernn.py` keeps its own copy of the training loop that also exists in `src/cybernn/training.py`.
- The scripts and the dashboard page are not covered by automated tests; the page was checked by hand in the browser.
- In one screenshot at threshold 0.50 the last bar of the score histogram looked clipped. The chart does not depend on the slider, so this was most likely a display glitch; it was not investigated.
- Instructions were tested on Windows with PowerShell only.

## What would be needed before real use

1. Train on traffic from the target network and on several attack families.
2. Evaluate on a held-out day and on an attack family the model has not seen.
3. Choose the threshold against a real alert budget, and measure the false-alarm rate on that network.
4. Repeat training with several seeds and report the spread.
5. Add full identity/authorization, audit logging and monitoring (the API currently uses a shared key only), and keep a human in the loop for every decision.

## Not done in this project

The UNSW-NB15 dataset, the other days of CIC-IDS2017 (Wednesday, Thursday and the rest of Friday), multiple seeds, a benign-only model for unseen attacks, and adversarial testing.
