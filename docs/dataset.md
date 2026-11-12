# Dataset notes

## Source and license

CyberNN uses **CIC-IDS2017** from the Canadian Institute for Cybersecurity: https://www.unb.ca/cic/datasets/ids-2017.html

The dataset page describes labeled network flows, packet captures and CSV files for machine learning, publicly available for researchers, and asks users to cite the paper below. **The data is not included in this repository**; download it from the source and follow its terms.

> Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018.

## How the data was produced (from the dataset page)

- Five days of captured traffic, Monday July 3 to Friday July 7, 2017.
- Benign traffic comes from profiles of 25 simulated users of HTTP, HTTPS, FTP, SSH and email. It is machine-generated, not traffic from real people.
- Flows were extracted with CICFlowMeter; the page says they are labeled based on the time stamp, source and destination IP addresses and ports, protocol and attack.

| Day | Content |
|---|---|
| Monday | Benign traffic only |
| Tuesday | FTP-Patator (9:20 to 10:20) and SSH-Patator (14:00 to 15:00), both brute force |
| Wednesday | DoS (slowloris, Slowhttptest, Hulk, GoldenEye) and Heartbleed |
| Thursday | Web attacks (brute force, XSS, SQL injection) and infiltration |
| Friday | Botnet (ARES), port scan, and DDoS (named LOIT on the dataset page) |

## What was downloaded and used

`MachineLearningCSV.zip` (235,102,953 bytes) holds eight CSV files in a folder named `MachineLearningCVE`. The packet captures (7.8 to 13 GB per day) were not used.

| File | Bytes | Raw rows | Use in CyberNN |
|---|---|---|---|
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 135,078,995 | 445,909 | Training, validation and test |
| `Monday-WorkingHours.pcap_ISCX.csv` | 176,927,918 | 529,918 | Fresh benign data (false-alarm rate) |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 77,123,859 | 225,745 | A real unseen attack family |

## Schema facts (Tuesday file)

- 79 columns: 78 numeric flow features and a `Label` column.
- Most column names have a **hidden leading space** (for example ` Destination Port`, ` Label`); a few do not. All names are stripped on load.
- There are **no IP address, flow ID or timestamp columns** in this version. The only identifier-like column is `Destination Port`.
- One column, ` Fwd Header Length.1`, is an exact copy of ` Fwd Header Length` and is dropped (after checking that it is identical).
- 10 columns are constant in the Tuesday data and are dropped: `Bwd PSH Flags`, `Fwd URG Flags`, `Bwd URG Flags`, `CWE Flag Count`, `Fwd Avg Bytes/Bulk`, `Fwd Avg Packets/Bulk`, `Fwd Avg Bulk Rate`, `Bwd Avg Bytes/Bulk`, `Bwd Avg Packets/Bulk`, `Bwd Avg Bulk Rate`.
- 10 columns contain negative values (including `Flow Duration`, `Flow IAT Min`, `Fwd Header Length`), which is physically impossible and probably comes from the flow-extraction tool. 34 columns have values above 1,000,000.

## Data quality found in the Tuesday file

| Finding | Count |
|---|---|
| Missing values | 201, all in `Flow Bytes/s` |
| Infinite values | 327 (63 in `Flow Bytes/s`, 264 in `Flow Packets/s`) in 264 rows |
| Exact duplicate rows | 24,065 (benign 19,382; SSH-Patator 2,678; FTP-Patator 2,005) |
| Identical features with different labels | 0 (exact matches only) |

Duplicates are much more common among attacks (about 45% of SSH-Patator rows and 25% of FTP-Patator rows, against about 4.5% of benign rows), because brute-force tools repeat nearly identical connection attempts.

## Cleaning and splitting

1. Strip column names; drop the copied column; remove exact duplicate rows; turn infinite values into missing values; add `is_attack` (0 for BENIGN, 1 for any attack).
2. Split 70/15/15, stratified on the original label, with seed 42.
3. Fit the preprocessor (constant columns, medians, signed log, standardization) on the training split only.

| Split | Rows | BENIGN | FTP-Patator | SSH-Patator |
|---|---|---|---|---|
| Cleaned Tuesday (total) | 421,844 | 412,692 | 5,933 | 3,219 |
| Train | 295,290 | 288,884 | 4,153 | 2,253 |
| Validation | 63,277 | 61,904 | 890 | 483 |
| Test | 63,277 | 61,904 | 890 | 483 |

Attacks are about 2.2% of the cleaned data. The cleaned Monday file has 502,983 rows (all benign), and the cleaned Friday DDoS file has 223,112 rows (95,096 benign, 128,016 DDoS).

## The port shortcut

In the Tuesday file, 7,937 of 7,938 FTP-Patator flows use destination port 21 (one uses port 80) and all 5,897 SSH-Patator flows use port 22. Benign traffic also uses port 22 (2,136 flows). A model that can see the port can score well by memorizing it, which would not carry over to a real network. CyberNN therefore excludes `Destination Port` from the model; the dashboard still shows it next to every flagged flow.

## Known problems with this dataset

Independent research groups have studied CIC-IDS2017 and reported labeling flaws, flow-construction and feature-extraction problems, and capture artifacts such as packet duplication and misordering. This project consulted summaries of these papers, not the full texts, and did not check which problems affect the files used here:

- "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study"
- "Errors in the CICIDS2017 dataset and the significant differences in detection performances it makes"
- "Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018"

Observations from this project that fit those reports (none verified as errors):

- Most attacks the model misses are one-packet-each-way flows lasting microseconds, with no payload bytes. They are labeled attacks but look like harmless traffic.
- Some micro-flows produce absurd rates (up to about 7,000,000 bytes per second).
- A benign-labeled flow on port 137 receives the highest possible score, and one FTP-Patator flow uses port 80.

## How to get the data

1. Download `MachineLearningCSV.zip` from the dataset page and save it in `data\raw`.
2. Extract only the files you need, for example:

```
tar -xf data\raw\MachineLearningCSV.zip -C data\raw MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv
```

3. Run `python scripts\prepare_data.py` to produce the splits in `data\processed`.
