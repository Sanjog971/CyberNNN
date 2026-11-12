import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cybernn.evaluation import evaluate_binary
from cybernn.model import load_model, predict_scores
from cybernn.preprocessing import Preprocessor
from cybernn.report import build_report, flagged_export, sha256_bytes, sha256_file
from cybernn.validation import UploadError, check_schema, ground_truth, read_upload, repair_report

MAX_SHOWN = 200
PREFERRED_COLUMNS = ["Destination Port", "Flow Duration", "Total Fwd Packets",
                     "Total Backward Packets", "Flow Bytes/s", "Flow Packets/s", "Label"]

st.set_page_config(page_title="CyberNN", layout="wide")


@st.cache_resource
def load_artifacts():
    pre = Preprocessor.load(ROOT / "models" / "preprocessor.json")
    model, config = load_model(ROOT / "models" / "cybernn_mlp.pt")
    hashes = {"model": sha256_file(ROOT / "models" / "cybernn_mlp.pt"),
              "preprocessor": sha256_file(ROOT / "models" / "preprocessor.json")}
    return pre, model, config, hashes


@st.cache_data
def score_upload(file_bytes):
    df = read_upload(file_bytes)
    pre, model, _, _ = load_artifacts()
    check_schema(df, pre.feature_names)
    report = repair_report(df, pre.feature_names)
    scores = predict_scores(model, pre.transform(df))
    return df, scores, report


pre, model, config, hashes = load_artifacts()
default_threshold = round(float(config["threshold_selected_on_validation"]), 2)

st.title("CyberNN - network flow review")
st.caption("Research prototype for learning and demonstration. Not a production intrusion detection system.")

with st.sidebar:
    st.header("Alert threshold")
    threshold = st.slider("Flag flows with score at or above", 0.05, 0.99, default_threshold, 0.01)
    st.caption(f"Default {default_threshold} was chosen on validation data to maximize F1. "
               "Lower it to catch more attacks at the cost of more false alarms.")
    st.header("Model information")
    st.write(f"Model: `cybernn_mlp.pt` (MLP, hidden layers {config['hidden']})")
    st.write(f"Input features: {config['n_features']} (Destination Port excluded on purpose)")
    st.write(f"Trained on: CIC-IDS2017 Tuesday file only ({pre.n_train_rows:,} training flows)")
    st.write(f"Model file SHA-256: `{hashes['model'][:12]}...`")
    st.write(f"Seed: {config['seed']} | best epoch: {config['best_epoch']} | attack weight: {config['pos_weight']:.1f}")

with st.expander("What these scores mean, and what this tool cannot do", expanded=True):
    st.markdown(
        "- A score is a **ranking value** between 0 and 1, not a probability. The model was trained with extra weight on attacks, "
        "so scores are pushed upward. Higher means the flow looks more like the attack examples it saw in training.\n"
        "- Scores are **investigation leads, not proof** of malicious activity. False positives and false negatives both occur.\n"
        "- The model saw only **two attack types** (FTP-Patator and SSH-Patator brute force) from **one day** of a research dataset "
        "with machine-generated benign traffic. In our tests it failed to detect an attack type it had not seen.\n"
        "- The dataset has known labeling and feature errors reported in published research.\n"
        "- This tool never blocks traffic or takes action. A human decides.")

uploaded = st.file_uploader("Upload a network-flow CSV file", type=["csv"])
if uploaded is None:
    st.info("Waiting for a CSV file with the CIC-IDS2017 flow columns.")
    st.stop()

try:
    with st.spinner("Validating and scoring..."):
        df, scores, report = score_upload(uploaded.getvalue())
except UploadError as error:
    st.error(str(error))
    st.stop()

st.success(f"Scored {len(df):,} flows from `{uploaded.name}`.")
if report["rows_affected"]:
    st.warning(
        f"{report['rows_affected']:,} rows had values that needed repair before scoring "
        f"({report['missing_cells']:,} missing, {report['infinite_cells']:,} infinite, "
        f"{report['non_numeric_cells']:,} non-numeric cells). They were replaced with training-set medians.")

flagged_mask = scores >= threshold
c1, c2, c3 = st.columns(3)
c1.metric("Flows scored", f"{len(df):,}")
c2.metric("Flagged as suspicious", f"{int(flagged_mask.sum()):,}")
c3.metric("Share flagged", f"{flagged_mask.mean() * 100:.2f}%")

st.subheader("Score distribution")
low_count = int((scores < 0.05).sum())
edges = np.linspace(0.05, 1.0, 20)
counts, _ = np.histogram(scores[scores >= 0.05], bins=edges)
labels = [f"{low:.2f}-{low + 0.05:.2f}" for low in edges[:-1]]
st.bar_chart(pd.DataFrame({"flows": counts}, index=labels))
st.caption(f"{low_count:,} flows scored below 0.05 and are not drawn, so the rest stay visible.")

st.subheader(f"Most suspicious flows (top {MAX_SHOWN} at or above {threshold:.2f})")
view = df.copy()
view.insert(0, "score", scores)
view.insert(0, "csv_line", view.index + 2)             # line number in the uploaded file
view = view[flagged_mask].sort_values("score", ascending=False).head(MAX_SHOWN)
shown = ["csv_line", "score"] + [c for c in PREFERRED_COLUMNS if c in view.columns]
if view.empty:
    st.info("No flows reach this threshold.")
else:
    st.dataframe(view[shown], hide_index=True, use_container_width=True,
                 column_config={"score": st.column_config.NumberColumn("score", format="%.6f")})

st.subheader("Evaluation against the file's own labels")
truth = ground_truth(df)
metrics = None
if truth is None:
    st.info("This file has no 'Label' column, so no performance metrics are shown.")
elif truth.nunique() < 2:
    st.info("The labels contain only one class, so precision and recall are not defined for this file.")
else:
    m = evaluate_binary(truth, scores, threshold)
    metrics = m
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Precision", f"{m['precision']:.3f}")
    k2.metric("Recall", f"{m['recall']:.3f}")
    k3.metric("F1", f"{m['f1']:.3f}")
    k4.metric("False-positive rate", f"{m['fpr']:.4f}")
    st.dataframe(pd.DataFrame([[m["TN"], m["FP"]], [m["FN"], m["TP"]]],
                              index=["Actual BENIGN", "Actual ATTACK"],
                              columns=["Predicted BENIGN", "Predicted ATTACK"]))
    st.caption("Computed from this file's Label column at the current threshold. If the file was part of the "
               "training or validation data, these numbers are NOT an honest estimate of real-world performance.")

st.subheader("Export for investigation")
export_df, truncated = flagged_export(df, scores, threshold)
report_md = build_report(
    uploaded.name, sha256_bytes(uploaded.getvalue()), df, scores, threshold, default_threshold, report, metrics,
    {"model_sha256": hashes["model"], "preprocessor_sha256": hashes["preprocessor"],
     "n_features": config["n_features"], "hidden": config["hidden"], "n_train_rows": pre.n_train_rows,
     "seed": config["seed"], "best_epoch": config["best_epoch"]})
e1, e2 = st.columns(2)
e1.download_button("Download flagged flows (CSV)", data=export_df.to_csv(index=False).encode("utf-8"),
                   file_name="cybernn_flagged_flows.csv", mime="text/csv", disabled=export_df.empty)
e2.download_button("Download investigation summary (Markdown)", data=report_md.encode("utf-8"),
                   file_name="cybernn_summary.md", mime="text/markdown")
if truncated:
    st.caption("The CSV is limited to the 50,000 highest-scoring flagged flows.")
st.caption("In the CSV, text cells beginning with = + - or @ get a leading apostrophe so spreadsheets "
           "cannot run them as formulas.")