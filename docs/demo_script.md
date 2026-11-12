# Demo script (about 6 minutes)

A walkthrough you can give live or record. Every number below was produced by this project and can be checked against `evaluation.md`. Say what the numbers show; do not oversell them.

## Before you start

1. Open a terminal in the project folder (the prompt should start with `(.venv)`).
2. Start the dashboard:

```
python -m streamlit run app\cybernn_app.py --server.address localhost --browser.gatherUsageStats false
```

3. Open http://localhost:8501 in a browser window with no personal tabs visible.
4. Keep these files ready: `data\processed\test.csv` and `data\test_uploads\empty.csv`.
5. Have `README.md` open in a second window, scrolled to "Results at a glance".

## The walkthrough

| Time | Do | Say |
|---|---|---|
| 0:00 | Show the README title and the warning box | "CyberNN scores network flows with a small neural network so an analyst can review the most suspicious ones first. It is a research prototype, and it has a limit I will show at the end." |
| 0:30 | Show "Results at a glance" | "On a held-out test set of 63,277 flows, the neural network reaches F1 0.985 with 4 false alarms. A random forest does slightly better, and I report that." |
| 1:00 | Switch to the dashboard; point at the sidebar and the open "What these scores mean" panel | "The page says up front that scores are ranking values, not probabilities, and that the tool never acts on its own. The sidebar shows which model is running, including a fingerprint of the file." |
| 1:45 | Upload `test.csv` | "63,277 flows scored." |
| 2:00 | Point at the yellow repair message | "32 rows had missing values. The tool did not hide that: it counted them and replaced them with training medians." |
| 2:15 | Point at the three headline numbers | "At the default threshold, 1,339 flows are flagged, 2.12% of the file." |
| 2:30 | Show the score chart | "Most flows score near zero. Scores below 0.05 are not drawn so the rest stay visible." |
| 2:45 | Show the flagged-flows table | "Each row shows its line number in the uploaded file, its score and the destination port. The port was not used by the model, but the analyst can see it." |
| 3:00 | Point at the top row, labeled BENIGN with score 1.000000 | "The highest score in this file belongs to a flow labeled benign. It could be a mislabel or a real false alarm; the columns cannot settle it. This is why a score is a lead, not a verdict." |
| 3:30 | Scroll to the evaluation section | "Because this file has labels, the page shows metrics: precision 0.997, recall 0.972. There is a caption saying these numbers are only honest if the file was not used for training." |
| 3:50 | Move the slider to 0.50 | "Lowering the threshold catches more attacks: 1,480 flows are flagged, recall rises to 0.985, but false alarms go from 4 to 127. The threshold is a decision about alert volume, and I chose the default on validation data, not on this test set." |
| 4:30 | Scroll to Export; click both download buttons | "The summary records fingerprints of the input, the model and the preprocessor, so a result can be traced. In the CSV, text beginning with an equals sign gets an apostrophe so a spreadsheet cannot run it as a formula." |
| 5:00 | Upload `empty.csv` | "Bad input gets a clear message. I also tested an unparseable file and a header-only file." |
| 5:15 | Return to the README and show the limitations | "The honest limit: I scored a day of traffic with a DDoS attack the model had never seen. It detected none of the 128,016 DDoS flows. A supervised model learns the attacks it is shown. My next steps are training on several attack families and testing on a held-out day." |
| 5:45 | Close | "Everything here is reproducible from the repository, and 65 automated tests cover the code behind the dashboard." |

## If something goes wrong

- **The page is blank or says it cannot connect:** the dashboard is not running. Start it again with the command above and refresh.
- **A number differs from this script:** you are probably using a different file or threshold. The default threshold is 0.86 and the file is `data\processed\test.csv`.
- **A download seems to do nothing:** check the browser's downloads list (`Ctrl+J`). Files are saved as `cybernn_flagged_flows.csv` and `cybernn_summary.md`.

## Questions to expect

- Why is the destination port not a feature? See `dataset.md`, "The port shortcut".
- Why does the random forest beat the neural network? See `evaluation.md`, section 3.
- How do you know the model does not just memorize? See `evaluation.md`, sections 4 and 5.
- What would you do next? See `limitations.md`, "What would be needed before real use".
