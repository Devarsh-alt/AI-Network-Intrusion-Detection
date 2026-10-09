# AI-Based Network Intrusion Detection Dashboard

A machine-learning Network Intrusion Detection System (IDS) that detects malicious network traffic, names the type of attack, raises alerts on a live dashboard, and answers questions in plain English.

It is trained and evaluated on two benchmark datasets, **CICIDS2017** and **NSL-KDD**.

## Features

- **Attack detection** (Normal vs Attack) and **attack classification** (DoS, Port Scan, Brute Force, Botnet, Web Attack, Other) with a Random Forest
- **Second benchmark**: NSL-KDD with its DoS / Probe / R2L / U2R categories
- **Model comparison**: Naive Bayes, Logistic Regression, K-Nearest Neighbours, Decision Tree, Random Forest, Gradient Boosting
- **Feature selection comparison**: all features, top 20 by importance, hand-picked, PCA
- **Live Monitor**: a simulated traffic stream with real-time predictions, alerts and severity levels
- **Explanations**: which features drove each prediction
- **CSV upload** for batch classification, and **exportable** alert logs and incident reports
- **Assistant**: a retrieval-augmented chatbot that answers questions about the alerts, the attacks, the models and the results

> The Live Monitor replays held-out CICIDS2017 test flows. It does not capture packets from a real network.

## Results

All numbers below were produced by the scripts in `src/` and are saved in `results/reports/`.

### CICIDS2017 (Random Forest, all 77 flow features, 504,160 test flows)

| Task | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Attack detection (binary, Attack class) | 99.85% | 99.62% | 99.50% | 99.56% |
| Attack type (7 classes, macro average) | 99.85% | 97.47% | 90.84% | 93.62% |

| Attack type | Precision | Recall | F1 | Test flows |
|---|---:|---:|---:|---:|
| Normal | 99.90% | 99.92% | 99.91% | 419,012 |
| DoS | 99.89% | 99.81% | 99.85% | 64,352 |
| PortScan | 98.91% | 98.94% | 98.93% | 18,139 |
| Brute Force | 100.00% | 99.84% | 99.92% | 1,830 |
| Botnet | 84.07% | 73.26% | 78.30% | 389 |
| Web Attack | 99.52% | 97.44% | 98.47% | 429 |
| Other | 100.00% | 66.67% | 80.00% | 9 |

Botnet is the weakest large category: about a quarter of botnet flows are classified as normal. "Other" (Infiltration and Heartbleed) has only 9 test flows, so its score is not reliable.

### Feature selection (Random Forest, full CICIDS2017, same split)

| Feature set | Accuracy | Macro F1 | Botnet F1 | Web Attack F1 |
|---|---:|---:|---:|---:|
| All features (77) | 99.85% | 93.62% | 78.30% | 98.47% |
| PCA (20 components) | 99.82% | 83.33% | 67.72% | 97.76% |
| Top 20 by Random Forest importance | 99.76% | 82.95% | 77.16% | 24.46% |
| Hand-picked (20) | 99.63% | 79.26% | 56.32% | 21.13% |
| Hand-picked without Destination Port (19) | 99.57% | 78.80% | 54.86% | 20.03% |

Accuracy is above 99.5% for every feature set, but the reduced sets miss most web attacks. This is why the deployed model uses all features, and why per-class F1 matters more than accuracy on imbalanced data.

### Model comparison on CICIDS2017 (sample capped at 60,000 flows per category)

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Random Forest | 99.81% | 98.18% |
| Decision Tree | 99.80% | 97.08% |
| K-Nearest Neighbours | 99.39% | 92.06% |
| Gradient Boosting | 99.33% | 85.25% |
| Logistic Regression | 96.82% | 82.16% |
| Naive Bayes | 71.34% | 55.31% |

### NSL-KDD (trained on KDDTrain+, tested on KDDTest+)

| Model (5 classes) | Accuracy | Macro F1 |
|---|---:|---:|
| Gradient Boosting | 79.83% | 60.10% |
| K-Nearest Neighbours | 76.60% | 55.50% |
| Logistic Regression | 76.15% | 57.01% |
| Decision Tree | 76.00% | 57.41% |
| Random Forest | 74.10% | 50.83% |
| Naive Bayes | 47.30% | 32.99% |

Random Forest binary detection on NSL-KDD: 77.95% accuracy, 96.89% precision, 63.29% recall.

NSL-KDD scores are much lower because KDDTest+ contains attack types that never appear in the training file. R2L and U2R attacks are missed most often. This is a more realistic picture of how an IDS handles unseen attacks than the CICIDS2017 random split.

## How to Run

### 1. Clone and install

```bash
git clone https://github.com/Devarsh-alt/AI-Network-Intrusion-Detection.git
```

```bash
cd AI-Network-Intrusion-Detection
```

Python 3.9 or newer is required. Create and activate a virtual environment.

Linux / macOS:

```bash
python3 -m venv venv && source venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Get the datasets

The datasets are not in the repository because of their size.

**CICIDS2017**: download the "MachineLearningCSV" archive (8 CSV files, about 885 MB unzipped) from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) or a mirror such as Kaggle, and place the CSV files in `data/raw/`.

**NSL-KDD**: download `KDDTrain+.txt` and `KDDTest+.txt` (from the [CIC NSL-KDD page](https://www.unb.ca/cic/datasets/nsl.html) or a mirror) and place them in `data/nsl_kdd/`.

```text
data/
├── raw/
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   └── ... (8 files)
└── nsl_kdd/
    ├── KDDTrain+.txt
    └── KDDTest+.txt
```

### 3. Train and evaluate

```bash
python src/train_model.py
```

Trains the CICIDS2017 binary and multi-class models, evaluates them, and builds the traffic stream for the dashboard. Takes about 5 minutes on a 12-core laptop and needs several GB of free RAM, because the full dataset is loaded into memory.

```bash
python src/train_nsl_kdd.py
```

Trains and compares all models on NSL-KDD. Takes under a minute.

```bash
python src/compare_models.py
```

Runs the CICIDS2017 model comparison and feature selection comparison. Takes about 10 minutes. The dashboard works without it, but those two sections will be missing.

Trained models go to `models/` and results to `results/`. To re-evaluate saved models without retraining, run `python src/evaluate.py`.

### 4. Run the dashboard

```bash
streamlit run app/app.py
```

Open http://localhost:8501.

### 5. Set up the assistant (optional)

The assistant uses an LLM through [Groq](https://console.groq.com) or [OpenRouter](https://openrouter.ai). Create an API key with either provider, then copy `.env.example` to `.env` and add the key:

```text
GROQ_API_KEY=your_key_here
```

You can also paste the key into the Assistant page. `.env` is ignored by git, so never commit a key anywhere else. Without a key, the assistant still works in a limited mode that shows the most relevant knowledge base passages.

## Dashboard Pages

| Page | What it shows |
|---|---|
| Overview | Headline results for both datasets and the class distribution |
| Live Monitor | Simulated traffic stream, alerts per second, severity, alert log, CSV and incident report export |
| Investigate | Explanation of any alert, a single-flow tester with editable values, CSV upload |
| Model Performance | Metrics, confusion matrix, per-class results, model and feature selection comparisons |
| Assistant | Chatbot for questions about the session's alerts and the project |

## How the Assistant Works

The assistant combines two techniques:

1. **Retrieval (RAG)**: the files in `knowledge/` and the saved results in `results/reports/` are split into passages and indexed with TF-IDF. The passages most relevant to each question are given to the LLM as context, and listed under "Sources" in the chat.
2. **Tools**: questions about the traffic in the current session ("how many DoS alerts?", "which ports were targeted?") are answered by the LLM calling fixed query functions over the alert log, so counts are computed and not guessed.

To extend what the assistant knows, add a `##` section to a file in `knowledge/`.

## Alerts

A flow raises an alert when the model classifies it as an attack and its attack probability is at or above the threshold set in the sidebar.

| Attack type | Severity |
|---|---|
| PortScan | Medium |
| DoS, Brute Force, Web Attack | High |
| Botnet, Other | Critical |

Alerts with model confidence below 70% are downgraded to Low.

## Project Structure

```text
AI-Network-Intrusion-Detection/
├── app/
│   └── app.py                 Streamlit dashboard
├── knowledge/                 Knowledge base for the assistant
├── results/
│   ├── figures/               Confusion matrices
│   └── reports/               Metrics, reports and comparison tables
├── src/
│   ├── config.py              Paths, feature lists and label mappings
│   ├── data_loader.py         Dataset exploration script
│   ├── preprocessing.py       Loading and cleaning for both datasets
│   ├── train_model.py         CICIDS2017 training
│   ├── train_nsl_kdd.py       NSL-KDD training and model comparison
│   ├── compare_models.py      CICIDS2017 model and feature comparisons
│   ├── evaluate.py            Metrics, reports and confusion matrices
│   ├── alerts.py              Alert scoring, explanations, incident report
│   └── assistant.py           Retrieval, tools and LLM chat
├── .env.example
├── requirements.txt
└── README.md
```

## Limitations

- The Live Monitor is a simulation. Real deployment needs live flow extraction (for example with CICFlowMeter) feeding the model.
- The model only recognises attack types present in its training data, as the NSL-KDD results show.
- CICIDS2017 is lab-generated traffic from 2017 with known labelling errors, and a random train/test split gives optimistic scores because flows from the same attack session land in both sets.
- Botnet recall is 73%, and the rarest categories have too few samples for reliable scores.
- Prediction explanations are an approximation (occlusion against typical normal values).

## Technologies

Python, Pandas, NumPy, Scikit-learn, Streamlit, Plotly, Matplotlib, Seaborn, Joblib, and the OpenAI-compatible APIs of Groq and OpenRouter.

## Team


