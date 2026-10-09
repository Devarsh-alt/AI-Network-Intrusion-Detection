# The System

## What this project is

This is an AI-based Network Intrusion Detection System (IDS). It uses supervised machine learning to classify network flows as normal or malicious and to name the type of attack. A Streamlit dashboard shows a live traffic monitor, alerts, model results and this assistant. A signature-based IDS matches traffic against known attack patterns and misses new ones; a machine-learning IDS learns statistical patterns of traffic instead.

## Datasets

CICIDS2017 comes from the Canadian Institute for Cybersecurity. It contains five days of traffic (Monday to Friday, July 2017) with about 2.8 million labelled flows and 78 features extracted by CICFlowMeter. Monday is normal traffic only; the other days contain attacks. NSL-KDD is an older benchmark, a cleaned version of KDD Cup 99 with duplicate records removed. It has 41 features per connection, a training file (KDDTrain+) and a separate test file (KDDTest+).

## Preprocessing

For CICIDS2017 the pipeline strips column names, replaces infinite values, drops rows with missing values, removes duplicate rows, and maps each original label to an attack category. About 2.52 million flows remain. All 77 numeric flow features are used and standardised with StandardScaler. The data is split 80% training and 20% testing, stratified by attack category. For NSL-KDD the three text columns (protocol_type, service, flag) are one-hot encoded and the numeric columns are standardised.

## Models

The deployed model is a Random Forest with 100 trees. Two versions are trained on each dataset: a binary model (Normal vs Attack) and a multi-class model that names the attack category. The dashboard uses the CICIDS2017 multi-class model. Naive Bayes, Logistic Regression, K-Nearest Neighbours, Decision Tree and Gradient Boosting are trained as well for comparison.

## Feature selection and why all features are used

The first version of the project used 20 hand-picked features. The feature selection comparison trains the same Random Forest on five feature sets: all 77 features, the top 20 by Random Forest importance, the 20 hand-picked features, the hand-picked features without Destination Port, and 20 PCA components. Overall accuracy is above 99.5 percent for all five, which hides the real difference. The top-20 and hand-picked sets miss most Web Attack flows (F1 around 0.2 instead of 0.98), because the features that separate web attacks from normal traffic have low overall importance and get dropped. PCA keeps web attacks but does worse on Botnet and the rare Other category and takes longer to train. Only the full feature set detects every attack category well, so the deployed model uses all features. The lesson is that accuracy alone is misleading on imbalanced data and per-class F1 has to be checked.

## Why class weights are not used

Balanced class weights were tried to help the rare classes. They raised recall for Botnet and Web Attack but made the model flag thousands of normal flows as those attacks (precision near 10 percent), which would flood an analyst with false alarms. The deployed model therefore uses no class weighting.

## Why NSL-KDD accuracy is much lower than CICIDS2017

On CICIDS2017 the test set is a random 20% of the same traffic the model trained on, so test flows closely resemble training flows and scores are very high. On NSL-KDD the model is trained on KDDTrain+ and tested on KDDTest+, which deliberately contains attack types that never appear in training. Accuracy there is between 74 and 80 percent for every model except Naive Bayes, which is in line with published results. This is the more realistic picture of how an IDS handles unseen attacks, and R2L and U2R attacks are missed most often.

## Alerts and severity

A flow raises an alert when the model predicts an attack category and the attack probability is at or above the alert threshold set in the dashboard. Severity depends on the attack type: Port Scan is Medium, DoS, Brute Force and Web Attack are High, and Botnet and Other are Critical. If the model's confidence is below 70 percent the alert is downgraded to Low.

## How predictions are explained

For a single flow, the dashboard replaces each feature in turn with its typical value in normal traffic (the median) and measures how much the probability of the predicted class drops. Features with the largest drop contributed most to the prediction. This is an occlusion-style explanation and is an approximation: several flow features carry nearly the same information, so replacing one of them alone can understate its importance.

## Live monitor

The live monitor replays held-out test flows from CICIDS2017 as a simulated stream, with bursts of attacks mixed into normal traffic. It does not capture packets from a real network interface. Timestamps are assigned at replay time, and the dataset files used here do not include IP addresses.

## Limitations

The model only recognises attack types present in its training data. CICIDS2017 is from 2017, was generated in a lab, and has known labelling errors documented by later research. A random train/test split gives optimistic scores because flows from the same attack session land in both sets. The rarest categories have too few samples for reliable scores. Flow statistics cannot see packet payloads, so application-layer attacks are harder to detect. Real deployment would need live flow extraction and periodic retraining.
