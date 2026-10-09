import pandas as pd

from config import NORMAL


SEVERITY = {
    "PortScan": "Medium",
    "DoS": "High",
    "Brute Force": "High",
    "Web Attack": "High",
    "Botnet": "Critical",
    "Other": "Critical"
}

SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]

# Predictions below this confidence are downgraded to Low severity
LOW_CONFIDENCE = 0.7


# Classify flows and decide which ones raise an alert

def score_flows(model, flows, threshold=0.5):

    probabilities = model.predict_proba(flows[model.feature_names_in_])

    classes = list(model.classes_)

    scored = flows.copy()

    scored["Predicted"] = [classes[i] for i in probabilities.argmax(axis=1)]
    scored["Confidence"] = probabilities.max(axis=1)
    scored["Attack Probability"] = 1 - probabilities[:, classes.index(NORMAL)]

    scored["Alert"] = (
        (scored["Predicted"] != NORMAL)
        & (scored["Attack Probability"] >= threshold)
    )

    scored["Severity"] = scored["Predicted"].map(SEVERITY).fillna("None")

    scored.loc[
        scored["Alert"] & (scored["Confidence"] < LOW_CONFIDENCE),
        "Severity"
    ] = "Low"

    scored.loc[~scored["Alert"], "Severity"] = "None"

    return scored


# Explain one prediction: replace each feature with its typical normal value
# and measure how much the predicted class probability drops

def explain_flow(model, flow, baseline, top=6):

    features = list(model.feature_names_in_)

    row = pd.DataFrame([flow[features]], columns=features).astype(float)

    probabilities = model.predict_proba(row)[0]

    predicted = probabilities.argmax()

    variants = pd.concat([row] * len(features), ignore_index=True)

    for i, feature in enumerate(features):
        variants.loc[i, feature] = baseline[feature]

    changed = model.predict_proba(variants)[:, predicted]

    explanation = pd.DataFrame({
        "Feature": features,
        "Value": row.iloc[0].values,
        "Typical Normal Value": [baseline[f] for f in features],
        "Contribution": probabilities[predicted] - changed
    })

    return (
        explanation
        .sort_values("Contribution", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )


# Markdown incident report built from the processed traffic

def incident_report(history):

    alerts = history[history["Alert"]]

    lines = [
        "# Network Intrusion Detection - Incident Report",
        "",
        f"Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## Summary",
        "",
        f"- Flows analysed: {len(history):,}",
        f"- Alerts raised: {len(alerts):,} "
        f"({len(alerts) / max(len(history), 1):.1%} of traffic)"
    ]

    if alerts.empty:
        return "\n".join(lines + ["", "No alerts were raised."])

    lines += [
        f"- First alert: {alerts['Time'].min():%Y-%m-%d %H:%M:%S}",
        f"- Last alert: {alerts['Time'].max():%Y-%m-%d %H:%M:%S}",
        "",
        "## Alerts by Attack Type",
        "",
        "| Attack Type | Alerts | Mean Confidence | Most Targeted Port |",
        "|---|---:|---:|---:|"
    ]

    for category, group in alerts.groupby("Predicted"):
        lines.append(
            f"| {category} | {len(group):,} | {group['Confidence'].mean():.1%} "
            f"| {int(group['Destination Port'].mode().iloc[0])} |"
        )

    lines += [
        "",
        "## Alerts by Severity",
        "",
        "| Severity | Alerts |",
        "|---|---:|"
    ]

    counts = alerts["Severity"].value_counts()

    for severity in reversed(SEVERITY_ORDER):
        if severity in counts:
            lines.append(f"| {severity} | {counts[severity]:,} |")

    lines += [
        "",
        "## Most Targeted Destination Ports",
        "",
        "| Port | Alerts |",
        "|---:|---:|"
    ]

    for port, count in alerts["Destination Port"].value_counts().head(10).items():
        lines.append(f"| {int(port)} | {count:,} |")

    return "\n".join(lines)
