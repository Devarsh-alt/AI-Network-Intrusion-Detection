import json
import pandas as pd

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import KNOWLEDGE_DIR, REPORTS_DIR


# Both providers expose an OpenAI-compatible API

PROVIDERS = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_variable": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile"
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_variable": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.3-70b-instruct"
    }
}

MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT = """You are the assistant inside an AI-based Network Intrusion \
Detection dashboard. You help a security analyst understand the alerts, the \
attacks, the machine-learning models and their results.

Rules:
- Questions about the traffic or alerts seen in this session (counts, ports, \
times, attack types) must be answered by calling the traffic tools. Never \
guess these numbers.
- Questions about attacks, features, datasets, models or results should be \
answered from the knowledge context below. Call search_knowledge if the \
context does not cover the question.
- If neither the context nor the tools contain the answer, say so plainly.
- Be concise and use plain language. Give numbers exactly as the tools or \
context report them.

Knowledge context:
{context}"""


# ----------------------------------------------------------------------------
# Retrieval: knowledge base built from knowledge/*.md and the saved results
# ----------------------------------------------------------------------------

def _markdown_chunks():

    chunks = []

    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):

        for section in path.read_text().split("\n## ")[1:]:

            title, _, text = section.partition("\n")

            chunks.append({
                "source": path.name,
                "title": title.strip(),
                "text": text.strip()
            })

    return chunks


def _result_chunks():

    chunks = []

    for path in sorted(REPORTS_DIR.glob("*_metrics.json")):

        metrics = json.loads(path.read_text())

        lines = [
            f"Evaluation results of the Random Forest model: {metrics['title']}.",
            f"Test samples: {metrics['test_samples']:,}.",
            f"Accuracy {metrics['accuracy']:.4f}, "
            f"precision {metrics['precision']:.4f}, "
            f"recall {metrics['recall']:.4f}, "
            f"F1 {metrics['f1']:.4f} ({metrics['average']} average), "
            f"weighted F1 {metrics['weighted_f1']:.4f}.",
            "Per-class results:"
        ]

        for name in metrics["labels"]:
            scores = metrics["per_class"][name]
            lines.append(
                f"- {name}: precision {scores['precision']:.4f}, "
                f"recall {scores['recall']:.4f}, "
                f"F1 {scores['f1-score']:.4f}, "
                f"test samples {int(scores['support']):,}"
            )

        chunks.append({
            "source": path.name,
            "title": f"{metrics['title']} model performance, accuracy and metrics",
            "text": "\n".join(lines)
        })

    tables = {
        "cicids_model_comparison.csv":
            "CICIDS2017 comparison of machine learning models "
            "(multi-class, 20 features, class-capped sample)",
        "cicids_feature_selection.csv":
            "CICIDS2017 comparison of feature selection methods "
            "with Random Forest (PCA, importance, Destination Port)",
        "cicids_feature_importance.csv":
            "CICIDS2017 feature importance of the deployed Random Forest",
        "cicids_class_distribution.csv":
            "CICIDS2017 class distribution: rows per label and category after cleaning",
        "nsl_kdd_model_comparison.csv":
            "NSL-KDD comparison of machine learning models (multi-class, KDDTest+)",
        "nsl_kdd_class_distribution.csv":
            "NSL-KDD class distribution in the training and test files"
    }

    for name, title in tables.items():

        path = REPORTS_DIR / name

        if path.exists():
            chunks.append({
                "source": name,
                "title": title,
                "text": pd.read_csv(path).round(4).to_string(index=False)
            })

    return chunks


class KnowledgeBase:

    def __init__(self):

        self.chunks = _markdown_chunks() + _result_chunks()

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        # The title is repeated so that it weighs more than the body
        self.matrix = self.vectorizer.fit_transform(
            f"{chunk['title']}. {chunk['title']}. {chunk['text']}"
            for chunk in self.chunks
        )

    def search(self, query, k=4):

        scores = cosine_similarity(
            self.vectorizer.transform([query]),
            self.matrix
        )[0]

        best = scores.argsort()[::-1][:k]

        return [self.chunks[i] for i in best if scores[i] > 0]


def format_chunks(chunks):

    if not chunks:
        return "No relevant knowledge found."

    return "\n\n".join(
        f"[{chunk['source']}: {chunk['title']}]\n{chunk['text']}"
        for chunk in chunks
    )


# ----------------------------------------------------------------------------
# Tools: fixed queries over the traffic processed in this session
# ----------------------------------------------------------------------------

NO_TRAFFIC = (
    "No traffic has been processed in this session yet. "
    "The user needs to start the Live Monitor first."
)

GROUP_COLUMNS = {
    "attack_type": "Predicted",
    "severity": "Severity",
    "destination_port": "Destination Port",
    "minute": "Minute"
}


def traffic_summary(traffic):

    if traffic is None or traffic.empty:
        return NO_TRAFFIC

    alerts = traffic[traffic["Alert"]]

    summary = {
        "flows_processed": int(len(traffic)),
        "alerts": int(len(alerts)),
        "alert_rate": round(len(alerts) / len(traffic), 4),
        "first_flow_time": str(traffic["Time"].min()),
        "last_flow_time": str(traffic["Time"].max()),
        "predicted_class_counts": traffic["Predicted"].value_counts().to_dict(),
        "alerts_by_severity": alerts["Severity"].value_counts().to_dict(),
        "top_alert_ports": {
            int(port): int(count)
            for port, count in
            alerts["Destination Port"].value_counts().head(5).items()
        }
    }

    if "Category" in traffic.columns:
        summary["live_accuracy_against_true_labels"] = round(
            float((traffic["Predicted"] == traffic["Category"]).mean()), 4
        )

    return json.dumps(summary)


def query_alerts(
    traffic,
    attack_type=None,
    severity=None,
    min_confidence=None,
    destination_port=None,
    last_minutes=None,
    group_by=None,
    limit=10
):

    if traffic is None or traffic.empty:
        return NO_TRAFFIC

    alerts = traffic[traffic["Alert"]]

    if attack_type:
        alerts = alerts[
            alerts["Predicted"].str.lower() == str(attack_type).lower()
        ]

    if severity:
        alerts = alerts[
            alerts["Severity"].str.lower() == str(severity).lower()
        ]

    if min_confidence is not None:
        alerts = alerts[alerts["Confidence"] >= float(min_confidence)]

    if destination_port is not None:
        alerts = alerts[alerts["Destination Port"] == int(destination_port)]

    if last_minutes is not None:
        since = pd.Timestamp.now() - pd.Timedelta(minutes=float(last_minutes))
        alerts = alerts[alerts["Time"] >= since]

    result = {"matching_alerts": int(len(alerts))}

    if group_by in GROUP_COLUMNS and not alerts.empty:

        alerts = alerts.assign(
            Minute=alerts["Time"].dt.strftime("%H:%M")
        )

        counts = alerts[GROUP_COLUMNS[group_by]].value_counts().head(20)

        result["counts"] = {str(key): int(value) for key, value in counts.items()}

    else:

        recent = alerts.sort_values("Time", ascending=False).head(int(limit or 10))

        result["most_recent"] = [
            {
                "time": f"{row['Time']:%H:%M:%S}",
                "attack_type": row["Predicted"],
                "severity": row["Severity"],
                "confidence": round(float(row["Confidence"]), 3),
                "destination_port": int(row["Destination Port"])
            }
            for _, row in recent.iterrows()
        ]

    return json.dumps(result)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "traffic_summary",
            "description": (
                "Overall statistics of the traffic processed in this session: "
                "flows processed, number of alerts, counts per predicted class, "
                "alerts per severity, most targeted ports, live accuracy."
            ),
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_alerts",
            "description": (
                "Filter, count and list the alerts raised in this session. "
                "All filters are optional. Use group_by to get counts per group."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "attack_type": {
                        "type": "string",
                        "enum": [
                            "DoS", "PortScan", "Brute Force",
                            "Botnet", "Web Attack", "Other"
                        ]
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["Low", "Medium", "High", "Critical"]
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Between 0 and 1"
                    },
                    "destination_port": {"type": "integer"},
                    "last_minutes": {
                        "type": "number",
                        "description": "Only alerts from the last N minutes"
                    },
                    "group_by": {
                        "type": "string",
                        "enum": list(GROUP_COLUMNS)
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many recent alerts to list"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search the knowledge base about attack types, flow features, "
                "datasets, models, evaluation results and limitations."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    }
]


def run_tool(name, arguments, knowledge, traffic, sources):

    try:

        arguments = {
            key: value
            for key, value in json.loads(arguments or "{}").items()
            if value is not None
        }

        if name == "traffic_summary":
            return traffic_summary(traffic)

        if name == "query_alerts":
            return query_alerts(traffic, **arguments)

        if name == "search_knowledge":
            chunks = knowledge.search(arguments["query"])
            sources.extend(chunks)
            return format_chunks(chunks)

        return f"Unknown tool: {name}"

    except Exception as error:
        return f"Tool error: {error}"


# ----------------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------------

def get_client(provider, api_key):

    return OpenAI(
        base_url=PROVIDERS[provider]["base_url"],
        api_key=api_key
    )


# Answer the last user message. Returns the reply, the knowledge chunks used
# and the names of the tools that were called.

def answer(client, model, conversation, knowledge, traffic):

    sources = knowledge.search(conversation[-1]["content"])

    tools_used = []

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(context=format_chunks(sources))
        }
    ] + conversation

    for _ in range(MAX_TOOL_ROUNDS):

        message = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            temperature=0.2
        ).choices[0].message

        if not message.tool_calls:
            return message.content or "", sources, tools_used

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments
                    }
                }
                for call in message.tool_calls
            ]
        })

        for call in message.tool_calls:

            tools_used.append(call.function.name)

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": run_tool(
                    call.function.name,
                    call.function.arguments,
                    knowledge,
                    traffic,
                    sources
                )
            })

    return (
        "I could not finish answering within the allowed number of steps.",
        sources,
        tools_used
    )
