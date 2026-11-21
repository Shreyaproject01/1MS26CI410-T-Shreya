import os
import json
import time
import platform
import mlflow
import mlflow.pyfunc
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
SEED = 42
MAX_LENGTH = 256
EVAL_FILE = "data/eval_inputs.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model A is the current Champion; Model B is the Challenger.
MODELS = {
    "A_sentiment_distilbert-sst2": {
        "id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "role": "champion_candidate",
    },
    "B_sentiment_bert-sst2": {
        "id": "textattack/bert-base-uncased-SST-2",
        "role": "challenger_candidate",
    },
}

def normalize_label(result):
    label = str(result["label"]).upper()
    if label in ("POSITIVE", "LABEL_1", "1"):
        return 1
    if label in ("NEGATIVE", "LABEL_0", "0"):
        return 0
    raise ValueError(f"Unexpected sentiment label: {result['label']}")

class HuggingFaceSentimentModel(mlflow.pyfunc.PythonModel):
    """A pyfunc wrapper: MLflow callers do not need to call HF pipelines."""
    def __init__(self, model_id):
        self.model_id = model_id
        self.pipe = None

    def load_context(self, context):
        from transformers import pipeline as hf_pipeline
        self.pipe = hf_pipeline("sentiment-analysis", model=self.model_id, device=-1)

    def predict(self, context, model_input):
        texts = model_input["text"].astype(str).tolist()
        raw = self.pipe(texts, truncation=True, max_length=MAX_LENGTH)
        return pd.DataFrame({
            "prediction": [normalize_label(x) for x in raw],
            "label": [x["label"] for x in raw],
            "confidence": [float(x["score"]) for x in raw],
        })

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    data = pd.read_csv(EVAL_FILE)
    data.to_csv(os.path.join(OUTPUT_DIR, "eval_inputs_copy.csv"), index=False)
    y_true = data["label"].astype(int).tolist()
    all_results = {}

    for run_name, cfg in MODELS.items():
        print(f"\nLoading {cfg['id']} ...")
        clf = pipeline("sentiment-analysis", model=cfg["id"], device=-1)
        start = time.perf_counter()
        raw = clf(data["text"].astype(str).tolist(), truncation=True, max_length=MAX_LENGTH)
        elapsed = time.perf_counter() - start
        preds = [normalize_label(x) for x in raw]
        avg_latency_ms = (elapsed / len(data)) * 1000
        metrics = {
            "accuracy": accuracy_score(y_true, preds),
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall": recall_score(y_true, preds, zero_division=0),
            "f1": f1_score(y_true, preds, zero_division=0),
            "avg_latency_ms": avg_latency_ms,
            "total_inference_s": elapsed,
        }
        pred_df = data.copy()
        pred_df["prediction"] = preds
        pred_df["predicted_label"] = ["POSITIVE" if p == 1 else "NEGATIVE" for p in preds]
        pred_df["confidence"] = [float(x["score"]) for x in raw]
        pred_df["correct"] = pred_df["label"] == pred_df["prediction"]
        pred_path = os.path.join(OUTPUT_DIR, f"predictions_{'model_a' if run_name.startswith('A_') else 'model_b'}.csv")
        pred_df.to_csv(pred_path, index=False)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("hf_model_id", cfg["id"])
            mlflow.log_param("task", "sentiment-classification")
            mlflow.log_param("role", cfg["role"])
            mlflow.log_param("max_length", MAX_LENGTH)
            mlflow.log_param("batch_size", 1)
            mlflow.log_param("eval_samples", len(data))
            mlflow.log_param("seed", SEED)
            mlflow.log_param("device", "cpu")
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
            mlflow.set_tag("student_usn", os.getenv("STUDENT_USN", "UPDATE_WITH_YOUR_USN"))
            mlflow.set_tag("dataset_name", "project_movie_review_sample")
            mlflow.set_tag("transformers_version", __import__("transformers").__version__)
            mlflow.set_tag("torch_version", __import__("torch").__version__)
            mlflow.set_tag("mlflow_version", mlflow.__version__)
            mlflow.log_artifact(pred_path)
            metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{'model_a' if run_name.startswith('A_') else 'model_b'}.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            mlflow.log_artifact(metrics_path)
            # Log a loadable model. This may download the HF model again when serving/loading.
            mlflow.pyfunc.log_model(
                artifact_path="sentiment_model",
                python_model=HuggingFaceSentimentModel(cfg["id"]),
                pip_requirements=["mlflow", "transformers", "torch", "pandas"],
            )
            all_results[run_name] = {"run_id": run.info.run_id, "model_id": cfg["id"], **metrics}
            print(f"{run_name}: {metrics}")

        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["NEGATIVE", "POSITIVE"])
        disp.plot(values_format="d")
        plt.title(run_name + " confusion matrix")
        plot_path = os.path.join(OUTPUT_DIR, f"confusion_{'model_a' if run_name.startswith('A_') else 'model_b'}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    a_name, b_name = list(all_results.keys())
    a, b = all_results[a_name], all_results[b_name]
    # Predeclared rule: challenger must improve F1 by >= 0.01; otherwise prefer faster model.
    margin = 0.01
    delta = b["f1"] - a["f1"]
    if delta >= margin:
        winner = b_name
    elif abs(delta) < margin and b["avg_latency_ms"] < a["avg_latency_ms"]:
        winner = b_name
    else:
        winner = a_name

    summary = {
        "primary_metric": "f1",
        "minimum_improvement_margin": margin,
        "direction": "higher_is_better",
        "tie_breaker": "If absolute F1 difference is below 0.01, choose the faster model.",
        "winner": winner,
        "delta_f1_challenger_minus_champion": delta,
        "model_a": a,
        "model_b": b,
    }
    summary_path = os.path.join(OUTPUT_DIR, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    compare_df = pd.DataFrame([
        {"model": "Model A - DistilBERT", **{k: a[k] for k in ["accuracy", "precision", "recall", "f1", "avg_latency_ms"]}},
        {"model": "Model B - BERT", **{k: b[k] for k in ["accuracy", "precision", "recall", "f1", "avg_latency_ms"]}},
    ])
    compare_df.to_csv(os.path.join(OUTPUT_DIR, "comparison.csv"), index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    compare_df.set_index("model")[["accuracy", "precision", "recall", "f1"]].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Champion vs Challenger - Evaluation Metrics")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()

    # Five qualitative disagreement examples.
    pa = pd.read_csv(os.path.join(OUTPUT_DIR, "predictions_model_a.csv"))
    pb = pd.read_csv(os.path.join(OUTPUT_DIR, "predictions_model_b.csv"))
    merged = pa[["text", "label", "prediction"]].merge(
        pb[["text", "prediction"]], on="text", suffixes=("_a", "_b")
    )
    disagreements = merged[merged["prediction_a"] != merged["prediction_b"]].head(5)
    disagreements["comment"] = "Models disagree; inspect the review context and confidence scores."
    disagreements.to_csv(os.path.join(OUTPUT_DIR, "qualitative_disagreements.csv"), index=False)

    with mlflow.start_run(run_name="compare_sentiment") as run:
        mlflow.log_param("primary_metric", "f1")
        mlflow.log_param("margin", margin)
        mlflow.log_metric("delta_primary_metric", delta)
        mlflow.set_tag("winner", winner)
        mlflow.log_artifact(summary_path)
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "comparison.csv"))
        mlflow.log_artifact(chart_path)
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "qualitative_disagreements.csv"))
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "confusion_model_a.png"))
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "confusion_model_b.png"))

    print("\n=== CHAMPION–CHALLENGER DECISION ===")
    print(f"Winner: {winner}")
    print(f"Model B F1 - Model A F1: {delta:.4f}")
    print("Next: run register_champion.py")
    print(f"MLflow UI: {TRACKING_URI}")

if __name__ == "__main__":
    main()
