import os
import json
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError("Experiment not found. Run python evaluate.py first.")

    with open("outputs/metrics_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    winner_name = summary["winner"]
    a = summary["model_a"]
    b = summary["model_b"]
    run_to_model = {
        a["run_id"]: ("champion" if winner_name == "A_sentiment_distilbert-sst2" else "challenger", a["model_id"]),
        b["run_id"]: ("champion" if winner_name == "B_sentiment_bert-sst2" else "challenger", b["model_id"]),
    }

    try:
        client.create_registered_model(REGISTERED_MODEL_NAME)
    except Exception:
        pass

    for run_id, (role, model_id) in run_to_model.items():
        model_uri = f"runs:/{run_id}/sentiment_model"
        mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "hf_model_id", model_id)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "role", role)
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, role, mv.version)
        print(f"Registered version {mv.version}: alias={role}, hf_model_id={model_id}")

    print(f"\nChampion URI: models:/{REGISTERED_MODEL_NAME}@champion")
    print(f"Challenger URI: models:/{REGISTERED_MODEL_NAME}@challenger")

if __name__ == "__main__":
    main()
