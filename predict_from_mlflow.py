import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = "sentiment-model"
INPUT_FILE = "data/new_data.csv"
OUTPUT_FILE = "outputs/new_data_predictions.csv"

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    hf_model_id = champion.tags.get("hf_model_id", "metadata-not-found")
    print(f"Loaded registry model: {MODEL_NAME}")
    print(f"Champion version: {champion.version}")
    print(f"Hugging Face model ID (registry metadata): {hf_model_id}")

    # Important: prediction goes through MLflow only; no HF pipeline is called here.
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
    new_data = pd.read_csv(INPUT_FILE)
    result = model.predict(new_data[["text"]])
    output = new_data.copy()
    output["prediction"] = result["prediction"].astype(int)
    output["sentiment"] = output["prediction"].map({1: "POSITIVE", 0: "NEGATIVE"})
    if "confidence" in result.columns:
        output["confidence"] = result["confidence"]
    output.to_csv(OUTPUT_FILE, index=False)
    print("\nPredictions on NEW data:")
    print(output.to_string(index=False))
    print(f"\nSaved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
