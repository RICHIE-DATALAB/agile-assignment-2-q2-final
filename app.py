
import mlflow
import mlflow.sklearn

from flask import Flask, jsonify
from prometheus_client import Counter, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from sklearn.datasets import load_iris

app = Flask(__name__)

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

iris = load_iris()
model = None

def load_model():
    global model
    model = mlflow.sklearn.load_model(
        "models:/IrisClassifier/2"
    )

@app.route("/")
def home():
    return jsonify({
        "application": "Iris ML Model",
        "status": "running",
        "model_loaded": model is not None
    })

@app.route("/predict/<int:sample_id>")
def predict(sample_id):
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 503

    if sample_id < 0 or sample_id >= len(iris.data):
        return jsonify({"error": "Invalid sample ID"}), 400

    PREDICTION_REQUESTS.inc()
    result = model.predict([iris.data[sample_id]])[0]

    return jsonify({
        "sample_id": sample_id,
        "predicted_class": iris.target_names[int(result)]
    })

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {
        "Content-Type": CONTENT_TYPE_LATEST
    }

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)
