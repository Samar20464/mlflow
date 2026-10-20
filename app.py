
import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Download the wine-quality dataset
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/"
        "master/tests/datasets/winequality-red.csv"
    )

    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training and test CSV: %s", e
        )
        raise

    # Split the data into training and testing sets
    train, test = train_test_split(data)

    # Separate features and target
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Read hyperparameters from command-line arguments
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        # Train the ElasticNet model
        lr = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=42
        )
        lr.fit(train_x, train_y)

        # Make predictions
        predicted_qualities = lr.predict(test_x)

        # Evaluate the model
        rmse, mae, r2 = eval_metrics(
            test_y, predicted_qualities
        )

        print(
            f"ElasticNet model (alpha={alpha:f}, "
            f"l1_ratio={l1_ratio:f}):"
        )
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        # Track parameters and metrics in MLflow
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Save the trained model
        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)

        tracking_url_type_store = urlparse(
            mlflow.get_tracking_uri()
        ).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr,
                name="model",
                registered_model_name="ElasticnetWineModel",
                signature=signature,
            )
        else:
            mlflow.sklearn.log_model(
                lr,
                artifact_path="model",
                signature=signature,
            )
