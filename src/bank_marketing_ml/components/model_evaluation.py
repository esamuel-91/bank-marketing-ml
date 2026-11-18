from src.bank_marketing_ml.config.configuration import ModelEvaluationConfig
from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from src.bank_marketing_ml.utils.common import load_arr, load_bin, save_json
import mlflow
import traceback

import numpy as np
import sys


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig) -> None:
        self.config = config

    def initiate_evaluation(self):
        """
        ModelEvaluation is responsible for evaluating the performance
        of the trained classification model using an independent test dataset.

        The evaluation process involves the following steps:

            1. Load the transformed test dataset.
            2. Separate the independent features (X_test) and target
            variable (y_test).
            3. Load the previously trained model from the artifacts directory.
            4. Generate predictions using the loaded model.
            5. Generate prediction probabilities or decision scores
            when supported by the model.
            6. Calculate classification evaluation metrics:
                - Accuracy
                - Precision
                - Recall
                - F1-score
                - ROC-AUC
                - Confusion Matrix
            7. Store the evaluation metrics in a JSON file.
            8. Log the evaluation results and execution status using MLflow.

        Attributes:
            config (ModelEvaluationConfig):
                Configuration object containing the test dataset path,
                trained model path, and evaluation metrics output path.

        Notes:
            - Evaluation is performed on the test dataset, which should
            remain independent of model training and selection.
            - ROC-AUC is calculated using prediction probabilities or
            decision scores.
            - Evaluation metrics are saved as a JSON artifact.
            - MLflow is used for experiment tracking and artifact logging.
        """
        with mlflow.start_run(nested=True, run_name="Evaluating Model"):
            try:
                logger.info("Loading the trained model..")
                model = load_bin(self.config.model_path)

                logger.info("Loading test array for model evaluation...")
                test_arr = load_arr(self.config.test_arr)
                logger.info("Splitting the data into feature and target split..")
                X_test = test_arr[:,:-1]
                y_test = test_arr[:, -1]

                y_pred = model.predict(X_test)

                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, "decision_function"):
                    y_proba = model.decision_function(X_test)
                else:
                    y_proba = None

                accuracy, precision, recall, f1, confi_matrix, roc_auc = (
                    self._eval_model(y_test, y_pred, y_proba)
                )

                evaluation_metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "roc_auc": roc_auc,
                    "Confusion_matrix": str(confi_matrix),
                }

                logger.info(
                    f"Saving the evaluation metrics into the artifacts folder..."
                )
                self.config.metrics_file_path.parent.mkdir(parents=True, exist_ok=True)
                save_json(self.config.metrics_file_path, evaluation_metrics)

                mlflow.log_artifact(
                    str(self.config.metrics_file_path),
                    artifact_path="evaluation_metrics",
                )

            except Exception as e:
                mlflow.set_tag("Evaluation_Status", "Fail")
                mlflow.log_text(
                    traceback.format_exc(), "model_evaluation_traceback.txt"
                )
                logger.exception("Exception occured while evaluating the model.")
                raise BankMarketingException(e, sys) from e

    def _eval_model(self, true, predict, proba=None):
        try:
            accuracy = accuracy_score(true, predict)
            precision = precision_score(true, predict)
            recall = recall_score(true, predict)
            f1 = f1_score(true, predict)
            confi_matrix = confusion_matrix(true, predict)

            if proba is not None:
                roc_auc = roc_auc_score(true, proba)
            else:
                roc_auc = np.nan

            return accuracy, precision, recall, f1, confi_matrix, roc_auc
        except Exception as e:
            raise ValueError("Error occured while evaluating the model..")
