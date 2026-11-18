from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger
from src.bank_marketing_ml.utils.common import save_bin

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

import numpy as np
import mlflow
import traceback

from src.bank_marketing_ml.config.configuration import ModelTrainerConfig
from src.bank_marketing_ml.utils.common import load_arr
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna
from optuna.samplers import TPESampler
import sys


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config = config

    def create_model(self, trial, model_name):
        if model_name == "XGBClassifier":
            xgb_params = self.config.MODEL_PARAMS.XGBClassifier
            return XGBClassifier(
                n_estimators=trial.suggest_int(
                    "n_estimators",
                    xgb_params.n_estimators.low,
                    xgb_params.n_estimators.high,
                ),
                max_depth=trial.suggest_int(
                    "max_depth", xgb_params.max_depth.low, xgb_params.max_depth.high
                ),
                learning_rate=trial.suggest_float(
                    "learning_rate",
                    xgb_params.learning_rate.low,
                    xgb_params.learning_rate.high,
                    log=xgb_params.learning_rate.log,
                ),
                subsample=trial.suggest_float(
                    "subsample", xgb_params.subsample.low, xgb_params.subsample.high
                ),
                colsample_bytree=trial.suggest_float(
                    "colsample_bytree",
                    xgb_params.colsample_bytree.low,
                    xgb_params.colsample_bytree.high,
                ),
                gamma=trial.suggest_float(
                    "gamma", xgb_params.gamma.low, xgb_params.gamma.high
                ),
                min_child_weight=trial.suggest_int(
                    "min_child_weight",
                    xgb_params.min_child_weight.low,
                    xgb_params.min_child_weight.high,
                ),
                reg_alpha=trial.suggest_float(
                    "reg_alpha",
                    xgb_params.reg_alpha.low,
                    xgb_params.reg_alpha.high,
                    log=xgb_params.reg_alpha.log,
                ),
                reg_lambda=trial.suggest_float(
                    "reg_lambda",
                    xgb_params.reg_lambda.low,
                    xgb_params.reg_lambda.high,
                    log=xgb_params.reg_lambda.log,
                ),
                scale_pos_weight=trial.suggest_float(
                    "scale_pos_weight",
                    xgb_params.scale_pos_weight.low,
                    xgb_params.scale_pos_weight.high,
                ),
                eval_metric="logloss",
                random_state=self.config.MODEL_PARAMS.random_state,
                n_jobs=self.config.MODEL_PARAMS.n_jobs,
            )

        elif model_name == "CatBoostClassifier":
            cat_params = self.config.MODEL_PARAMS.CatBoostClassifier
            return CatBoostClassifier(
                iterations=trial.suggest_int(
                    "iterations", cat_params.iterations.low, cat_params.iterations.high
                ),
                depth=trial.suggest_int(
                    "depth", cat_params.depth.low, cat_params.depth.high
                ),
                learning_rate=trial.suggest_float(
                    "learning_rate",
                    cat_params.learning_rate.low,
                    cat_params.learning_rate.high,
                    log=cat_params.learning_rate.log,
                ),
                l2_leaf_reg=trial.suggest_float(
                    "l2_leaf_reg",
                    cat_params.l2_leaf_reg.low,
                    cat_params.l2_leaf_reg.high,
                ),
                random_strength=trial.suggest_float(
                    "random_strength",
                    cat_params.random_strength.low,
                    cat_params.random_strength.high,
                ),
                bagging_temperature=trial.suggest_float(
                    "bagging_temperature",
                    cat_params.bagging_temperature.low,
                    cat_params.bagging_temperature.high,
                ),
                verbose=False,
                random_seed=self.config.MODEL_PARAMS.random_state,
                thread_count=self.config.MODEL_PARAMS.n_jobs,
            )

        elif model_name == "GradientBoostingClassifier":
            gbc_params = self.config.MODEL_PARAMS.GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=trial.suggest_int(
                    "n_estimators",
                    gbc_params.n_estimators.low,
                    gbc_params.n_estimators.high,
                ),
                learning_rate=trial.suggest_float(
                    "learning_rate",
                    gbc_params.learning_rate.low,
                    gbc_params.learning_rate.high,
                    log=gbc_params.learning_rate.log,
                ),
                max_depth=trial.suggest_int(
                    "max_depth", gbc_params.max_depth.low, gbc_params.max_depth.high
                ),
                subsample=trial.suggest_float(
                    "subsample", gbc_params.subsample.low, gbc_params.subsample.high
                ),
                min_samples_split=trial.suggest_int(
                    "min_samples_split",
                    gbc_params.min_samples_split.low,
                    gbc_params.min_samples_split.high,
                ),
                min_samples_leaf=trial.suggest_int(
                    "min_samples_leaf",
                    gbc_params.min_samples_leaf.low,
                    gbc_params.min_samples_leaf.high,
                ),
                random_state=self.config.MODEL_PARAMS.random_state,
            )
        elif model_name == "RandomForestClassifier":
            rf_params = self.config.MODEL_PARAMS.RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=trial.suggest_int(
                    "n_estimators",
                    rf_params.n_estimators.low,
                    rf_params.n_estimators.high,
                ),
                max_depth=trial.suggest_int(
                    "max_depth", rf_params.max_depth.low, rf_params.max_depth.high
                ),
                min_samples_split=trial.suggest_int(
                    "min_samples_split",
                    rf_params.min_samples_split.low,
                    rf_params.min_samples_split.high,
                ),
                min_samples_leaf=trial.suggest_int(
                    "min_samples_leaf",
                    rf_params.min_samples_leaf.low,
                    rf_params.min_samples_leaf.high,
                ),
                max_features=trial.suggest_categorical(
                    "max_features", rf_params.max_features.choices
                ),
                bootstrap=rf_params.bootstrap,
                class_weight=rf_params.class_weight,
                random_state=self.config.MODEL_PARAMS.random_state,
                n_jobs=self.config.MODEL_PARAMS.n_jobs,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def initiate_trainer(self):
        """
        ModelTrainer handles hyperparameter optimization, model selection,
        and training using Optuna and stratified K-fold cross-validation.

        The component evaluates four classifiers:
            1. XGBClassifier
            2. CatBoostClassifier
            3. GradientBoostingClassifier
            4. RandomForestClassifier

        The training process involves:
            1. Loading transformed training data.
            2. Separating features and target variables.
            3. Optimizing model hyperparameters using Optuna (TPE sampler).
            4. Evaluating models using stratified K-fold cross-validation.
            5. Selecting the best model based on the configured scoring metric.
            6. Retraining the best model on the complete training dataset.
            7. Saving the trained model.
            8. Logging parameters, metrics, and training status using MLflow.

        Attributes:
            config (ModelTrainerConfig): Configuration containing training
                data paths, optimization settings, and model artifact details.
        """

        with mlflow.start_run(nested=True, run_name="Optuna Model Training"):
            try:
                logger.info("Loading training array.")

                train_arr = load_arr(self.config.train_arr)

                X_train = train_arr[:, :-1]
                y_train = train_arr[:, -1]

                logger.info("Initializing StratifiedKFold.")

                cv = StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=self.config.MODEL_PARAMS.random_state,
                )

                model_names = [
                    "XGBClassifier",
                    "CatBoostClassifier",
                    "GradientBoostingClassifier",
                    "RandomForestClassifier",
                ]

                best_score = -1
                best_model = None
                best_model_name = None
                best_params = None

                for model_name in model_names:
                    logger.info(f"Starting Optuna optimization for {model_name}")

                    def objective(trial):

                        model = self.create_model(trial, model_name)

                        scores = cross_val_score(
                            estimator=model,
                            X=X_train,
                            y=y_train,
                            scoring=self.config.MODEL_PARAMS.scoring,
                            cv=cv,
                            n_jobs=self.config.MODEL_PARAMS.n_jobs,
                        )

                        mean_recall = float(np.mean(scores))
                        return mean_recall

                    study = optuna.create_study(
                        direction="maximize", sampler=TPESampler(seed=42)
                    )

                    study.optimize(
                        objective,
                        n_trials=self.config.MODEL_PARAMS.n_trials,
                        show_progress_bar=True,
                    )

                    model_score = study.best_value
                    model_params = study.best_params

                    logger.info(f"{model_name} Best Recall: {model_score:.4f}")

                    logger.info(f"{model_name} Best Parameters: {model_params}")

                    # Log best results for this model
                    with mlflow.start_run(nested=True, run_name=f"{model_name}_Optuna"):
                        mlflow.log_param("model_name", model_name)
                        mlflow.log_param("optimization", "Optuna_TPE")
                        mlflow.log_param("n_trials", 30)
                        mlflow.log_param("cv_folds", 5)
                        mlflow.log_param("scoring_metric", "recall")

                        mlflow.log_params(model_params)

                        mlflow.log_metric("best_cv_recall", model_score)

                        mlflow.set_tag("Optimization_Status", "Success")

                    if model_score > best_score:
                        best_score = model_score
                        best_model_name = model_name
                        best_params = model_params

                        # Create the model using the best parameters
                        best_model = self.create_final_model(model_name, best_params)

                if best_model is None:
                    raise ValueError("Optuna model optimization failed.")

                logger.info(f"Best Model: {best_model_name}")

                logger.info(f"Best Recall: {best_score:.4f}")

                logger.info("Retraining best model on complete training data.")

                best_model.fit(X_train, y_train)

                self.config.model_path.parent.mkdir(parents=True, exist_ok=True)

                save_bin(path_to_bin=self.config.model_path, data=best_model)

                mlflow.log_param("best_model_name", best_model_name)

                mlflow.log_metric("best_cv_recall", best_score)

                mlflow.set_tag("Training_Status", "Success")

                logger.info("Optuna model training completed successfully.")

                return {
                    "model_name": best_model_name,
                    "model_path": self.config.model_path,
                    "cv_recall": best_score,
                    "best_params": best_params,
                }

            except Exception as e:
                mlflow.set_tag("Training_Status", "Fail")

                mlflow.log_text(traceback.format_exc(), "model_trainer_traceback.txt")

                logger.exception("Exception occurred while training the model.")

                raise BankMarketingException(e, sys) from e

    def create_final_model(self, model_name, params):
        if model_name == "XGBClassifier":
            return XGBClassifier(
                **params,
                eval_metric="logloss",
                random_state=self.config.MODEL_PARAMS.random_state,
                n_jobs=1,
            )

        elif model_name == "CatBoostClassifier":
            return CatBoostClassifier(
                **params,
                verbose=False,
                random_seed=self.config.MODEL_PARAMS.random_state,
                thread_count=1,
            )

        elif model_name == "GradientBoostingClassifier":
            return GradientBoostingClassifier(
                **params, random_state=self.config.MODEL_PARAMS.random_state
            )

        elif model_name == "RandomForestClassifier":
            return RandomForestClassifier(
                **params,
                bootstrap=True,
                class_weight="balanced",
                random_state=self.config.MODEL_PARAMS.random_state,
                n_jobs=1,
            )

        else:
            raise ValueError(f"Unknown model: {model_name}")
