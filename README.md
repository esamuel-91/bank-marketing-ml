# Bank Marketing ML

**An End-to-End Machine Learning and MLOps Project for Bank Marketing Subscription Prediction**

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green.svg)](https://xgboost.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Boosting-yellow.svg)](https://catboost.ai/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Pipeline%20Management-purple.svg)](https://dvc.org/)

---

## Project Overview

Bank marketing campaigns are an important way for financial institutions to promote products such as term deposits. However, contacting customers who are unlikely to subscribe can result in inefficient marketing campaigns and increased operational costs.

This project aims to develop a machine learning classification system that predicts whether a customer will subscribe to a term deposit based on historical bank marketing campaign data.

The project follows a modular, end-to-end machine learning architecture, incorporating data processing, model training, evaluation, experiment tracking, and pipeline orchestration.

It also provides a foundation for model deployment and real-time prediction through a dedicated prediction pipeline and web application interface.

### Business Objective

Predict whether a customer will subscribe to a term deposit following a bank marketing campaign.

**Target variable:**

- `yes` – Customer subscribed to the term deposit.
- `no` – Customer did not subscribe to the term deposit.

### Project Goals

- Develop a complete end-to-end machine learning pipeline.
- Implement modular and reusable ML components.
- Perform data ingestion, validation, and transformation.
- Train and compare multiple classification algorithms.
- Select the best-performing model using cross-validation.
- Evaluate model performance using appropriate classification metrics.
- Track experiments and model performance using MLflow.
- Manage pipeline dependencies and reproducibility using DVC.
- Build a prediction pipeline for new customer records.
- Establish a foundation for model deployment and serving.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost, CatBoost |
| Experiment Tracking | MLflow |
| Pipeline Management | DVC |
| Dependency Management | UV |
| Configuration Management | YAML |
| Model Serialization | Joblib / Pickle |
| Data Validation | Python, Pandas |
| Development Environment | Jupyter Notebook |
| Web Application | Python, HTML |
| Containerization | Docker |
| Database Integration | MongoDB (connection module) |
| Cloud Integration | AWS (connection module) |

---

## Project Architecture

The project follows a modular architecture with six sequential stages in the training workflow and a separate prediction pipeline.

### Project Structure

```text
bank-marketing-ml/
│
├── config/
│   ├── config.yaml
│   ├── params.yaml
│   └── schema.yaml
│
├── logs/
│   └── project_skeleton.log
│
├── research/
│   ├── eda.ipynb
│   ├── trials.ipynb
│   └── model_trainer.ipynb
│
├── src/
│   └── bank_marketing_ml/
│       │
│       ├── components/
│       │   ├── data_ingestion.py
│       │   ├── data_validation.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   ├── model_evaluation.py
│       │   └── model_pusher.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   ├── configuration.py
│       │   ├── mongodb_connection.py
│       │   └── aws_connection.py
│       │
│       ├── constants/
│       │   └── __init__.py
│       │
│       ├── entity/
│       │   ├── __init__.py
│       │   └── config_entity.py
│       │
│       ├── exception/
│       │   └── __init__.py
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── training_pipeline.py
│       │   ├── prediction_pipeline.py
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_data_validation.py
│       │   ├── stage_03_data_transformation.py
│       │   ├── stage_04_model_trainer.py
│       │   ├── stage_05_model_evaluation.py
│       │   └── stage_06_model_pusher.py
│       │
│       ├── utils/
│       │   ├── __init__.py
│       │   └── utils_common.py
│       │
│       ├── __init__.py
│       └── logger.py
│
├── templates/
│   ├── form.html
│   ├── index.html
│   └── result.html
│
├── artifacts/
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   ├── model_trainer/
│   └── model_evaluation/
│
├── app.py
├── test.py
├── Dockerfile
├── dvc.yaml
├── requirements.txt
├── requirements_dev.txt
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

*Note: The structure represents the project skeleton and expected organization. Generated artifacts and runtime files are created during execution. Some optional modules may not yet have complete implementations.*

---

## End-to-End Machine Learning Workflow

The training pipeline consists of six stages.

### Stage 1: Data Ingestion

**Component:** `data_ingestion.py`

**Pipeline:** `stage_01_data_ingestion.py`

Responsibilities:

- Load the source bank marketing dataset.
- Perform initial data ingestion.
- Split the dataset into training and testing datasets.
- Support configurable test size and random state.
- Store ingested datasets in the artifacts directory.

**Output:** Ingested training and testing datasets.

### Stage 2: Data Validation

**Component:** `data_validation.py`

**Pipeline:** `stage_02_data_validation.py`

Responsibilities:

- Validate ingested datasets.
- Check missing-value percentages.
- Detect potential data drift.
- Generate validation reports.
- Determine whether the data meets configured validation requirements.

**Output:** Data validation reports and validation status.

### Stage 3: Data Transformation

**Component:** `data_transformation.py`

**Pipeline:** `stage_03_data_transformation.py`

Responsibilities:

- Separate independent features and target variables.
- Encode categorical features.
- Scale numerical features.
- Apply consistent transformations to training and testing datasets.
- Save fitted transformation objects.
- Store transformed datasets for model training.

**Configured techniques:**

- Standard Scaling
- One-Hot Encoding

**Output:** Transformed datasets and reusable preprocessing objects.

### Stage 4: Model Training

**Component:** `model_trainer.py`

**Pipeline:** `stage_04_model_trainer.py`

The project trains and compares multiple machine learning classification algorithms.

**Implemented candidate models:**

1. XGBoost Classifier
2. CatBoost Classifier
3. Gradient Boosting Classifier
4. Random Forest Classifier

Responsibilities:

- Load transformed training data.
- Initialize models using configurable hyperparameters.
- Perform cross-validation.
- Compare model performance.
- Select the best-performing model.
- Train the selected model on the complete training dataset.
- Serialize and save the trained model.
- Track experiments using MLflow.

**Model selection:** Recall is used as the primary cross-validation scoring metric, with supporting metrics recommended for a more complete comparison.

**Output:** Trained model artifact.

### Stage 5: Model Evaluation

**Component:** `model_evaluation.py`

**Pipeline:** `stage_05_model_evaluation.py`

Responsibilities:

- Load the trained model.
- Load the independent test dataset.
- Generate predictions.
- Calculate classification evaluation metrics.
- Save evaluation results in JSON format.
- Log evaluation metrics and artifacts using MLflow.

**Evaluation metrics:**

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall prediction correctness |
| Precision | Measures the reliability of positive predictions |
| Recall | Measures the proportion of actual subscribers identified |
| F1-score | Balances precision and recall |
| ROC-AUC | Measures classification discrimination |
| Confusion Matrix | Shows true and incorrect classifications |

For an imbalanced classification problem, recall, precision, F1-score, and Precision-Recall AUC are particularly useful.

**Output:** Evaluation metrics and confusion matrix.

### Stage 6: Model Pushing

**Component:** `model_pusher.py`

**Pipeline:** `stage_06_model_pusher.py`

This component is intended to handle the movement of the trained model from the training workflow to its model-serving or deployment destination.

Potential responsibilities include:

- Load the selected trained model.
- Package the model and associated preprocessing objects.
- Transfer model artifacts to the configured destination.
- Prepare the model for inference or deployment.

The final implementation depends on the configured model storage and deployment infrastructure.

---

## Prediction Pipeline

**File:** `prediction_pipeline.py`

The prediction pipeline is separate from the model training workflow.

It is designed to generate predictions for new customer records using the trained model and saved preprocessing objects.

Expected workflow:

1. Receive new customer information.
2. Validate and prepare the input features.
3. Load the saved preprocessing object.
4. Transform the input data.
5. Load the trained classification model.
6. Generate subscription predictions.
7. Return the prediction result.

This separation allows model inference to be performed without executing the complete training pipeline.

---

## Experiment Tracking with MLflow

MLflow is used to organize and track machine learning experiments.

The project uses MLflow to track:

- Experiment runs.
- Model-specific training runs.
- Hyperparameters.
- Cross-validation performance.
- Evaluation metrics.
- Model evaluation artifacts.
- Execution status.

### Launch MLflow UI

Run the following command from the project root:

```bash
uv run mlflow ui \
    --backend-store-uri sqlite:///mlflow.db \
    --port 5000
```

Open the MLflow dashboard:

```text
http://127.0.0.1:5000
```

Keep the terminal running while accessing the dashboard.

---

## Pipeline Management with DVC

DVC is used to manage pipeline stages, dependencies, parameters, and generated artifacts.

The pipeline configuration is maintained in:

```text
dvc.yaml
```

### Reproduce the complete pipeline

```bash
uv run dvc repro
```

### Reproduce individual stages

```bash
uv run dvc repro data_ingestion

uv run dvc repro data_validation

uv run dvc repro data_transformation

uv run dvc repro model_trainer

uv run dvc repro model_evaluation

uv run dvc repro model_pusher
```

DVC automatically determines whether upstream stages need to be reproduced based on the declared dependencies and outputs.

### Push DVC-tracked artifacts

After configuring a DVC remote:

```bash
uv run dvc push
```

---

## Configuration Management

The project maintains separate configuration files for pipeline settings, model hyperparameters, and dataset schema definitions.

### 1. config.yaml

Contains configuration for:

- Data ingestion.
- Data validation.
- Data transformation.
- Model training.
- Model evaluation.
- Model pushing.
- Artifact directories and file paths.

### 2. params.yaml

Contains configurable pipeline parameters and model hyperparameters.

Example:

```yaml
INGESTION:
  test_size: 0.25
  random_state: 20
  stratify: true

VALIDATION:
  missing_rate: 0.20
  DRIFT_SHARE_THRESHOLD: 0.50

TRANSFORMATION:
  scaler: standard
  encoder: onehot

TRAINER:
  scoring: recall
  cv: 5
  verbose: 1
  n_jobs: -1
```

Model-specific hyperparameters are maintained under the `TRAINER` section.

### 3. schema.yaml

Defines the expected dataset schema and validation requirements.

It can be used to maintain:

- Expected feature names.
- Data types.
- Target column definitions.
- Schema validation rules.

---

## Installation and Setup

### Prerequisites

- Python 3.12 or a compatible version supported by the project dependencies.
- Git.
- UV package manager.

### 1. Clone the repository

```bash
git clone https://github.com/esamuel-91/bank-marketing-ml.git

cd bank-marketing-ml
```

### 2. Install UV

```bash
pip install uv
```

### 3. Create the virtual environment and install dependencies

```bash
uv sync
```

### 4. Review project configuration

Before executing the pipeline, review:

```text
config/config.yaml
config/params.yaml
config/schema.yaml
```

Update dataset locations, artifact paths, and hyperparameters according to your environment.

---

## Running the Project

### Option 1: Execute the complete training pipeline

```bash
uv run python src/bank_marketing_ml/pipeline/training_pipeline.py all
```

### Option 2: Execute individual pipeline stages

```bash
uv run python src/bank_marketing_ml/pipeline/training_pipeline.py data_ingestion

uv run python src/bank_marketing_ml/pipeline/training_pipeline.py data_validation

uv run python src/bank_marketing_ml/pipeline/training_pipeline.py data_transformation

uv run python src/bank_marketing_ml/pipeline/training_pipeline.py model_trainer

uv run python src/bank_marketing_ml/pipeline/training_pipeline.py model_evaluation

uv run python src/bank_marketing_ml/pipeline/training_pipeline.py model_pusher
```

These commands assume the corresponding stage names are registered in the training pipeline.

### Prediction

The prediction workflow is maintained separately in:

```text
src/bank_marketing_ml/pipeline/prediction_pipeline.py
```

The web application entry point is:

```text
app.py
```

---

## Cross-Validation and Model Selection

The project uses cross-validation to compare candidate classification models.

**Primary scoring metric: Recall**

Recall is particularly relevant when the business objective is to identify as many potential term-deposit subscribers as possible.

Supporting metrics:

- Precision
- F1-score
- ROC-AUC
- Average Precision (PR-AUC)

A Stratified K-Fold strategy can help preserve the class distribution across folds.

The final selected model is evaluated using an independent test dataset.

---

## Web Application

The project includes a web application structure for integrating the prediction workflow.

Relevant files:

```text
app.py

templates/
├── form.html
├── index.html
└── result.html
```

The intended application workflow is:

- Display a customer input interface.
- Collect customer information.
- Pass input to the prediction pipeline.
- Display the model prediction.

---

## Docker

A `Dockerfile` is included in the project to provide a foundation for containerization.

Build the Docker image after configuring the Dockerfile:

```bash
docker build -t bank-marketing-ml .
```

Run the container using the appropriate application port and environment configuration.

---

## Future Improvements

Potential extensions include:

- Hyperparameter optimization using Optuna.
- Model explainability using SHAP.
- Precision-Recall curve visualization.
- Automated model registration and promotion.
- REST API development using FastAPI.
- Model deployment using Docker.
- CI/CD integration using GitHub Actions.
- Model performance monitoring.
- Automated data drift monitoring.
- Scheduled model retraining.
- Cloud-based model artifact storage.

---

## Repository

**GitHub:** https://github.com/esamuel-91/bank-marketing-ml

---

## Author

**Esamuel**

Machine Learning | Data Science | MLOps

---

## License

This project is intended for educational, research, and portfolio purposes.

Add an appropriate open-source license if you intend to distribute or reuse the project publicly.