"""
Does experiments tracking with MLflow.
"""

import dagshub
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.tracking.artifact_utils import get_artifact_uri
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

# Read and prepare data
df = pd.read_csv('../dataset/TravelInsurancePrediction.csv')

# Clean-up, separate categorical and numeric features, etc
df = df.drop(['Unnamed: 0'], axis=1)

df["ChronicDiseases"] = df["ChronicDiseases"].map({0: "No", 1: "Yes"})
df["TravelInsurance"] = df["TravelInsurance"].map({0: "not purchased", 1: "purchased"})

object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

# Splitting the data into Train and Test, etc
X = df[['AnnualIncome', 'FamilyMembers', 'Age']]
y = df['TravelInsurance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Hyperparameters tuning
gbc = GradientBoostingClassifier()
parameters = {
    "n_estimators": [1, 50],
    "max_depth": [1, 3],
    "learning_rate": [0.01, 0.1]
}

# Use mlflow to log results
DAGSHUB_ACCOUNT = <your-dagshub-account>
REPO_NAME = "my-2nd-repo"
TRACKING_URI = f"https://dagshub.com/{DAGSHUB_ACCOUNT}/{REPO_NAME}.mlflow"

dagshub.init(REPO_NAME, DAGSHUB_ACCOUNT, mlflow=True)
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(experiment_name="my_experiments_23")

mlflow.autolog()
with mlflow.start_run():
    cv = GridSearchCV(gbc, parameters, cv=5)
    cv.fit(X_train, y_train.values.ravel())

# Find the best parameters' set
client = MlflowClient(TRACKING_URI)
candidates = client.search_runs(
    experiment_ids=client.get_experiment_by_name("my_experiments_23").experiment_id,
    max_results=5,
    order_by=["metrics.training_roc_auc DESC"]
)
best_params = candidates[0].data.params
run_id = candidates[0].info.run_id

# Register model, create version and move to 'Staging' state
client.create_registered_model(name="my_model")

source = get_artifact_uri(run_id=run_id, artifact_path="my_model")
versioned_model = client.create_model_version(
    name="my_model",
    source=source,
    run_id=run_id
)

client.transition_model_version_stage(
    name="my_model",
    version=1,
    stage="Staging"
)
