"""
Builds model based on experiments metrics and serializes it as .joblib file.
"""

import dagshub
import mlflow
import pandas as pd
from joblib import dump
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DAGSHUB_ACCOUNT = <your-dagshub-account>
REPO_NAME = "my-2nd-repo"
TRACKING_URI = f"https://dagshub.com/{DAGSHUB_ACCOUNT}/{REPO_NAME}.mlflow"

dagshub.init(REPO_NAME, DAGSHUB_ACCOUNT, mlflow=True)
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(TRACKING_URI)

# Find the best parameters' set
candidates = client.search_runs(
    experiment_ids=client.get_experiment_by_name("my_experiments_23").experiment_id,
    max_results=5,
    order_by=["metrics.training_roc_auc DESC"]
)
best_params = candidates[0].data.params

# Build model using best_params

df = pd.read_csv('../dataset/TravelInsurancePrediction.csv')

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

gbc = GradientBoostingClassifier(
    learning_rate=float(best_params['best_learning_rate']),
    max_depth=int(best_params['best_max_depth']),
    n_estimators=int(best_params['best_n_estimators'])
)
gbc.fit(X_train, y_train)

# Serialize model
dump(gbc, '../deployment_docker/my_model.joblib')
