import pandas as pd
import skops.io as skio
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# Load and Explore Data
org_df = pd.read_csv("Data/loan_data.csv")
org_df = org_df.sample(frac=1)
loan_df = org_df[
    ["previous_loan_defaults_on_file", "person_home_ownership", "loan_status"]
]


# Split Data
X = loan_df.drop("loan_status", axis=1).values
y = loan_df[["loan_status"]].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=45
)

# Preprocess Data
transform = ColumnTransformer(
    [
        ("loan_fault_encoder", OneHotEncoder(drop="first"), [0]),
        ("home_ownership_encoder", OrdinalEncoder(), [1]),
    ]
)

# Tranasform and Process Data
pipe = Pipeline(steps=[("preprocessing", transform), ("model", GaussianNB())])
pipe.fit(X_train, y_train.ravel())


# Model Evaluation
y_pred = pipe.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="micro")

# Save eval results
with open("Results/metrics.txt", "w") as model_eval:
    model_eval.write(f"Accuracy = {accuracy}\nF1-score= {f1}")

matrix_graph = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.savefig("Results/model_conf_matrix.png", dpi=120)


# Save Model
skio.dump(pipe, "Models/loan_approval_pipeline.skops")
