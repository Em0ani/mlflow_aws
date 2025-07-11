import pandas as pd

import pickle
from sklearn.metrics import accuracy_score

import yaml
import os
import mlflow



os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Em0ani/testrepo.git"
os.environ["MLFLOW_TRACKING_USERNAME"] = 'Em0ani'
os.environ["MLFLOW_TRACKING_PASSWORD"]= "bfba5f616c6d68664e3d5579983197a8946b52b5"

# # load
# train_params = yaml.safe_load(open("params.yaml"))['train']
# dagshub_params = yaml.safe_load(open("params.yaml"))['dagshub']


# def evaluate(data_path,model_path):
#     data =pd.read_csv(data_path)
#     X = data.drop(columns=["Outcome"])
#     y = data["Outcome"]

#     mlflow.set_tracking_uri(dagshub_params["MLFLOW_TRACKING_URI"])

#     model = pickle.load(open(model_path,'rb'))

#     predictions = model.predict(X)
#     accuracy = accuracy_score(y,predictions)

#     mlflow.log_metric("eval_accuracy",accuracy)

#     print(f"Model acccuracy :{accuracy}")

# if __name__=="__main__":
#     evaluate(train_params["data"],train_params["model_path"])


# load
train_params = yaml.safe_load(open("params.yaml"))['train']
dagshub_params = yaml.safe_load(open("params.yaml"))['dagshub']


def evaluate(data_path,model_path):
    data =pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri(dagshub_params["MLFLOW_TRACKING_URI"])

    #? model = pickle.load(open(model_path,'rb'))
    import joblib

    model = joblib.load(model_path)

    predictions = model.predict(X)
    accuracy = accuracy_score(y,predictions)

    mlflow.log_metric("eval_accuracy",accuracy)

    print(f"Model acccuracy :{accuracy}")

if __name__=="__main__":
    evaluate(train_params["data"],train_params["model_path"])
