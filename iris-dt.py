import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load the iris dataset
iris = load_iris()
X = iris.data   
y = iris.target

import dagshub
dagshub.init(repo_owner='senorhimanshu', repo_name='mlflow-dagshub-demo', mlflow=True)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# define the parameters for the DecisionTreeClassifier
max_depth = 15

# apply mlflow

mlflow.set_experiment("iris-dt")
# mlflow.set_tracking_uri('http://localhost:5000')
# mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_tracking_uri('https://dagshub.com/senorhimanshu/mlflow-dagshub-demo.mlflow')

# with mlflow.start_run(run_name="himanshu-dt"):    # to name the run
with mlflow.start_run():

    dt = DecisionTreeClassifier(max_depth = max_depth, random_state = 42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # log parameters, metrics, and model
    mlflow.log_metric("accuracy", accuracy) 
    
    mlflow.log_param("max_depth", max_depth)

    # create a confusion matrix plot
    plt.figure(figsize = (6,6))
    sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues", xticklabels = iris.target_names, yticklabels = iris.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    # save the plot as an artifact
    plt.savefig("confusion_matrix.png")

    # mlflow code
    mlflow.log_artifact("confusion_matrix.png")

    # log the current script
    mlflow.log_artifact(__file__)  

    # log the model
    mlflow.sklearn.log_model(dt, name = "decision_tree_model")

    # set tags (helpful for searching and filtering runs)
    mlflow.set_tag("author", "himanshu")
    mlflow.set_tag("model", "DecisionTreeClassifier")

    print("accuracy", accuracy)
    print("confusion matrix\n", cm)
