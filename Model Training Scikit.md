# Model Training using MLflow and Scikit Learn

Read cleaned spark dataframe into a databricks notebook. Next, convert the spark dataframe into pandas 

```
#Convert Spark Dataframe to Pandas
import numpy as np
import pandas as pd
# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

pdf = df.toPandas() 
pdf.set_index("IDColumn", inplace=True)
pdf.head()
```

### Train-Test Split

```
from sklearn.model_selection import train_test_split
from collections import Counter

#Select ID, Input and Target
#Identifier_col = pdf["IDColumn"]
y = pdf["target"]
X = pdf.drop(["target", "DateKey"], axis=1)

# split 80/20 train-test : stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.3, stratify=y)

print("Shape of Train X: ", X_train.shape, " Shape of Train y: ", y_train.shape)
print("Shape of Test X: ", X_test.shape, " Shape of Test y: ", y_test.shape)
print("Counter for y_train")
print(Counter(y_train))
print("Counter for y_test")
print(Counter(y_test))
```

```
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.models.signature import infer_signature
mlflow.sklearn.autolog()

import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, classification_report, accuracy_score, precision_score, recall_score
import sklearn.model_selection
import sklearn.ensemble

# These commands are only required if you are using a cluster running DBR 7.3 LTS ML or below. 
import cloudpickle
assert cloudpickle.__version__ >= "1.4.0", "Update the cloudpickle library using `%pip install --upgrade cloudpickle`"
```

### Sklearn Wrapper Function
```
# The predict method of sklearn's RandomForestClassifier returns a binary classification (0 or 1). 
# The following code creates a wrapper function, SklearnModelWrapper, that uses 
# the predict_proba method to return the probability that the observation belongs to each class. 
 
class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]
```

### Method 3 - sklearn model, autolog() and eval_and_log_metrics()

```
mlflow.sklearn.autolog()

with mlflow.start_run(run_name='autolog_random_forest'):
  n_estimators = 200
  model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
  model.fit(X_train, y_train)
 
  pred_test = model.predict(X_test)
  
  metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="eval_")
 
  # Log the model with a signature that defines the schema of the model's inputs and outputs. 
  # When the model is deployed, this signature will be used to validate inputs.
  signature = infer_signature(X_train, model.predict(X_train))
  
  # MLflow contains utilities to create a conda environment used to serve models.
  # The necessary dependencies are added to a conda.yaml file which is logged along with the model.
  conda_env =  _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
        additional_conda_channels=None,
    )
  
  mlflow.sklearn.log_model(model, "autolog_rf_sklearn_model", conda_env=conda_env, signature=signature)
```

```
# Feature Importances
list(zip(X_train, model.feature_importances_))
```

```
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_test))
```

## Predict From Model

```
import mlflow

model_name = "Model Name V1"
model_version_uri = "models:/{model_name}/production".format(model_name=model_name)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_version_uri))
loaded_model = mlflow.sklearn.load_model(model_version_uri)
```

```
# Predict 0 or 1 on a Pandas DataFrame.
import pandas as pd
y_pred = loaded_model.predict(X_test)
y_pred
```

```
# Predict Probability on Pandas dataframe
y_prob = loaded_model.predict_proba(X_test)[:,1]
y_prob
```

```
#Save a copy of test dataframe
test_data = X_test

# Save the predicted value, predict_prob value and actual target into the test dataframe
test_data["target"] = y_test
test_data["y_pred"] = y_pred
test_data["y_prob"] = y_prob

#Move Loan App ID as Index Column and Save the dataframe
test_data.reset_index(inplace=True)
test_data = test_data.rename(columns = {'index':'IDColumn'})

test_data.head()
```
