# SHAP Explainer

Steps to create a SHAP explainer for a trained and stored scikit learn model in mlflow model registry. 

```
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import col,sum

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, classification_report, accuracy_score, precision_score, recall_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

# These commands are only required if you are using a cluster running DBR 7.3 LTS ML or below. 
import cloudpickle
assert cloudpickle.__version__ >= "1.4.0", "Update the cloudpickle library using `%pip install --upgrade cloudpickle`"

mlflow.sklearn.autolog()
```

```
#Convert Spark Dataframe to Pandas
import numpy as np
import pandas as pd
# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

pdf = df.toPandas() 
pdf.set_index("LoanAppID", inplace=True)
#pdf.head()
```

```
from sklearn.model_selection import train_test_split

#Select ID, Input and Target
y = pdf["target"]
X = pdf.drop(["target"], axis=1) 

# split 80/20 train-test : stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.3, stratify=y)
```

Import Model from Model Registry

```
import mlflow

model_name = "Classifier Model M1"
model_version_uri = "models:/{model_name}/production".format(model_name=model_name)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_version_uri))
loaded_model = mlflow.sklearn.load_model(model_version_uri)
```

```
# Predict 0 or 1 on a Pandas DataFrame.
import pandas as pd
y_pred = loaded_model.predict(X_test)

# Predict Probability on Pandas dataframe
y_prob = loaded_model.predict_proba(X_test)[:,1]
```

## Build SHAP Explainer

```
import shap
import matplotlib.pyplot as plt

shap.initjs()
```

```
# take a sample of 100 rows in a dataframe called background
feature_names = list(X_test.columns)
background = X_test[0: 100]
background.shape
background.head()
```

```
#Initialize Shap Explainer
explainer = shap.TreeExplainer(loaded_model, check_additivity=False, model_output="raw")
explainer
```

## SHAP Local Explainer

```
#Select a row from test set (You can also select multiple rows - but keep a limit on the count since it is computationally expensive)
choosen_instance = X_test.loc[[1950248]]
shap_values = explainer.shap_values(choosen_instance)

p =shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance, matplotlib=True)
display(p)
```

## SHAP Global Explainer 

```
from datetime import datetime
startDT = datetime.now()
print("Start datetime to Compute Shap values:", startDT)
```

```
# explain the model's predictions using SHAP 
# Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.

example = background.sample(n=50)
shap_values = explainer.shap_values(background)
#shap_interaction_values = explainer.shap_interaction_values(background)
#expected_value = explainer.expected_value
```

```
shap_values1 = shap_values[1]
```

```
# Extract components from shap - SHAP bar plot
shap.summary_plot(shap_values1, background, feature_names, plot_type='bar')
```

```
# SHAP Model Importance 
shap.summary_plot(shap_values1, background, feature_names, class_names=['target'])
```

```
endDT =datetime.now()
print("End datetime to Compute Shap values:", endDT)
```

## Store SHAP Values

```
from datetime import datetime
startDT = datetime.now()
print("Start datetime to Compute Shap values:", startDT)
```


```
#Shap Golbal Explainer with 1 sample - TAKES 8.25 hours to run for 1 sample
example = background.sample(n=1)

shap_values = explainer.shap_values(background)
shap_interaction_values = explainer.shap_interaction_values(background)
expected_value = explainer.expected_value
shap_values1 = shap_values[1]
```

```
shap_values0 = shap_values[0]
shap_values0
```

```
#Store the SHAP Values 
cols = background.columns
explainer_output_path = '/dbfs/mnt/folder path/'

#save the probability of "true" variant 
shap_values_df = pd.DataFrame(shap_values[1], columns=cols)
shap_values_save_path = explainer_output_path+'shap_values_1.csv'
shap_values_df.to_csv(shap_values_save_path, index=False)
mlflow.log_artifact(shap_values_save_path)

# #save expected_values
# save both True/False variations here
expected_values_df = pd.DataFrame({'expectedValue': [expected_value]}) # pd.DataFrame({0:[dq_expected_value[0]], 1:[dq_expected_value[1]]})
expected_values_save_path = explainer_output_path+'expected_values.csv'
expected_values_df.to_csv(expected_values_save_path, index=False)
mlflow.log_artifact(expected_values_save_path)

#save dq_shap_interaction_values
shap_interaction_values_df = pd.Dataframe(shap_interaction_values[1])
shap_interaction_values_df_save_path = explainer_output_path+'shap_interaction_values_1.csv'
shap_interaction_values_df.to_csv(shap_interaction_values_df_save_path, index=False)
mlflow.log_artifact(shap_interaction_values_df_save_path)
```

```
#save explainer object
import joblib
explainer_save_path = explainer_output_path+'explainer.bz2'
joblib.dump(explainer, explainer_save_path, compress=('bz2', 9))
mlflow.log_artifact(explainer_save_path)
```

```
endDT =datetime.now()
print("End datetime to Compute Shap values:", endDT)
```

```
shap.summary_plot(shap_values1, example, feature_names, plot_type='bar')
```

```
mlflow.end_run()
```

## SHAP using mlflow log

```
import os
import mlflow

# log an explanation
with mlflow.start_run() as run:
    mlflow.shap.log_explanation(loaded_model.predict_proba, example)

# list artifacts
client = mlflow.tracking.MlflowClient()
artifact_path = "model_explanations_shap"
artifacts = [x.path for x in client.list_artifacts(run.info.run_id, artifact_path)]
print("# artifacts:")
print(artifacts)

# load back the logged explanation
dst_path = client.download_artifacts(run.info.run_id, artifact_path)
base_values = np.load(os.path.join(dst_path, "base_values.npy"))
shap_values = np.load(os.path.join(dst_path, "shap_values.npy"))

mlflow.end_run()
```

```
import os
import mlflow

# log an explanation
with mlflow.start_run(run_name='loan_app_explainability') as run:
    #mlflow.shap.log_explainer(explainer=explainer,artifact_path ="shap_explainer")
    mlflow.shap.log_explanation(loaded_model.predict, example)

# list artifacts
client = mlflow.tracking.MlflowClient()
artifact_path = "model_explanations_shap"
artifacts = [x.path for x in client.list_artifacts(run.info.run_id, artifact_path)]
print("# artifacts:")
print(artifacts)

# load back the logged explanation
dst_path = client.download_artifacts(run.info.run_id, artifact_path)
base_values = np.load(os.path.join(dst_path, "base_values.npy"))
shap_values = np.load(os.path.join(dst_path, "shap_values.npy"))

mlflow.end_run()
```
