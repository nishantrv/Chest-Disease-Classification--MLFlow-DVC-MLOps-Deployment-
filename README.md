<<<<<<< HEAD
## Workflow

Update config.yaml
Update params.yaml
Update the entity
Update the configuration manager in src config
Update the components
Update the pipeline
Update the main.py
Update the dvc.yaml

###

import dagshub
dagshub.init(repo_owner='nishantrv', repo_name='testing', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
=======
# MLOps-End-to-End-Chest-Disease-Classification-From-CT-Scan-Image-


# MLFlow Dagshub connection uri

MLFLOW_TRACKING_URI=https://dagshub.com/nishantrv/testing.mlflow \
MLFLOW_TRACKING_USERNAME=nishantrv \
MLFLOW_TRACKING_PASSWORD= 82a542eedc8c127c1e65c083e6950b114f3808c0 \
python script.py


# Run from Bash

export MLFLOW_TRACKING_URI=https://dagshub.com/nishantrv/testing.mlflow

export MLFLOW_TRACKING_USERNAME=nishantrv 

export MLFLOW_TRACKING_PASSWORD=82a542eedc8c127c1e65c083e6950b114f3808c0

>>>>>>> 576172b327ff437e5b9c75254f1c28ee581c89a0
