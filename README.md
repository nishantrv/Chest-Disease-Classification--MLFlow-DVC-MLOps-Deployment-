
## Workflow

1.Update config.yaml
2.Update params.yaml
3.Update the entity
4.Update the configuration manager in src config
5.Update the components
6.Update the pipeline
7.Update the main.py
8.Update the dvc.yaml

# MLFlow

import dagshub
dagshub.init(repo_owner='nishantrv', repo_name='testing', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

# MLOps-End-to-End-Chest-Disease-Classification-From-CT-Scan-Image-


# MLFlow Dagshub connection uri

MLFLOW_TRACKING_URI=https://dagshub.com/nishantrv/Chest-Disease-Classification--MLFlow-DVC-MLOps-Deployment-.mlflow \
MLFLOW_TRACKING_USERNAME=nishantrv \
MLFLOW_TRACKING_PASSWORD= 82a542eedc8c127c1e65c083e6950b114f3808c0 \
python script.py
14271269983f6a2969909801eb96c9c7406151e4

https://dagshub.com/nishantrv/Chest-Disease-Classification--MLFlow-DVC-MLOps-Deployment-.mlflow

# Run from Bash

export MLFLOW_TRACKING_URI=https://dagshub.com/nishantrv/Chest-Disease-Classification--MLFlow-DVC-MLOps-Deployment-.mlflow

export MLFLOW_TRACKING_USERNAME=nishantrv 

export MLFLOW_TRACKING_PASSWORD=14271269983f6a2969909801eb96c9c7406151e4

