# 🫁 Chest Disease Classification — End-to-End MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?logo=mlflow)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?logo=dvc&logoColor=white)](https://dvc.org)
[![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![AWS](https://img.shields.io/badge/AWS-EC2%20%2B%20ECR-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-grade deep learning pipeline** for classifying chest diseases (adenocarcinoma cancer) from CT scan images using transfer learning (VGG16), with full MLOps infrastructure — experiment tracking, data versioning, CI/CD, containerised deployment, and a Flask web interface for inference.

---

## 📑 Table of Contents

- [Why This Project](#-why-this-project)
- [Architecture](#-architecture)
- [ML Model & Deep Learning Details](#-ml-model--deep-learning-details)
- [Tech Stack](#-tech-stack)
- [DVC Pipeline Stages](#-dvc-pipeline-stages)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [CI/CD & Deployment](#-cicd--deployment)
- [Results & Metrics](#-results--metrics)
- [Future Improvements](#-future-improvements)

---

## 🎯 Why This Project

Building a CNN that classifies images is straightforward. Building one that's **reproducible, version-controlled, experiment-tracked, and deployable** is where real engineering starts. This project demonstrates the full MLOps lifecycle — not just model accuracy, but how you'd actually ship and maintain a medical imaging classifier in a team setting.

---

## 🏗 Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  CT Scan      │────▶│  Data        │────▶│  Base Model      │
│  Images       │     │  Ingestion   │     │  Preparation     │
│  (Google Drive)│    │  + Validation │     │  (VGG16 Transfer)│
└──────────────┘     └──────────────┘     └────────┬─────────┘
                                                    │
                                                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Flask Web    │◀────│  Docker      │◀────│  Model Training  │
│  App (EC2)    │     │  Container   │     │  + Evaluation     │
└──────┬───────┘     └──────────────┘     │  (MLflow logged)  │
       │                     ▲            └──────────────────┘
       │              ┌──────┴───────┐
       │              │  CI/CD       │
       └──────────────│  (GitHub     │
                      │   Actions)   │
                      └──────────────┘
```

**Pipeline Flow:** CT scan images (sourced from Google Drive) → Data ingestion & validation → VGG16 base model preparation with ImageNet weights → Fine-tuning & training (logged to MLflow via DagsHub) → Evaluation with metric logging → Dockerise → Push to AWS ECR → Deploy to EC2 via CI/CD.

**Data versioning** is handled by DVC throughout, ensuring every experiment is tied to a specific dataset version.

---

## 🤖 ML Model & Deep Learning Details

### Model Architecture

| Component | Detail |
|---|---|
| **Base Model** | VGG16 (pre-trained on ImageNet) |
| **Approach** | Transfer Learning + Fine-Tuning |
| **Input** | Chest CT scan images (resized to 224×224×3) |
| **Output** | Binary classification — Adenocarcinoma vs Normal |
| **Top Layers** | Custom classification head (Flatten → Dense → Softmax) |
| **Frozen Layers** | VGG16 convolutional base frozen during initial training; selectively unfrozen for fine-tuning |
| **Loss Function** | Categorical Cross-Entropy |
| **Optimiser** | Adam (with configurable learning rate) |
| **Data Augmentation** | Rotation, shift, zoom, horizontal flip via `ImageDataGenerator` |

### Why VGG16?

VGG16 is a proven architecture for medical imaging tasks. Its deep stack of 3×3 convolution filters captures fine-grained texture patterns — critical for distinguishing between cancerous and normal tissue in CT scans. Using ImageNet pre-trained weights gives the model a strong feature extraction foundation, which is then adapted to the medical domain through fine-tuning. This transfer learning approach works especially well when the available medical dataset is relatively small.

### Key ML/DL Libraries

- **TensorFlow / Keras** — Model definition, training, data augmentation, callbacks
- **VGG16 (keras.applications)** — Pre-trained CNN backbone with ImageNet weights
- **MLflow** — Experiment tracking (hyperparameters, loss, accuracy, model artifacts)
- **DVC** — Data versioning, pipeline orchestration, reproducibility
- **DagsHub** — Remote MLflow tracking server + DVC remote storage integration
- **NumPy / PIL** — Image preprocessing and array manipulation
- **scikit-learn** — Evaluation metrics (confusion matrix, classification report)
- **Matplotlib / Seaborn** — Training curves, confusion matrix visualisation

---

## 🛠 Tech Stack

### ML & Data

| Tool | Role |
|---|---|
| **Python 3.8+** | Core language |
| **TensorFlow / Keras** | Deep learning framework — model building, training, inference |
| **VGG16 (ImageNet)** | Pre-trained CNN for transfer learning |
| **MLflow** | Experiment tracking — logs hyperparameters (learning rate, epochs, batch size), metrics (loss, accuracy, val_accuracy), and model artifacts per run |
| **DVC** | Data Version Control — tracks dataset versions, defines reproducible pipeline stages, ensures experiment ↔ data traceability |
| **DagsHub** | Remote MLflow tracking URI + DVC storage backend — collaborative experiment management |

### Application & Serving

| Tool | Role |
|---|---|
| **Flask** | Web application for upload-and-predict inference |
| **HTML/CSS (Jinja2)** | Frontend for image upload and result display |
| **Gunicorn** | Production WSGI server |

### Infrastructure & DevOps

| Tool | Role |
|---|---|
| **Docker** | Containerises the full application (trained model + Flask API + dependencies) |
| **AWS ECR** | Elastic Container Registry — stores Docker images |
| **AWS EC2** | Hosts the deployed containerised application |
| **GitHub Actions** | CI/CD — triggers on push to `main`, builds Docker image, pushes to ECR, deploys to EC2 |
| **Git / GitHub** | Version control |

### How They Connect

```
Developer pushes code
        │
        ▼
   ┌─────────┐                    ┌───────────────┐
   │ GitHub   │ ──── triggers ──▶ │ GitHub Actions │
   └─────────┘                    └───────┬───────┘
                                          │
                              ┌───────────┴───────────┐
                              │  1. Build Docker image │
                              │  2. Push to AWS ECR    │
                              │  3. Deploy to EC2      │
                              └───────────┬───────────┘
                                          │
                                          ▼
   ┌─────────────┐              ┌──────────────────┐
   │ DagsHub     │◀── logs ──── │  EC2 Instance     │
   │ (MLflow UI) │              │  (Flask + Model)  │
   └─────────────┘              └──────────────────┘
        ▲
        │
   DVC tracks data versions
   MLflow tracks experiments
```

---

## 🔄 DVC Pipeline Stages

The entire ML workflow is orchestrated as a DVC pipeline (`dvc.yaml`), making it fully reproducible with a single command:

```bash
dvc repro
```

| Stage | What It Does |
|---|---|
| **data_ingestion** | Downloads CT scan dataset from Google Drive, extracts and organises into train/test splits |
| **prepare_base_model** | Loads VGG16 with ImageNet weights, freezes convolutional base, adds custom classification head |
| **training** | Fine-tunes the model on chest CT images with data augmentation, logs to MLflow |
| **evaluation** | Evaluates on test set, logs final metrics (loss, accuracy) to MLflow, saves model artifacts |

---

## 📁 Project Structure

```
Chest-Disease-Classification/
├── .github/
│   └── workflows/
│       └── main.yaml            # CI/CD pipeline (GitHub Actions → ECR → EC2)
├── config/
│   └── config.yaml              # Data paths, model params, training config
├── research/
│   └── *.ipynb                  # Experimentation notebooks
├── src/
│   └── cnnClassifier/
│       ├── components/
│       │   ├── data_ingestion.py        # Download + extract CT scan data
│       │   ├── prepare_base_model.py    # VGG16 setup + custom head
│       │   ├── model_training.py        # Training loop with augmentation
│       │   └── model_evaluation.py      # Evaluation + MLflow logging
│       ├── config/
│       │   └── configuration.py         # Config manager (reads config.yaml)
│       ├── pipeline/
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_prepare_base_model.py
│       │   ├── stage_03_model_training.py
│       │   └── stage_04_model_evaluation.py
│       ├── entity/
│       │   └── config_entity.py         # Dataclass configs
│       ├── constants/
│       │   └── __init__.py              # Path constants
│       └── utils/
│           └── common.py                # Utility functions (yaml reader, dir creator)
├── templates/
│   └── index.html               # Flask frontend
├── artifacts/                   # Generated during pipeline (models, data)
├── Dockerfile                   # Container build
├── dvc.yaml                     # DVC pipeline definition
├── params.yaml                  # Hyperparameters (epochs, batch_size, lr, image_size)
├── application.py               # Flask app entry point
├── main.py                      # Runs full DVC pipeline programmatically
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Docker (for containerised deployment)
- AWS account with EC2 + ECR access (for cloud deployment)
- DagsHub account (for remote MLflow tracking)

### Local Development

```bash
# 1. Clone the repo
git clone https://github.com/nishantrv/Chest-Disease-Classification--MLFlow-DVC-MLOps-Deployment-.git
cd Chest-Disease-Classification--MLFlow-DVC-MLOps-Deployment-

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set MLflow tracking (DagsHub)
export MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/<your-repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
export MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>

# 5. Run the full pipeline
dvc repro
# OR
python main.py

# 6. Launch the Flask app for inference
python application.py
# App available at http://localhost:8080
```

### Docker

```bash
docker build -t chest-classifier .
docker run -p 8080:8080 chest-classifier
```

---

## 📊 MLflow Experiment Tracking

Every training run is logged to MLflow (hosted on DagsHub) with:

- **Parameters** — image size, batch size, epochs, learning rate, augmentation settings
- **Metrics** — training loss, training accuracy, validation loss, validation accuracy
- **Artifacts** — trained model (`.h5`), training history plots

View experiments in the MLflow UI:

```bash
mlflow ui --port 5001
# Or visit your DagsHub repo's experiment tab
```

---

## ☁️ CI/CD & Deployment

### AWS Deployment Architecture

```
GitHub Actions (on push to main)
        │
        ├── 1. Build Docker image
        ├── 2. Push to AWS ECR
        └── 3. SSH into EC2 → pull image → run container
```

### AWS Setup Required

1. **EC2 instance** — Ubuntu, with Docker installed
2. **ECR repository** — to store Docker images
3. **IAM policies** — `AmazonEC2ContainerRegistryFullAccess` + `AmazonEC2FullAccess`
4. **GitHub Secrets** — `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ECR_LOGIN_URI`, `ECR_REPOSITORY_NAME`

### EC2 Setup Commands

```bash
sudo apt-get update -y
sudo apt-get upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

---

## 🤝 Contributing

Contributions and feedback are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

**Built by [Nishant Ranjan Verma](https://github.com/nishantrv)** | Dublin, Ireland