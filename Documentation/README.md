## Trainable IDS Project

This project is a multi-file defensive Intrusion Detection System (IDS) prototype designed for scalability and extensibility. It supports both machine learning–based detection and optional LLM-assisted analysis.

## Features

Training on labeled datasets

Runtime prediction on incoming logs

Optional LLM-based explanations

SQLite alert storage

FastAPI-based alert endpoint

## Project Structure
llm_ids_project/
├── app/
├── data/
├── models/
├── requirements.txt
├── train.py
└── README.md
Installation

## Install required dependencies:

pip install -r requirements.txtTrainin

#Train the model using available datasets:
python train.py

## Run the IDS detection engine:

python -m app.main
Run API

## Start the FastAPI server for alert access:

uvicorn app.api:app --reload
Notes

## The model must be trained before running the detector

Trained models are saved in the models/ directory

The system is designed to support large datasets and multiple data sources

## Primary Dataset

The current implementation is trained on the following dataset:

Kitsune Network Attack Dataset
[https://www.kaggle.com/datasets/ymirsky/network-attack-dataset-kitsune](https://www.kaggle.com/datasets/ymirsky/network-attack-dataset-kitsune)

Originally introduced in:

Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018).
Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection.
Proceedings of the Network and Distributed System Security Symposium (NDSS).

---

## Dataset Description

The Kitsune dataset is a benchmark dataset for intrusion detection research. It contains network traffic captured from IoT environments and includes multiple attack scenarios with corresponding labeled data.

The dataset supports both anomaly detection and supervised classification tasks and is widely used in academic and applied cybersecurity research.

---

## Attack Categories Included

The dataset includes multiple attack types, such as:

Active Wiretap
ARP Man-in-the-Middle (MitM)
Fuzzing
Mirai Botnet
OS Scanning
SSDP Flood
SSL Renegotiation
SYN Denial of Service (DoS)
Video Injection

These scenarios cover a range of behaviors including reconnaissance, denial of service, botnet activity, and injection-based attacks.

---

## Data Usage in This Project

The training pipeline processes datasets as follows:

1. Recursively scans dataset directories
2. Identifies dataset and label file pairs
3. Merges feature data with labels by row index
4. Normalizes labels into a common format:

   * benign
   * suspicious
   * malicious
5. Converts structured network features into a unified representation for model training
6. Trains a scalable machine learning model using incremental learning

Packet capture files (`.pcap`, `.pcapng`) are retained for reference and validation but are not directly used during model training.

---

## Files Used for Training

Used directly in training:

* `*_dataset.csv` — extracted network traffic features
* `*_labels.csv` — ground-truth labels

Not used directly in training:

* `*.pcap`
* `*.pcapng`

---

## Data Attribution

All datasets used in this project are externally sourced. This project does not claim ownership of:

* raw network traffic captures
* extracted feature datasets
* labeling annotations

These datasets are used strictly for academic, research, and defensive cybersecurity purposes.

---

## Citation

When using or extending this project, the original dataset should be cited:

Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018).
Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection.
NDSS 2018.

Users should also follow the licensing terms provided by the dataset source.

---

## Extensibility

The training pipeline is designed to support additional datasets. Future integrations may include:

* CICIDS2017 / CICIDS2018
* UNSW-NB15
* Zeek network logs
* Suricata EVE JSON
* Sysmon event logs

New datasets can be added by placing them in the data directory, provided they include either:

* labeled feature data, or
* separable dataset and label files

---

## Reproducibility

To train the model:

```bash
python train.py data/
```

The training process will:

* discover dataset files automatically
* normalize and merge labels
* process large datasets in chunks
* save the trained model to:

```text
models/ids_model.pkl
```

---

## Ethical Use

This project is intended for defensive cybersecurity research, intrusion detection development, and academic use. It should not be used for unauthorized or offensive activities.
