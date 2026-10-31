# Aircraft Anomaly Detection

A machine learning project focused on detecting anomalous aircraft vibration patterns using accelerometer data.

## Dataset

This project uses the Airbus Helicopter Accelerometer Dataset.

* **Original publisher:** ETH Zurich
* **Original rights holder:** Airbus SAS
* **Licence:** CC BY-NC-SA 4.0

Dataset:  
https://www.research-collection.ethz.ch/entities/researchdata/a4acb31d-9711-4843-8a60-e2468c0326be

The raw dataset is not included in this repo and must be downloaded separately.

## Setup

```bash
git clone https://github.com/oakflow/aircraft-anomaly-detection
cd aircraft-anomaly-detection

python -m venv .venv

# activate venv then install dependencies
pip install -r requirements.txt
```