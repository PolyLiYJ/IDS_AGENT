# IDS-Agent: An LLM Agent for Explainable Intrusion Detection in IoT Networks

## Overview
This repository contains the implementation of IDS-Agent, an intrusion detection system agent capable of detecting unknown attacks in IoT networks. The system consists of two main components:
- `run_agent_v2.py`: Primary detection module with unknown attack detection capability

## Features
- Detection of unknown attacks in IoT networks
- Explainable AI for intrusion detection
- (Additional features to be documented)

## Installation
1. Clone the repository:
```bash
git clone [repository_url]
cd IDS_AGENT
```

2. Create and activate conda environment:
```bash
conda create -n ids_agent python=3.8
conda activate ids_agent
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Jupyter Notebook (if not already installed):
```bash
pip install jupyter
```
### Data Preprocessing
1. Preprocess main dataset:
```bash
jupyter notebook process.ipynb
```
- Output: Preprocessed dataset files
- Expected artifacts:
  - Cleaned dataset
  - Feature engineered data
  - Train/test splits

2. Preprocess CICIoT dataset:
```bash
jupyter notebook process_CICIoT.ipynb
``` 
- Output: Standardized IoT dataset
- Expected artifacts:
  - Normalized features
  - Attack type labels
  - Combined dataset

### Model Training
```bash
python train_model.py
```
- Expected outputs:
  - Trained model weights (.joblib file)
  - Training logs
  - Performance metrics
  - Model evaluation results


### Running the Detection Agent
```bash
python run_agent_v2.py
```

### Generating Confusion Matrix
1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open and run `ids_agent.ipynb`:
- The notebook will generate a confusion matrix showing detection performance
- Expected output includes:
  - True Positive / False Positive rates
  - Accuracy metrics
  - Visualization of detection results

## Citation
If you use this work, please cite the original paper:
```
@article{li2024idsagent,
  title={IDS-Agent: An LLM Agent for Explainable Intrusion Detection in IoT Networks},
  author={Li, Yanjie and Xiang, Zhen and Bastian, Nathaniel D and Song, Dawn and Li, Bo},
  journal={NeurIPS 2024 Workshop Open-World Agents},
  year={2024}
}
```

## License
MIT