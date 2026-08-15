# Anomaly Detection for Network Security

## Introduction
Modern cybersecurity infrastructure requires proactive threat detection mechanisms capable of identifying novel network intrusion vectors in real time. Inspired by AgentGuard behavioral monitoring, this project implements unsupervised anomaly detection pipelines on the NSL-KDD network traffic benchmark to separate normal traffic from malicious cyberattacks.

## Methodology
Two complementary unsupervised learning paradigms were developed:
1. **Isolation Forest**: An tree-ensemble method that isolates anomalous observations by randomly selecting features and split values (`contamination=0.1`).
2. **Deep Autoencoder**: A 4-layer bottleneck neural network (Encoder: 32-16, Decoder: 16-32) trained exclusively on normal traffic patterns using MSE loss. Anomalies are detected when test reconstruction error exceeds the 95th percentile normal training threshold.

## Results
Performance comparison between Isolation Forest and Deep Autoencoder on the NSL-KDD test set:

| Model | Normal Precision | Attack Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: |
| **Isolation Forest** | 0.5128 | 0.2971 | **56.24%** |
| **Deep Autoencoder** | 0.7050 | 0.7065 | **80.14%** |

### Confusion Matrix - Isolation Forest:
```
[[9493  218]
 [9020 3813]]
```

### Confusion Matrix - Deep Autoencoder:
```
[[9001  710]
 [3767 9066]]
```

## Future Work
- Evaluate model generalization on the modern WSN-DS (Wireless Sensor Network Data Set) dataset to benchmark intrusion detection across IoT and sensor networks.

## How to Run
1. Open Jupyter Notebook in this directory:
   ```bash
   jupyter notebook 02_anomaly_detection.ipynb
   ```
2. Execute all cells sequentially.
