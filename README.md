# SmartShield 🛡️

**On-Device Indian SMS Scam & Phishing Detection with Explainable AI**

> PS3 — On-device threat, phishing & scam detection | Bharat AI-SoC Challenge 2026-27

## Overview

SmartShield is a fully offline, privacy-preserving SMS scam detection system designed for Indian users. It runs entirely on-device on ARM hardware (Android phone / Raspberry Pi 5) with zero cloud dependency.

## 3-Stage Hierarchical Detection Pipeline

```
SMS Input
   │
   ▼
┌─────────────────────────────┐
│ Stage 1: Rule Engine (<1ms) │ ← Regex patterns, India-specific
│ Catches ~65-70% obvious     │   scam templates, URL blacklists
└──────────┬──────────────────┘
           │ Ambiguous?
           ▼
┌─────────────────────────────┐
│ Stage 2: TinyBERT (5-10ms)  │ ← ONNX Runtime, INT8 quantized
│ Fine-tuned on Indian SMS    │   ~25MB model
└──────────┬──────────────────┘
           │ User requests explanation?
           ▼
┌─────────────────────────────┐
│ Stage 3: Llama 3.2 1B       │ ← ExecuTorch, 4-bit quantized
│ On-demand explanation only  │   ~700MB model
└─────────────────────────────┘
```

## India-Specific Threat Categories

| # | Category | Description |
|---|----------|-------------|
| 1 | Fake KYC Expiry | Fake bank/telecom KYC re-verification links |
| 2 | UPI Collect Fraud | Trick to approve collect requests disguised as credits |
| 3 | OTP Phishing | Directly asks user to share OTP |
| 4 | Prize/Lottery Scam | Fake prize claims with processing fees |
| 5 | Loan App Extortion | Threatens to contact friends/family for loan default |
| 6 | Bank Impersonation | Mimics bank alerts with fake numbers/links |
| 7 | Fake Delivery | Fake customs/redelivery fees via links |
| 8 | Job/Income Scam | Unrealistic daily income promises |

## Quick Start

### Prerequisites
- Python 3.10+
- ARM device (Android phone or Raspberry Pi 5)

### Installation
```bash
git clone https://github.com/<your-org>/SmartShield.git
cd SmartShield
pip install -r requirements.txt
```

### Run Interactive Mode
```bash
python src/pipeline.py --mode interactive
```

### Run on Test Dataset
```bash
python src/pipeline.py --mode batch --input data/test.csv
```

### Analyze Single Message
```bash
python src/pipeline.py --mode single --message "Your KYC expired, update now: http://fake-bank.xyz" --explain
```

## Project Structure

```
SmartShield/
├── src/
│   ├── rule_engine.py        # Stage 1: Rule-based detection
│   ├── pipeline.py           # Main pipeline integrating all 3 stages
│   └── __init__.py
├── rules/
│   └── scam_patterns.json    # India-specific regex patterns
├── data/
│   ├── train.csv             # Training set (200 scam + 83 legitimate)
│   ├── val.csv               # Validation set
│   └── test.csv              # Test set
├── models/                   # TinyBERT ONNX + Llama ExecuTorch models
├── scripts/
│   └── generate_dataset.py   # Dataset generation script
├── benchmarks/               # Latency, memory, temperature results
├── docs/                     # Technical report
├── demo/                     # Demo video
└── README.md
```

## Key Technologies

| Component | Technology |
|-----------|-----------|
| Rule Engine | Python regex + JSON pattern rules |
| ML Classifier | TinyBERT (HuggingFace) → ONNX Runtime ARM (INT8) |
| LLM Explainer | Llama 3.2 1B Instruct (Q4) via **ExecuTorch** |
| Target Hardware | ARM Android Phone / Raspberry Pi 5 |

## Offline Operation

All detection and explanation runs fully on-device:
- No internet required after initial model download
- No message content transmitted to any server
- Demonstrated with Wi-Fi and mobile data disabled

## Team

**Team VEDA** — PSG College of Technology, Coimbatore

## License

MIT License
