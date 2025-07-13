# Travel Guide Question Classification with FLAN-T5

This project implements a text classification system for travel-related queries using the FLAN-T5 model. The goal is to assign each user question to one of seven coarse-grained travel categories. The system is fully automated in Google Colab Pro (A100 GPU) and includes dataset preparation, prompt-based testing, fine-tuning, and evaluation.

---

## Overview

- **Model**: FLAN-T5 Base (encoder-decoder)
- **Task**: 7-class classification — TGU, TTD, TRS, ACM, FOD, ENT, WTH
- **Techniques**: Zero-shot / Few-shot prompting and supervised fine-tuning
- **Environment**: Google Colab Pro (A100), Hugging Face Transformers

---

## Dataset Labels

| Acronym | Meaning        |
|---------|----------------|
| TGU     | Travel Guide   |
| TTD     | Things to Do   |
| TRS     | Transport      |
| ACM     | Accommodation  |
| FOD     | Food           |
| ENT     | Entertainment  |
| WTH     | Weather        |

Dataset is automatically preprocessed, balanced, and split (train/val/test).

---

## Evaluation Results

### Prompt-Based Accuracy

| Setting       | Accuracy (%) |
|---------------|--------------|
| Zero-shot     | 85.00        |
| One-shot      | 85.00        |
| Three-shot    | 90.00        |
| Fine-tuned    | 92.00        |

Fine-tuning yielded the highest performance; few-shot prompting showed modest improvement over zero-shot.

---

### Sample Predictions (Fine-tuned FLAN-T5)

| Question                                                    | Predicted | True |
|-------------------------------------------------------------|-----------|------|
| How is the weather in September in Hong Kong?               | WTH       | WTH  |
| What is a decent bakery in Minorca?                         | FOD       | FOD  |
| Can I apply for tourist visa in St. Petersburg...?          | TGU       | TGU  |
| What is a good website to book cheap hotels around Italy?   | ACM       | TGU  |
| What are the visa requirements for UK nationals to Dubai?   | TGU       | TGU  |

Most predictions aligned with true labels; minor errors occurred on overlapping intent classes.

---

## Model Comparison

### Evaluation Metrics

| Metric        | FLAN-T5 Base | FLAN-T5 LoRA | LoRA 4-bit |
|---------------|--------------|--------------|------------|
| Loss          | 0.137        | **0.123**    | 0.183      |
| Accuracy (%)  | 95.45        | **97.42**    | 92.83      |
| F1 Score (%)  | 97.23        | **98.43**    | 95.93      |
| Precision (%) | 97.23        | **98.44**    | 96.01      |
| Recall (%)    | 97.23        | **98.41**    | 95.85      |

### Runtime and Efficiency

| Metric                     | Base   | LoRA   | LoRA 4-bit |
|----------------------------|--------|--------|------------|
| Training Time (min)        | 1.72   | 1.71   | **1.65**   |
| Evaluation Time (min)      | 0.036  | 0.036  | 0.050      |
| Samples/sec (eval)         | 326.56 | 325.75 | 235.10     |
| Steps/sec (training)       | 10.26  | 10.24  | 7.39       |

LoRA-based models improved accuracy with minimal additional training cost. The 4-bit LoRA version offers speed–accuracy tradeoffs for resource-constrained settings.

---

## Repository Contents

- `notebook.ipynb` – Full pipeline with outputs
- `requirements.txt` – Dependencies
- `data/` – Optional sample dataset

---

## Reproducibility

Open in Google Colab:  
[![Open in Colab](https://colab.research.google.com/gist/YuryChern/c078899c2ee7988cfe9de34e90656c9f/assignment-2-158736-yury-chernykh-id-14072993.ipynb)

> All outputs are preserved. The Colab workflow runs end-to-end, with runtime under 20 minutes using A100.

---

## Author

[Yury Chernykh]  
Massey University — 158736 Advanced Machine Learning  
[yuryc@me.com]
