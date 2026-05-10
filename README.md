# 🏦 AI Claim Analysis & Risk Assessment System (Malaysia)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-RandomForest-green)
![RAG](https://img.shields.io/badge/RAG-Enabled-purple)
![LLM](https://img.shields.io/badge/LLM-Ollama%20Llama3-orange)
![OCR](https://img.shields.io/badge/OCR-Tesseract-red)
![UI](https://img.shields.io/badge/UI-Gradio-blueviolet)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)

An end-to-end **Document Intelligence + Predictive Analytics + Retrieval-Augmented Generation (RAG) + Generative AI** system for insurance claim processing in Malaysia.

This system automatically:

* Extracts structured data from claim PDFs (OCR + NLP)
* Predicts claim risk using Machine Learning
* Retrieves similar historical cases using semantic search (RAG)
* Generates AI-powered executive summaries (LLM-ready)
* Provides interactive UI via Gradio

> Designed as a hybrid AI decision-support system for insurance analytics.

---

# 🧩 System Architecture (High-Level)

1. **Input:** Insurance Claim PDF
2. **Extraction:** pdfplumber → OCR fallback (Tesseract)
3. **Entity Parsing:** Regex-based structured extraction
4. **Feature Mapping:** Convert to ML-ready features
5. **Prediction:** RandomForest risk classification
6. **RAG Retrieval:** SentenceTransformer similarity search
7. **Summary:** LLM (Ollama Llama3) or fallback logic
8. **Output:** Risk label + structured data + explanation

---

# 🏗️ Technical Architecture

```text
Claim PDF
   ↓
PDF Extraction (pdfplumber / OCR)
   ↓
Entity Extraction (Regex / NLP)
   ↓
Feature Engineering
   ↓
RandomForest Classifier
   ↓
Risk Prediction (Low / Medium / High)
   ↓
RAG Retrieval (Semantic Search)
   ↓
LLM Summary (Ollama / fallback)
   ↓
Gradio UI Output
```

---

# ✨ Key Features

## 📄 Document Processing

* PDF text extraction using `pdfplumber`
* OCR fallback using `pytesseract`

## 🧠 Entity Extraction

Extracts:

* Claimant name
* Claim amount (RM)
* Incident type
* State
* Policy number
* Incident date

---

## 🤖 Risk Prediction (Machine Learning)

* Model: `RandomForestClassifier (300 trees)`
* Preprocessing: `OneHotEncoder + ColumnTransformer`
* Balanced class weighting
* Output classes:

  * Low Risk
  * Medium Risk
  * High Risk

---

## 🔎 RAG (Retrieval-Augmented Generation)

* Model: `SentenceTransformer (all-MiniLM-L6-v2)`
* Cosine similarity search
* Retrieves top-k similar historical claims

---

## 📝 AI Executive Summary

* Uses **Ollama Llama3**
* Fallback rule-based summary if LLM unavailable
* Generates insurance-style risk explanation

---

## 🎛️ Interactive UI

Built with **Gradio**, allows:

* Upload claim PDF
* View extracted structured data
* View risk prediction
* View similar historical cases
* View AI-generated summary

---

# 📊 Model Performance Evaluation

The model was evaluated on a **test set of 2,000 samples**, producing the following results:

### 🔍 Classification Report

```text
              precision    recall  f1-score   support

        High       1.00      0.98      0.99        91
         Low       0.97      1.00      0.99      1699
      Medium       0.99      0.79      0.88       210

    accuracy                           0.98      2000
   macro avg       0.99      0.92      0.95      2000
weighted avg       0.98      0.98      0.98      2000
```

---

## 📌 Key Insights

* **Overall Accuracy:** 98% → Strong multi-class classification performance
* **High Risk:** Excellent detection (F1 = 0.99)
* **Low Risk:** Perfect recall (1.00) → highly reliable automation
* **Medium Risk:** Lower recall (0.79) → needs improvement

---

## 🧠 Interpretation

This model is suitable for **insurance decision-support systems** because:

* ✔ High precision reduces false alarms
* ✔ Strong recall for low-risk supports automation approvals
* ⚠ Medium risk requires further tuning (class imbalance handling)

---

# 📊 Model Training Pipeline (`train_model.py`)

* End-to-end ML pipeline using **scikit-learn**
* Handles categorical + numerical features automatically
* Stratified train-test split (80/20)
* RandomForest classifier
* Saves full pipeline as:

```text
risk_model.pkl
```

✔ Includes preprocessing + model together
✔ Ready for production inference

---

# 📊 Example Output

## 📄 Extracted Structured Data

![Data Extracted](https://raw.githubusercontent.com/azlinaaaa/AI-Powered-Risk-Assessment-and-Claims-Analytics-Platform/067e7baca1b08d32a8019f0a70a8f7b7cbf13a1e/Output/Data_Extracted.png)

---

## 🧠 AI Executive Summary

![AI Summary](https://raw.githubusercontent.com/azlinaaaa/AI-Powered-Risk-Assessment-and-Claims-Analytics-Platform/067e7baca1b08d32a8019f0a70a8f7b7cbf13a1e/Output/AI_Summary.png)

---

## 📌 Insight

These outputs demonstrate:

* ✔ Accurate extraction of structured insurance data
* ✔ Reliable AI-based risk interpretation
* ✔ Full pipeline integration (OCR → ML → RAG → LLM)

---

# 📂 Project Structure

```text
.
├── app.py
├── train_model.py
├── risk_model.pkl
├── claims_dataset_malaysia_calibrated_10000.csv
├── knowledge_base_100_cases.txt
├── pdfs/
├── requirements.txt
└── README.md
```

---

# 🛠 Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### Install OCR Engine

* Windows → install Tesseract + add PATH
* Mac → `brew install tesseract`
* Ubuntu → `sudo apt install tesseract-ocr`

---

# ▶️ Run Application

```bash
python app.py
```

Open the local URL shown in terminal.

---

# 🧠 Design Philosophy

This system follows a **hybrid AI architecture**:

* Rule-based extraction (structured reliability)
* Machine Learning (risk prediction)
* Semantic retrieval (RAG knowledge grounding)
* Generative AI (natural language explanation)

✔ Reduces hallucination
✔ Improves interpretability
✔ Suitable for real-world deployment

---

# 🔐 Enterprise Considerations

* PII encryption required
* Role-based access control
* Audit logging
* Model drift monitoring
* Secure deployment pipeline

---

# 🚀 Future Enhancements

* SHAP explainability dashboard
* Fraud detection model
* FastAPI backend integration
* React frontend upgrade
* Docker deployment
* Database integration
* Cloud monitoring pipeline

---

# 🧾 Tech Stack

* Python (pandas, scikit-learn, joblib)
* OCR: pdfplumber, pytesseract
* ML: RandomForest
* RAG: SentenceTransformers
* UI: Gradio
* LLM: Ollama Llama3
