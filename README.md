# 🏦 AI Claim Analysis & Risk Assessment System (Malaysia)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-RandomForest-green)
![RAG](https://img.shields.io/badge/RAG-Enabled-purple)
![LLM](https://img.shields.io/badge/LLM-Ollama%20Llama3-orange)
![OCR](https://img.shields.io/badge/OCR-Tesseract-red)
![UI](https://img.shields.io/badge/UI-Gradio-blueviolet)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)

An end-to-end **Document Intelligence + Predictive Analytics + Retrieval-Augmented Generation (RAG) + Generative AI** prototype for insurance claim processing.

This system automatically:

* Extracts structured data from claim PDFs (OCR/NLP)
* Predicts claim risk using Machine Learning
* Retrieves semantically similar historical cases (RAG)
* Generates executive-level risk summaries (LLM-ready)
* Presents results via an interactive Gradio UI

> Designed as a hybrid AI decision-support system for insurance analytics.

---

# 🧩 System Architecture (High-Level)

1. **Input:** Claim PDF
2. **Extraction:** pdfplumber → OCR fallback (Tesseract)
3. **Entity Parsing:** Regex-based structured extraction
4. **Feature Mapping:** Convert entities into ML feature schema
5. **Prediction:** RandomForest risk classification
6. **RAG:** Vector similarity search on historical cases
7. **Summary:** LLM (Ollama) or rule-based fallback
8. **Output:** Structured table + risk label + summary + evidence

---

# 🏗️ Technical Architecture

```text
            ┌─────────────────────────┐
            │     Claim PDF Input     │
            └────────────┬────────────┘
                         │
              ┌──────────▼───────────┐
              │   PDF Extraction     │
              │ pdfplumber / OCR     │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │   Entity Extraction  │
              │   (Regex / NLP)      │
              └──────────┬───────────┘
                         │
        ┌────────────────▼────────────────┐
        │ Feature Engineering & Mapping   │
        └────────────────┬────────────────┘
                         │
              ┌──────────▼───────────┐
              │  ML Risk Classifier  │
              │  RandomForest        │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │   RAG Retrieval      │
              │ SentenceTransformer  │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │   LLM Summary        │
              │ (Ollama / Fallback)  │
              └──────────┬───────────┘
                         │
            ┌────────────▼─────────────┐
            │  Gradio UI Output Layer  │
            └──────────────────────────┘
```

---

# ✨ Key Features

### 📄 PDF Text Extraction + OCR Fallback

* `pdfplumber` for selectable text
* `pytesseract + pdf2image` for scanned PDFs

### 🧠 Entity Extraction

Extracts:

* Claimant name
* Claim amount (RM)
* Incident type
* Location (State)
* Policy number
* Date of incident

### 🤖 Risk Level Prediction

* `RandomForestClassifier`
* OneHotEncoder + ColumnTransformer pipeline
* Output: Low / Medium / High risk

### 🔎 RAG (Retrieval-Augmented Generation)

* `sentence-transformers (all-MiniLM-L6-v2)`
* Cosine similarity search
* Top-k similar historical cases retrieved

### 📝 Executive Summary Generation

* Ollama (llama3) if available
* Automatic fallback summary if LLM unavailable

### 🎛 Interactive Gradio UI

* Upload PDF
* View structured extraction
* View risk classification
* Inspect similar historical cases
* Read AI executive summary

---

# Example Output

### 📊 Extracted Structured Data Table

![Data Extracted](https://raw.githubusercontent.com/azlinaaaa/AI-Powered-Risk-Assessment-and-Claims-Analytics-Platform/067e7baca1b08d32a8019f0a70a8f7b7cbf13a1e/Output/Data_Extracted.png)

---

### ✅ AI Executive Summary Output

![AI Summary](https://raw.githubusercontent.com/azlinaaaa/AI-Powered-Risk-Assessment-and-Claims-Analytics-Platform/067e7baca1b08d32a8019f0a70a8f7b7cbf13a1e/Output/AI_Summary.png)

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

Install Tesseract:

* Windows → Install & add to PATH
* Mac → `brew install tesseract`
* Ubuntu → `sudo apt install tesseract-ocr`

---

# ▶️ Run Application

```bash
python app.py
```

Open the local URL shown in terminal.

---

# 🧠 Model Training

```bash
python train_model.py
```

Output:

```
risk_model.pkl
```

---

# 📊 Model Explainability

Risk prediction is influenced by:

* Claim amount
* Incident type
* State / region risk score
* Historical claim frequency
* Fraud flag
* Customer tenure

Future upgrades:

* SHAP explainability
* Confidence scoring
* Model drift monitoring

---

# 🧠 Design Philosophy

This system follows a **hybrid AI architecture**:

* Deterministic parsing (regex)
* Supervised learning (ML classifier)
* Semantic retrieval (embeddings)
* Optional generative reasoning (LLM)

This ensures:

* Reliability
* Reduced hallucination
* Interpretability
* Enterprise readiness

---

# 🔐 Enterprise & Production Considerations

* PII encryption required
* Role-based access control
* Audit logging
* Monitoring for model drift
* Secure cloud deployment

---

# 🚀 Future Enhancements

* Replace regex with LayoutLM / Document AI
* Add SHAP feature importance
* Convert to FastAPI backend + React frontend
* Deploy via Docker
* Add fraud anomaly detection model
* Database-backed case management
* Cloud-native monitoring pipeline

---

# 🧾 Tech Stack

* Python (pandas, scikit-learn, joblib)
* OCR: pdfplumber, pytesseract
* RAG: sentence-transformers
* UI: Gradio
* LLM: Ollama (llama3)

---
