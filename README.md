# 🏦 AI Claim Analysis & Risk Assessment System (Malaysia)

An end-to-end **Document Intelligence + Predictive Analytics + RAG + Generative AI** prototype for insurance claim processing.  
This system automatically extracts key fields from claim PDFs (OCR/NLP), predicts claim risk level using ML, retrieves similar historical cases using embeddings (RAG), and generates an executive summary for faster decision-making.

> **Tech Focus:** OCR + NLP extraction • ML Risk Classification • Retrieval-Augmented Generation (RAG) • LLM Summary • Gradio UI MVP

---

## ✨ Key Features

✅ **PDF Text Extraction + OCR Fallback**  
- Uses `pdfplumber` for selectable text extraction  
- Falls back to `pytesseract + pdf2image` when PDF is scanned/image-based

✅ **Entity Extraction (NLP Parsing)**  
Extracts structured fields:
- Claimant name
- Claim amount (RM)
- Incident type
- Location (State)
- Policy number
- Date of incident

✅ **Risk Level Prediction (Machine Learning)**  
- Trains a `RandomForestClassifier` on Malaysia-calibrated claims dataset  
- Predicts `risk_level` = `Low / Medium / High`  
- Pipeline includes `OneHotEncoder` + numeric passthrough with `ColumnTransformer`

✅ **RAG: Similar Historical Case Retrieval**  
- Uses `sentence-transformers (all-MiniLM-L6-v2)` embeddings  
- Retrieves top similar cases from `knowledge_base_100_cases.txt`

✅ **Executive Summary Generation (GenAI-ready)**  
- Attempts LLM generation via **Ollama (llama3)** if available  
- Auto fallback summary always works (no LLM required)

✅ **Interactive UI (Gradio MVP)**  
- Upload PDF → Extract entities → Predict risk → Show similar cases → Summary

---

## 🧩 System Architecture (High-level)

1. **Input:** Claim PDF  
2. **Extraction:** pdfplumber → OCR fallback (Tesseract)  
3. **Entity Parsing:** Regex extraction → structured fields  
4. **Feature Mapping:** Convert entities → ML model features  
5. **Prediction:** Risk classification (Low/Medium/High)  
6. **RAG:** Retrieve similar past cases using embeddings  
7. **Summary:** LLM (optional) or rules-based fallback  
8. **Output:** Table + risk label + summary + RAG evidence + raw text preview

---

## 📂 Project Structure

```text
.
├── app.py                         # Gradio app (main)
├── train_model.py                 # ML training script (saves risk_model.pkl)
├── risk_model.pkl                 # Trained model artifact (generated)
├── claims_dataset_malaysia_calibrated_10000.csv   # Training dataset (optional)
├── knowledge_base_100_cases.txt   # RAG knowledge base (cases)
├── pdfs/                          # Sample PDFs (optional)
├── requirements.txt               # Dependencies
└── README.md
````

---

## 🛠️ Installation

### 1) Create environment (optional but recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Install OCR dependencies (Tesseract)

**Windows:** Install Tesseract from official installer and add it to PATH
**Mac:** `brew install tesseract`
**Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`

---

## ▶️ Run the Gradio App

```bash
python app.py
```

Then open the local URL shown in terminal.

---

## 📌 Usage

1. Upload a claim PDF in the UI
2. System extracts entities (name, amount, incident, etc.)
3. Predicts risk level (`Low/Medium/High`)
4. Shows top similar historical cases (RAG)
5. Generates an executive summary (LLM if available, else fallback)

---

## 🧠 Model Training

To retrain the model:

```bash
python train_model.py
```

Output:

* `risk_model.pkl`

---

## 📊 Example Output

* **Predicted Risk Level:** Low
* **Extracted Fields:** Claimant, policy no, incident type, location, amount, date
* **Similar Cases (RAG):** Top 3 matches with similarity scores
* **Summary:** Risk drivers + recommended next actions

---

## 🧾 Tech Stack

* **Python:** pandas, scikit-learn, joblib
* **OCR & PDF:** pdfplumber, pytesseract, pdf2image
* **RAG:** sentence-transformers, cosine similarity
* **UI:** Gradio
* **LLM:** Ollama (llama3)
