"""
app.py
------
AI Claim Analysis & Risk Assessment System (Malaysia)
- Extract text from claim PDF (pdfplumber, OCR fallback)
- Extract structured entities (regex)
- Predict risk (trained ML model)
- Retrieve similar historical cases (RAG embeddings)
- Generate executive summary (Ollama llama3 if available; fallback always works)
- Gradio UI for interactive demo

Run:
    python app.py
"""

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import gradio as gr

import pdfplumber
import pytesseract
from pdf2image import convert_from_path

from sentence_transformers import SentenceTransformer


# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "risk_model.pkl"
KNOWLEDGE_BASE_PATH = "knowledge_base_100_cases.txt"

# If Tesseract isn't in PATH, set it manually (Windows example):
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -----------------------------
# 1) PDF Text Extraction
# -----------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF.
    1) Try selectable text extraction via pdfplumber.
    2) If not enough text, fallback to OCR via pytesseract + pdf2image.
    """
    text = ""

    # (A) Try normal text extraction
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                text += t + "\n"
    except Exception:
        # If pdfplumber fails (corrupt PDF etc.), fallback to OCR
        text = ""

    # If we got enough text, return it
    if len(text.strip()) > 50:
        return text

    # (B) OCR fallback for scanned PDFs
    try:
        images = convert_from_path(pdf_path, dpi=200)
        ocr_text = ""
        for img in images:
            ocr_text += pytesseract.image_to_string(img) + "\n"
        return ocr_text
    except Exception:
        return ""


# -----------------------------
# 2) Entity Extraction (Regex)
# -----------------------------
def extract_entities(text: str) -> dict:
    """
    Extract key claim entities from raw text using simple regex rules.
    NOTE: This is a prototype approach. In production, use layout-aware extraction.
    """
    def find(pattern, flags=re.IGNORECASE):
        m = re.search(pattern, text, flags)
        return m.group(1).strip() if m else None

    # These patterns are generic; adjust based on your real PDF format
    claimant_name = find(r"Claimant\s*Name\s*[:\-]\s*(.+)")
    policy_number = find(r"Policy\s*(No\.?|Number)\s*[:\-]\s*([A-Za-z0-9\-\/]+)")  # group 2 if match
    if policy_number:
        # If pattern captured group 2, policy_number may contain both; handle quickly:
        policy_number = policy_number.split()[-1]

    incident_type = find(r"(Incident\s*Type|Type\s*of\s*Incident)\s*[:\-]\s*(.+)")
    location = find(r"(Location|State)\s*[:\-]\s*(.+)")
    date_of_incident = find(r"(Date\s*of\s*Incident|Incident\s*Date)\s*[:\-]\s*([0-9]{4}\-[0-9]{2}\-[0-9]{2})")

    # Claim amount: match RM 12,345.67 or 12345
    amount_raw = None
    m_amt = re.search(r"(Claim\s*Amount|Amount\s*Claimed)\s*[:\-]\s*(RM)?\s*([0-9\.,]+)", text, re.IGNORECASE)
    if m_amt:
        amount_raw = m_amt.group(3)

    claim_amount = None
    if amount_raw:
        try:
            claim_amount = float(amount_raw.replace(",", ""))
        except Exception:
            claim_amount = None

    return {
        "claimant_name": claimant_name,
        "policy_number": policy_number,
        "incident_type": incident_type,
        "location": location,
        "date_of_incident": date_of_incident,
        "claim_amount": claim_amount
    }


# -----------------------------
# 3) Feature Mapping (Entities -> Model Input)
# -----------------------------
def map_to_features(entities: dict) -> pd.DataFrame:
    """
    Convert extracted entities into a feature row that matches the training schema.
    IMPORTANT: Your training dataset columns must align with the keys here.
    """
    # Extract state from location like "Johor Bahru, Johor"
    state = None
    if entities.get("location"):
        parts = [p.strip() for p in str(entities["location"]).split(",")]
        state = parts[-1] if len(parts) >= 2 else parts[0]

    # Parse incident date
    year, month = 2025, 1
    if entities.get("date_of_incident"):
        try:
            dt = datetime.strptime(entities["date_of_incident"], "%Y-%m-%d")
            year, month = dt.year, dt.month
        except Exception:
            pass

    # Build a single-row dataframe with default safe values
    row = {
        "year": year,
        "month": month,
        "state": state or "Selangor",
        "incident_type": (entities.get("incident_type") or "Property Damage").replace(" Damage", ""),
        "incident_severity": "Medium",
        "policy_type": "Individual",
        "sector": "Retail",
        "region_risk_score": 3,
        "previous_claims": 0,
        "claim_frequency_1yr": 0,
        "customer_tenure_years": 3,
        "fraud_flag": 0,
        "claim_amount": entities.get("claim_amount") or 10000.0
    }
    return pd.DataFrame([row])


# -----------------------------
# 4) Load ML Model
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        f"Please train first: python train_model.py"
    )

clf = joblib.load(MODEL_PATH)


# -----------------------------
# 5) RAG: Similar Case Retrieval
# -----------------------------
if not os.path.exists(KNOWLEDGE_BASE_PATH):
    raise FileNotFoundError(
        f"Knowledge base not found: {KNOWLEDGE_BASE_PATH}\n"
        f"Create a text file containing cases for retrieval."
    )

kb_text = open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8", errors="ignore").read()

# Split cases by "Case " delimiter (simple heuristic)
cases = [c.strip() for c in kb_text.split("Case ") if c.strip()]
cases = ["Case " + c for c in cases]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
case_emb = embedder.encode(cases, normalize_embeddings=True)


def search_cases(query: str, top_k: int = 3):
    """
    Retrieve top-k similar cases using cosine similarity on normalized embeddings.
    """
    q_emb = embedder.encode([query], normalize_embeddings=True)
    scores = (case_emb @ q_emb[0])
    idx = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), cases[i][:700]) for i in idx]


# -----------------------------
# 6) GenAI Summary (Ollama optional)
# -----------------------------
def generate_ai_summary(entities: dict, pred_risk: str, similar_cases: list) -> str:
    """
    Generate executive summary:
    - Try local Ollama llama3 if available.
    - Else fallback to rule-based summary (always works).
    """
    claimant = entities.get("claimant_name") or "N/A"
    incident = entities.get("incident_type") or "N/A"
    amount = entities.get("claim_amount")
    location = entities.get("location") or "N/A"
    policy = entities.get("policy_number") or "N/A"
    date_inc = entities.get("date_of_incident") or "N/A"

    top_cases = "\n\n".join(
        [f"[{i+1}] (sim={s:.3f}) {snip[:350]}" for i, (s, snip) in enumerate(similar_cases)]
    )

    prompt = f"""
You are an insurance risk analyst assistant.
Generate a concise executive summary (6-10 bullet points max) based on the claim details, predicted risk, and similar historical cases.
Also include:
- Key risk drivers
- Suggested next actions
- Any red flags

CLAIM DETAILS:
- Claimant: {claimant}
- Policy Number: {policy}
- Incident Type: {incident}
- Claim Amount (RM): {amount if amount is not None else "N/A"}
- Location: {location}
- Date of Incident: {date_inc}

MODEL OUTPUT:
- Predicted Risk Level: {pred_risk}

SIMILAR CASES (snippets):
{top_cases}
""".strip()

    # Try Ollama (works only if ollama server is running locally)
    try:
        import requests
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False},
            timeout=20
        )
        if resp.status_code == 200:
            data = resp.json()
            out = data.get("response", "").strip()
            if out:
                return out
    except Exception:
        pass

    # Fallback summary
    drivers = []
    if isinstance(amount, (int, float)) and amount >= 50000:
        drivers.append("High claim amount (≥ RM50,000).")
    if str(pred_risk).lower() == "high":
        drivers.append("Model classified as high risk based on claim characteristics.")
    if "fire" in str(incident).lower():
        drivers.append("Fire-related incidents tend to have higher severity and loss volatility.")
    if "flood" in str(incident).lower():
        drivers.append("Flood-related losses can be complex; verify coverage and evidence.")

    if not drivers:
        drivers.append("No strong red flags detected from extracted fields; proceed with standard validation.")

    if str(pred_risk).lower() == "high":
        action = "Escalate for manual review; request full evidence (police report, photos, receipts)."
    elif str(pred_risk).lower() == "medium":
        action = "Perform enhanced validation checks; verify key fields and coverage."
    else:
        action = "Proceed with standard processing and routine checks."

    sim_summary = "\n".join(
        [f"- Similar case (sim={s:.3f}) indicates comparable historical pattern." for s, _ in similar_cases[:2]]
    ) if similar_cases else "- No similar cases found."

    return f"""### Executive Summary (Auto)
- **Predicted Risk:** {pred_risk}
- **Incident:** {incident} | **Amount (RM):** {amount if amount is not None else "N/A"} | **Location:** {location} | **Policy:** {policy}
- **Key Risk Drivers:**
  - {drivers[0]}
{('  - ' + drivers[1]) if len(drivers) > 1 else ''}
- **Similar Case Signals:**
{sim_summary}
- **Recommended Next Action:** {action}
"""


# -----------------------------
# 7) Main Analysis Function (UI Hook)
# -----------------------------
def analyze_pdf(file):
    """
    Gradio callback:
    - Read PDF
    - Extract entities
    - Predict risk
    - Retrieve similar cases
    - Generate summary
    - Return UI outputs
    """
    pdf_path = file.name

    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Extract entities via regex
    entities = extract_entities(text)

    # Feature mapping -> model input
    feat = map_to_features(entities)

    # Predict risk
    pred_risk = clf.predict(feat)[0]

    # RAG query
    q = f"{pred_risk} risk {entities.get('incident_type')} claim amount {entities.get('claim_amount')} location {entities.get('location')}"
    hits = search_cases(q, top_k=3)

    # Convert entities to a clean table
    ent_df = pd.DataFrame([{
        "Claimant Name": entities.get("claimant_name", ""),
        "Incident Type": entities.get("incident_type", ""),
        "Claim Amount (RM)": entities.get("claim_amount", ""),
        "Location": entities.get("location", ""),
        "Policy Number": entities.get("policy_number", ""),
        "Date of Incident": entities.get("date_of_incident", "")
    }]).fillna("")

    # Render RAG results as Markdown
    rag_md = "## 🔎 Top Similar Cases (RAG)\n"
    for idx, (s, snip) in enumerate(hits, start=1):
        rag_md += f"\n### Case Match {idx} (Similarity: **{s:.3f}**)\n"
        rag_md += f"```text\n{snip}\n```\n"

    # Generate summary
    ai_summary = generate_ai_summary(entities, str(pred_risk), hits)

    # Raw text preview (first 2000 chars)
    return ent_df, str(pred_risk), ai_summary, rag_md, (text[:2000] if text else "")


# -----------------------------
# 8) Gradio UI
# -----------------------------
def build_ui():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")) as demo:
        gr.Markdown("""
# 🏦 **AI Claim Analysis & Risk Assessment System**
Automated claim document analysis powered by **OCR/NLP**, **Machine Learning**, and **RAG**.
""")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📂 Upload Document")
                file_input = gr.File(label="Upload Claim PDF File", file_types=[".pdf"])
                run_btn = gr.Button("START ANALYSIS", variant="primary")

                gr.Markdown("### ⚠️ Risk Summary")
                risk_output = gr.Textbox(
                    label="Predicted Risk Level",
                    placeholder="Risk classification will appear here...",
                    interactive=False
                )

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("📊 Extracted Entities"):
                        entity_table = gr.Dataframe(label="Structured Information", interactive=False)

                    with gr.TabItem("🧠 AI Summary"):
                        summary_out = gr.Markdown()

                    with gr.TabItem("🔎 Similar Historical Cases (RAG)"):
                        rag_output = gr.Markdown()

                    with gr.TabItem("📄 Raw Extracted Text"):
                        raw_text = gr.Textbox(label="Text Preview", lines=15, interactive=False)

        run_btn.click(
            fn=analyze_pdf,
            inputs=file_input,
            outputs=[entity_table, risk_output, summary_out, rag_output, raw_text]
        )

        gr.Markdown("---\n© 2026 Enterprise Risk Intelligence Platform | Confidential")

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=True)
