from openai import OpenAI
import fitz
from .logger import get_logger
import os
import io
import re
from PIL import Image
import pytesseract
from typing import List
import hashlib
import docx2txt
import json




logger = get_logger()

API_KEY = os.getenv("OPENAI_API_KEY")

def ocr_page(page: fitz.Page, zoom: float = 2.0) -> str:
    mat = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=mat, alpha=False)  # type: ignore
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)

def blocks_text(tp: fitz.TextPage) -> str:
    blocks = tp.extractBLOCKS() 
    blocks.sort(key=lambda b: (round(b[1], 2), round(b[0], 2)))
    parts = [(b[4] or "").strip() for b in blocks if (b[4] or "").strip()]
    return "\n\n".join(parts)

def normalize(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"[‘’]", "'", s)
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)     # de-hyphenate soft wraps
    s = re.sub(r"\n{3,}", "\n\n", s)           # collapse excess breaks
    return s

def extract_text_content(file_path):
    content = ""
    try:
        if file_path.endswith("txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        elif file_path.endswith("pdf"):
            content = extract_pdf_text(file_path)
        elif file_path.endswith("docx"):
            content = extract_docx_text(file_path)

        client = OpenAI(api_key=API_KEY)
        prompt = (
            f"""
            Examine the attached technology transactions document and then identify respective values for each of the following keys: source_url (url of the document), license, type of technology transactions document (doc_type, use only one or two words, use abbreviations where appropriate), the role of the party or parties (use only the roles, e.g., licensor, licensee; do not include the names of the parties), the jurisdiction, the governing law, the industry (use only one or two words), and the effective date of the agreement. Consider the substantive meaning of words (e.g., "Page 1 of 12" is not likely to be the license), placement in the document, surrounding text, applicable section, and any other factors that might inform your decision. Use standard legal abbreviations for the jurisdiction and governing law (e.g., US-CA for California state law, US-FED for US federal law, etc.). Format the date in a string "YYYY-MM-DD".
            Return a response in following JSON format only: 
            {{
                "source_url": source_url, 
                "license": license, 
                "doc_type": doc_type, 
                "party_role": party_role, 
                "jurisdiction": jurisdiction, 
                "governing_law": governing_law, 
                "industry": industry, 
                "effective_date": effective_date
            }}. 
            Do not return anything else outside of the JSON object. If you cannot identify a distinct value corresponding to one of the keys, respond with null for that key.
            --- PAGE TEXT START ---\n
            {content}\n
            --- PAGE TEXT END ---
            """
        )

        ai_model = "gpt-5-mini"

        response = client.chat.completions.create(
            model=ai_model,
            messages=[
                {"role": "system", "content": "Extract the document's source_url, license, type of technology transactions document, the role of the party or parties, the jurisdiction, the governing law, the industry, and the version or effective date based from the following text.\n"},
                {"role": "user", "content": prompt},
            ]
        )

        response = response.choices[0].message.content if response.choices else []
        logger.info(f"AI manifest raw response: {response}")
        candidate = (response or [])
        logger.info(f"AI manifest response: {candidate}")
        if not candidate:
            return "Unknown"
        
        return candidate

    except Exception as e:
        logger.error(f"Error extracting title from {file_path}: {e}")
        return "Unknown"

def get_manifest_info(file_path) -> dict:
    """Return structured manifest info as a dict.

    Uses the raw AI response (expected JSON) from extract_text_content and
    normalizes it into a dict with the required keys. Missing or unparsable
    values become None.
    """
    raw = extract_text_content(file_path)

    expected_keys = [
        "source_url",
        "license",
        "doc_type",
        "party_role",
        "jurisdiction",
        "governing_law",
        "industry",
        "effective_date",
    ]

    data = {}
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception:
            logger.warning("AI manifest response not valid JSON; returning defaults")
            data = {}
    else:
        logger.warning("AI manifest response unexpected type %s", type(raw))

    # Build normalized manifest dict
    manifest = {k: (data.get(k) if isinstance(data, dict) else None) for k in expected_keys}

    # Coerce obvious placeholder / unknown strings to None
    for k, v in manifest.items():
        if isinstance(v, str) and v.strip().lower() in {"unknown", "null", "n/a", "none", ""}:
            manifest[k] = None

    return manifest

def extract_pdf_text(file_path):
    doc = fitz.open(file_path)
    out_parts: List[str] = []
    for page in doc:
        tp = page.get_textpage()
        txt = blocks_text(tp).strip()
        if not txt:
            txt = ocr_page(page).strip()
        out_parts.append(txt)
        tp = None
    doc.close()
    raw = "\n\n\f\n\n".join(out_parts).strip()
    return normalize(raw)

def extract_docx_text(file_path):
    text = docx2txt.process(file_path)
    return normalize(text)

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_pdf_text(file_path)
    elif ext == ".docx":
        return extract_docx_text(file_path)
    elif ext == ".txt":
        return normalize(open(file_path, "r", encoding="utf-8").read())
    else:
        raise ValueError("Unsupported file format")

def extract_title(text: str) -> str:
    try:
        client = OpenAI(api_key=API_KEY)
        prompt = (
            f"""
            Examine the attached technology transactions document and then identify the title. Consider the substantive meaning of words (e.g., "Page 1 of 12" is not likely to be the title), placement in the document, proximity to the beginning of the document, case of the letters, length, justficiation, and any other factors that might inform your decision. Return only the title, with no other characters. If you cannot confidently identify a distinct value for any field, respond with Unknown."
            --- PAGE TEXT START ---\n
            {text}\n
            --- PAGE TEXT END ---
            """
        )

        ai_model = "gpt-5-nano"

        response = client.chat.completions.create(
            model=ai_model,
            messages=[
                {"role": "system", "content": "Extract the document's title based from the following text.\n"},
                {"role": "user", "content": prompt},
            ]
        )

        response = response.choices[0].message.content if response.choices else ""
        candidate = (response or "").strip()
        if not candidate:
            return "Unknown"
        return candidate
    except Exception as e:
        logger.error(f"Error extracting title from text: {e}")
        return "Unknown"

def sha256_file(file) -> str:
    file.seek(0)
    return sha256_bytes(file.read())

def sha256_bytes(bytes: bytes) -> str:
    return hashlib.sha256(bytes).hexdigest()

def sha256_text(text: str) -> str:
    """Compute the SHA-256 hash of a text string."""
    return sha256_bytes(text.encode('utf-8'))

