import google.generativeai as genai
from pypdf import PdfReader
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
import json

# ------------------------------
# Load Google API key
# ------------------------------
load_dotenv(find_dotenv())
google_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_key)

# Use a valid model
model = genai.GenerativeModel("models/gemini-2.5-flash-lite")

# ------------------------------
# PDF Text Extraction Function
# ------------------------------
def get_pdf_text(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ------------------------------
# LLM Extraction Function
# ------------------------------
def extracted_data(pages_data):
    """Send text to the model and get a JSON-formatted dictionary."""
    prompt = f"""
Extract the following values clearly from this bill text:

- Invoice ID
- DESCRIPTION
- Issue Date
- UNIT PRICE
- AMOUNT
- Bill For
- From
- Terms

Text:
{pages_data}

Return ONLY a valid JSON object.
Use double quotes, all keys must appear even if value is missing (use null).
Do not include any text outside the JSON.
"""
    response = model.generate_content(prompt)
    output_text = response.text.strip()

    # Sometimes the model returns extra text around JSON, try to extract JSON only
    try:
        # Find first { and last }
        start = output_text.find("{")
        end = output_text.rfind("}") + 1
        json_text = output_text[start:end]
        data_dict = json.loads(json_text)
    except Exception:
        data_dict = {}  # fallback if JSON invalid

    return data_dict
# ------------------------------
# Main function to handle multiple PDFs
# ------------------------------
def create_docs(user_pdf_list):
    df = pd.DataFrame(columns=[
        'Invoice ID', 'DESCRIPTION', 'Issue Date',
        'UNIT PRICE', 'AMOUNT', 'Bill For', 'From', 'Terms'
    ])
    failed_files = []

    for pdf_file in user_pdf_list:
        try:
            raw_text = get_pdf_text(pdf_file)
            if not raw_text.strip():
                failed_files.append(pdf_file.name)
                continue

            data_dict = extracted_data(raw_text)
            if data_dict:
                df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
            else:
                failed_files.append(pdf_file.name)

        except Exception:
            failed_files.append(pdf_file.name)

    return df, failed_files



