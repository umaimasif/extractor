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
Rules:
1. All keys must be present exactly as listed.
2. Use double quotes for strings.
3. No extra text outside the JSON.
4. If a value is missing, use null.

Example output:
{{
  "Invoice ID": "12345",
  "DESCRIPTION": "Product X",
  "Issue Date": "2025-12-09",
  "UNIT PRICE": 100.0,
  "AMOUNT": 300.0,
  "Bill For": "Customer",
  "From": "Company",
  "Terms": "Net 30"
}}
"""
    response = model.generate_content(prompt)
    return response.text

# ------------------------------
# Main function to handle multiple PDFs
# ------------------------------
def create_docs(user_pdf_list):
    """Process multiple PDFs and return a dataframe of extracted data."""
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

            llm_output = extracted_data(raw_text)

            # Safely parse JSON
            try:
                data_dict = json.loads(llm_output)
            except json.JSONDecodeError:
                failed_files.append(pdf_file.name)
                continue

            if data_dict:
                df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)

        except Exception:
            failed_files.append(pdf_file.name)

    return df, failed_files


