import google.generativeai as genai
from pypdf import PdfReader
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
import json

# Load Google API key
load_dotenv(find_dotenv())
google_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_key)

# Use a valid model
model = genai.GenerativeModel("models/gemini-2.5-flash-lite")

def get_pdf_text(pdf_doc):
    """Extract text from a PDF file."""
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

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
    """
    response = model.generate_content(prompt)
    return response.text

def create_docs(user_pdf_list):
    """Process multiple PDFs and return a dataframe of extracted data."""
    df = pd.DataFrame(columns=[
        'Invoice ID', 'DESCRIPTION', 'Issue Date',
        'UNIT PRICE', 'AMOUNT', 'Bill For', 'From', 'Terms'
    ])

    for pdf_file in user_pdf_list:
        raw_data = get_pdf_text(pdf_file)
        llm_output = extracted_data(raw_data)

        # Parse JSON safely
        try:
            data_dict = json.loads(llm_output)
        except json.JSONDecodeError:
            data_dict = {}  # skip or handle error if model output is invalid

        # Add to dataframe if dictionary is not empty
        if data_dict:
            df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)

    return df
