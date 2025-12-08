import google.generativeai as genai
from pypdf import PdfReader
import pandas as pd
import re
import os
from dotenv import load_dotenv, find_dotenv

# Load Google API key
load_dotenv(find_dotenv())
google_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_key)

# Use a valid model
model = genai.GenerativeModel("gemini-pro")

def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extracted_data(pages_data):
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

    Return ONLY a Python dictionary.
    """
    response = model.generate_content(prompt)
    return response.text

def create_docs(user_pdf_list):
    df = pd.DataFrame(columns=[
        'Invoice ID', 'DESCRIPTION', 'Issue Date',
        'UNIT PRICE', 'AMOUNT', 'Bill For', 'From', 'Terms'
    ])

    for filename in user_pdf_list:
        raw_data = get_pdf_text(filename)
        llm_output = extracted_data(raw_data)

        pattern = r'{(.+)}'
        match = re.search(pattern, llm_output, re.DOTALL)

        if match:
            extracted_text = match.group(1)
            data_dict = eval('{' + extracted_text + '}')
            df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)

    return df




