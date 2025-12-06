from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate  # updated for v1.1.2
from pypdf import PdfReader
import pandas as pd
import re
import os
from dotenv import find_dotenv, load_dotenv

# Load environment variables
load_dotenv(find_dotenv())
google_key = os.getenv("GEMINI_API_KEY")

# Extract text from PDF
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract data using LLM
def extracted_data(pages_data):
    template = """Extract all the following values : Invoice ID, DESCRIPTION, Issue Date, 
    UNIT PRICE, AMOUNT, Bill For, From and Terms from: {pages}

    Expected output: remove any dollar symbols {{'Invoice ID': '1001329','DESCRIPTION': 'UNIT PRICE','AMOUNT': '2','Date': '5/4/2023','AMOUNT': '1100.00', 'Bill For': 'james', 'From': 'excel company', 'Terms': 'pay this now'}}"""
    
    prompt_template = PromptTemplate(input_variables=["pages"], template=template)
    llm = ChatGoogleGenerativeAI(temperature=0.7)
    full_response = llm(prompt_template.format(pages=pages_data))
    
    return full_response

# Create DataFrame from uploaded PDFs
def create_docs(user_pdf_list):
    df = pd.DataFrame(columns=[
        'Invoice ID', 'DESCRIPTION', 'Issue Date',
        'UNIT PRICE', 'AMOUNT', 'Bill For', 'From', 'Terms'
    ])

    for filename in user_pdf_list:
        raw_data = get_pdf_text(filename)
        llm_extracted_data = extracted_data(raw_data)

        # Parse dictionary from LLM response
        pattern = r'{(.+)}'
        match = re.search(pattern, llm_extracted_data, re.DOTALL)
        if match:
            extracted_text = match.group(1)
            data_dict = eval('{' + extracted_text + '}')
        else:
            data_dict = {}
        
        if data_dict:
            df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)

    return df
