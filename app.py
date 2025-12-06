# streamlit_bill_extractor.py

import streamlit as st
from pypdf import PdfReader
import pandas as pd
import re
import ast
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
from dotenv import find_dotenv, load_dotenv

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv(find_dotenv())
google_key = os.getenv("GEMINI_API_KEY")

# ----------------------------
# Helper functions
# ----------------------------

def get_pdf_text(pdf_doc):
    """Extract all text from a PDF file"""
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def extracted_data(pages_data):
    """Use LLM to extract structured bill data from text"""
    template = """Extract all the following values: Invoice ID, DESCRIPTION, Issue Date, 
UNIT PRICE, AMOUNT, Bill For, From and Terms from: {pages}

Expected output: remove any dollar symbols 
{{'Invoice ID': '1001329','DESCRIPTION': 'UNIT PRICE','AMOUNT': '2','Date': '5/4/2023',
'AMOUNT': '1100.00', 'Bill For': 'james', 'From': 'excel company', 'Terms': 'pay this now'}}
"""
    prompt_template = PromptTemplate(input_variables=["pages"], template=template)
    llm = ChatGoogleGenerativeAI(temperature=0.7)
    return llm(prompt_template.format(pages=pages_data))

def create_docs(user_pdf_list):
    """Process uploaded PDFs and return a DataFrame with extracted data"""
    df = pd.DataFrame(columns=[
        'Invoice ID','DESCRIPTION','Issue Date','UNIT PRICE','AMOUNT','Bill For','From','Terms'
    ])

    for pdf_file in user_pdf_list:
        pdf_file.seek(0)  # Reset pointer
        raw_data = get_pdf_text(pdf_file)
        if not raw_data.strip():
            st.warning(f"No text found in {pdf_file.name}. Skipping.")
            continue

        llm_output = extracted_data(raw_data)
        pattern = r'({.*})'
        match = re.search(pattern, llm_output, re.DOTALL)

        if match:
            try:
                data_dict = ast.literal_eval(match.group(1))
                df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
            except Exception as e:
                st.warning(f"Failed to parse data from {pdf_file.name}: {e}")
        else:
            st.warning(f"No structured data found in {pdf_file.name}.")

    return df

# ----------------------------
# Streamlit App
# ----------------------------

def main():
    st.set_page_config(page_title="Bill Extractor")
    st.title("Bill Extractor AI Assistant 🤖")

    # Upload Bills
    pdf_files = st.file_uploader(
        "Upload your bills in PDF format only",
        type=["pdf"],
        accept_multiple_files=True
    )

    extract_button = st.button("Extract bill data")

    if extract_button:
        if not pdf_files:
            st.warning("Please upload at least one PDF file!")
            return

        with st.spinner("Extracting data... This may take a while."):
            data_frame = create_docs(pdf_files)

        if data_frame.empty:
            st.error("No data extracted from uploaded PDFs.")
        else:
            # Ensure 'AMOUNT' is numeric for average calculation
            data_frame['AMOUNT'] = pd.to_numeric(data_frame['AMOUNT'], errors='coerce')
            st.subheader("Extracted Data Preview:")
            st.dataframe(data_frame.head())

            avg_amount = data_frame['AMOUNT'].mean()
            st.write(f"**Average bill amount:** {avg_amount:.2f}")

            # Convert to CSV for download
            csv_data = data_frame.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download data as CSV",
                csv_data,
                "Bills_Extracted.csv",
                "text/csv",
                key="download-csv"
            )

            st.success("Extraction complete!")

# Run the app
if __name__ == "__main__":
    main()




