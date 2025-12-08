from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader
import pandas as pd
import json
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
google_key = os.getenv("GEMINI_API_KEY")


def get_pdf_text(pdf_doc):
    """Extract text from each PDF page."""
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extracted_data(pages_data):
    """LLM extraction using modern LangChain pipeline."""
    template = """
    Extract the following fields from this invoice text:
    Invoice ID, DESCRIPTION, Issue Date, UNIT PRICE, AMOUNT, Bill For, From, Terms.

    Text:
    {pages}

    Output pure JSON only. Example:
    {{
        "Invoice ID": "1001329",
        "DESCRIPTION": "Some item",
        "Issue Date": "5/4/2023",
        "UNIT PRICE": "2",
        "AMOUNT": "1100.00",
        "Bill For": "James",
        "From": "Excel Company",
        "Terms": "Pay this now"
    }}
    """

    prompt = PromptTemplate(
        input_variables=["pages"],
        template=template
    )

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

    chain = prompt | model | StrOutputParser()

    return chain.invoke({"pages": pages_data})


def create_docs(user_pdf_list):
    df = pd.DataFrame(columns=[
        'Invoice ID', 'DESCRIPTION', 'Issue Date',
        'UNIT PRICE', 'AMOUNT', 'Bill For', 'From', 'Terms'
    ])

    for file in user_pdf_list:
        raw_text = get_pdf_text(file)
        response = extracted_data(raw_text)

        try:
            cleaned = json.loads(response)
            df = pd.concat([df, pd.DataFrame([cleaned])], ignore_index=True)
        except:
            pass

    return df
