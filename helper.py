import os
import re
import pandas as pd
from pypdf import PdfReader
from dotenv import load_dotenv, find_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

load_dotenv(find_dotenv())
google_key = os.getenv("GEMINI_API_KEY")


# -------------------------------
# PDF TEXT READER
# -------------------------------
def get_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# -------------------------------
# STATE for LangGraph
# -------------------------------
class State(dict):
    text: str
    response: str


# -------------------------------
# NODE 1 — LLM extraction
# -------------------------------
def extract_node(state):
    template = """
Extract these fields from the bill text:
- Bill No
- Account No
- Billing Date
- Total Amount Due
- Customer Name
- Address
- Units Consumed
- Energy Charges
- Taxes

Bill text:
{bill_text}

Return ONLY JSON like this:
{
  "Bill No": "",
  "Account No": "",
  "Billing Date": "",
  "Total Amount Due": "",
  "Customer Name": "",
  "Address": "",
  "Units Consumed": "",
  "Energy Charges": "",
  "Taxes": ""
}
"""
    prompt = PromptTemplate(
        input_variables=["bill_text"],
        template=template
    )

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

    result = llm.invoke(prompt.format(bill_text=state["text"]))

    state["response"] = result
    return state


# -------------------------------
# BUILD LANGGRAPH
# -------------------------------
graph = StateGraph(State)
graph.add_node("extract", extract_node)
graph.set_entry_point("extract")
graph.set_finish_point("extract")

workflow = graph.compile()


# -------------------------------
# MAIN FUNCTION CALLED BY STREAMLIT
# -------------------------------
def create_docs(pdf_files):
    df = pd.DataFrame()

    for file in pdf_files:
        text = get_pdf_text(file)
        output = workflow.invoke({"text": text})

        try:
            clean_json = re.search(r"{(.+?)}", output["response"], re.S).group(0)
            data_dict = eval(clean_json)
            df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
        except:
            pass

    return df
