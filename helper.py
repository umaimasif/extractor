# langgraph_helper.py
"""
A lightweight LangGraph-style pipeline for your bill extractor.
Nodes: LoadPDF -> ExtractText -> LLMExtract -> ParseJSON -> CollectDF
Uses google.generativeai for LLM calls (configure GEMINI_API_KEY in env or Streamlit secrets).
"""

import json
import os
from typing import Any, Dict, List
from pypdf import PdfReader
import pandas as pd
from dotenv import find_dotenv, load_dotenv

# LLM client
import google.generativeai as genai

load_dotenv(find_dotenv())
GENAI_KEY = os.getenv("GEMINI_API_KEY")
if GENAI_KEY:
    genai.configure(api_key=GENAI_KEY)

# -------------------------
# Minimal LangGraph engine
# -------------------------
class Node:
    def run(self, input_data: Any) -> Any:
        raise NotImplementedError

class Graph:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes

    def run(self, input_data: Any) -> Any:
        data = input_data
        for n in self.nodes:
            data = n.run(data)
        return data

# -------------------------
# Nodes implementation
# -------------------------
class LoadPDFNode(Node):
    """Input: list of uploaded file-likes -> Output: list of {'name', 'file', 'raw_text'}"""
    def run(self, uploaded_files):
        items = []
        for f in uploaded_files:
            try:
                f.seek(0)
            except Exception:
                pass
            items.append({"name": getattr(f, "name", "unknown.pdf"), "file": f})
        return items

class ExtractTextNode(Node):
    """Input: list of items -> Output: list of items with 'raw_text' key"""
    def run(self, items):
        out = []
        for it in items:
            try:
                reader = PdfReader(it["file"])
                txt = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        txt += page_text + "\n"
            except Exception:
                txt = ""
            it2 = dict(it)
            it2["raw_text"] = txt
            out.append(it2)
        return out

class LLMExtractNode(Node):
    """
    Input: list of items with raw_text -> Output: list with 'llm_response' (string)
    Uses google.generativeai model to request JSON output from the invoice text.
    """
    def __init__(self, model="gemini-pro", temperature=0.0):
        self.model = model
        self.temperature = temperature

    def _build_prompt(self, text):
        return f"""
Extract the following fields from this invoice text: Invoice ID, DESCRIPTION, Issue Date, UNIT PRICE, AMOUNT, Bill For, From, Terms.

Text:
{text}

Return only valid JSON that contains those keys (use empty string for missing values). Example:
{{"Invoice ID":"1001329","DESCRIPTION":"...","Issue Date":"5/4/2023","UNIT PRICE":"...","AMOUNT":"1100.00","Bill For":"James","From":"Excel Company","Terms":"Pay now"}}
"""

    def run(self, items):
        out = []
        for it in items:
            text = it.get("raw_text", "")
            prompt = self._build_prompt(text)
            # call Google GenAI (official client)
            try:
                # Use the generative API - interface may vary by genai version
                # We use model.generate_content(prompt) pattern (works with many genai releases)
                model = genai.GenerativeModel(self.model)
                resp = model.generate_content(prompt)
                llm_text = getattr(resp, "text", "") or str(resp)
            except Exception as e:
                # fallback: empty response
                llm_text = ""
            it2 = dict(it)
            it2["llm_response"] = llm_text
            out.append(it2)
        return out

class ParseJSONNode(Node):
    """Parses LLM output (string) into a dict; tolerates extra text, tries to find first JSON object."""
    def _find_json(self, text: str):
        text = text.strip()
        # Look for first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                # try to clean common issues: single quotes -> double quotes
                try:
                    candidate2 = candidate.replace("'", '"')
                    return json.loads(candidate2)
                except Exception:
                    return None
        return None

    def run(self, items):
        out = []
        for it in items:
            parsed = {}
            try:
                parsed = self._find_json(it.get("llm_response", "") or "")
                if parsed is None:
                    parsed = {}
            except Exception:
                parsed = {}
            it2 = dict(it)
            it2["parsed"] = parsed
            out.append(it2)
        return out

class CollectDFNode(Node):
    """Collect parsed dicts into a single DataFrame"""
    def __init__(self, columns=None):
        self.columns = columns or ['Invoice ID', 'DESCRIPTION', 'Issue Date', 'UNIT PRICE', 'AMOUNT', 'Bill For', 'From', 'Terms']

    def run(self, items):
        rows = []
        for it in items:
            parsed = it.get("parsed") or {}
            # ensure keys exist
            row = {k: parsed.get(k, "") for k in self.columns}
            rows.append(row)
        df = pd.DataFrame(rows, columns=self.columns)
        return df

# -------------------------
# Convenience function to run full graph
# -------------------------
def run_invoice_extraction_graph(uploaded_files, model="gemini-pro"):
    nodes = [
        LoadPDFNode(),
        ExtractTextNode(),
        LLMExtractNode(model=model, temperature=0.0),
        ParseJSONNode(),
        CollectDFNode()
    ]
    graph = Graph(nodes)
    return graph.run(uploaded_files)
