# app.py
import streamlit as st
import pandas as pd
from helper import run_invoice_extraction_graph

def main():
    st.set_page_config(page_title="Bill Extractor (LangGraph)")
    st.title("Bill Extractor — LangGraph pipeline 🤖")

    pdf_files = st.file_uploader(
        "Upload your bills (PDF only)",
        type=["pdf"],
        accept_multiple_files=True
    )

    model_name = st.text_input("LLM model name (optional)", value="gemini-pro")
    if st.button("Extract bill data"):
        if not pdf_files:
            st.warning("Please upload at least one PDF.")
            return

        with st.spinner("Running LangGraph pipeline..."):
            df = run_invoice_extraction_graph(pdf_files, model=model_name)

        if df is None or df.empty:
            st.error("No structured data could be extracted.")
            return

        st.dataframe(df)
        # safe numeric conversion
        df["AMOUNT"] = pd.to_numeric(df["AMOUNT"], errors="coerce")
        st.write("Average amount:", df["AMOUNT"].mean())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "extracted_bills.csv", "text/csv")
        st.success("Done.")

if __name__ == "__main__":
    main()

