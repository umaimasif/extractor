import streamlit as st
import pandas as pd
from helper import create_docs

def main():
    st.set_page_config(page_title="Bill Extractor")
    st.title("Bill Extractor AI Assistant...🤖")

    pdf_files = st.file_uploader(
        "Upload your bills in PDF format only",
        type=["pdf"],
        accept_multiple_files=True
    )

    extract_button = st.button("Extract bill data...")

    if extract_button:
        if not pdf_files:
            st.warning("Please upload at least one PDF file.")
            return

        with st.spinner("Extracting... this may take a minute..."):
            try:
                data_frame = create_docs(pdf_files)
                if data_frame.empty:
                    st.error("No data could be extracted from the uploaded PDFs.")
                    return

                st.write(data_frame.head())

                data_frame["AMOUNT"] = pd.to_numeric(data_frame["AMOUNT"], errors='coerce')
                st.write("Average bill amount: ", data_frame['AMOUNT'].mean())

                csv_data = data_frame.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download data as CSV",
                    csv_data,
                    "CSV_Bills.csv",
                    "text/csv",
                    key="download-csv"
                )

                st.success("Extraction successful!")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()



