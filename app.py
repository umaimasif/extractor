import streamlit as st
import pandas as pd
from helper import create_docs

def main():
    st.set_page_config(page_title="Bill Extractor")
    st.title("Bill Extractor AI Assistant 🤖")

    pdf_files = st.file_uploader(
        "Upload your bills (PDF only)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Extract bill data"):
        if not pdf_files:
            st.warning("Please upload at least one PDF.")
            return

        with st.spinner("Extracting…"):
            try:
                data_frame = create_docs(pdf_files)

                if data_frame.empty:
                    st.error("No data extracted.")
                    return

                st.dataframe(data_frame)

                # numeric conversion
                data_frame["AMOUNT"] = pd.to_numeric(
                    data_frame["AMOUNT"], errors="coerce"
                )

                st.write("Average amount:", data_frame["AMOUNT"].mean())

                csv = data_frame.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    csv,
                    "extracted_bills.csv",
                    "text/csv"
                )

                st.success("Done! Extraction successful.")

            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

