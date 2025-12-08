import streamlit as st
import pandas as pd
from helper import create_docs

def main():
    st.title("AI Bill Extractor")

    pdfs = st.file_uploader("Upload Bills", type=["pdf"], accept_multiple_files=True)

    if st.button("Extract"):
        if not pdfs:
            st.warning("Upload at least one file.")
            return
        
        with st.spinner("Extracting..."):
            df = create_docs(pdfs)

        if df.empty:
            st.error("No data extracted. Check bill format.")
        else:
            st.success("Done!")
            st.dataframe(df)

            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "extracted_bills.csv",
                "text/csv"
            )

if __name__ == "__main__":
    main()
