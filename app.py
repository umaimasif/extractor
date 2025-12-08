import streamlit as st
import pandas as pd
from helper import create_docs, get_pdf_text, extracted_data
import json

def main():
    st.title("AI Bill Extractor")

    # Upload multiple PDFs
    pdfs = st.file_uploader("Upload Bills", type=["pdf"], accept_multiple_files=True)

    if st.button("Extract"):
        if not pdfs:
            st.warning("Upload at least one file.")
            return
        
        failed_files = []
        all_dataframes = []

        with st.spinner("Extracting..."):
            for pdf_file in pdfs:
                try:
                    # Extract data from each PDF
                    raw_text = get_pdf_text(pdf_file)
                    llm_output = extracted_data(raw_text)
                    data_dict = json.loads(llm_output)

                    # Add to list if extraction successful
                    if data_dict:
                        all_dataframes.append(pd.DataFrame([data_dict]))
                    else:
                        failed_files.append(pdf_file.name)

                except Exception as e:
                    # Catch parsing errors
                    failed_files.append(pdf_file.name)

            # Combine all successful extractions
            if all_dataframes:
                df = pd.concat(all_dataframes, ignore_index=True)
                st.success("Done!")
                st.dataframe(df)

                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    "extracted_bills.csv",
                    "text/csv"
                )
            else:
                st.error("No data extracted. Check bill format.")

        # Show failed files
        if failed_files:
            st.warning(f"Failed to extract data from: {', '.join(failed_files)}")

if __name__ == "__main__":
    main()
