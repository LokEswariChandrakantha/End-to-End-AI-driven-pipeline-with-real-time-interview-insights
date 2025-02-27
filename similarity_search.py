import streamlit as st
from datetime import datetime
import pandas as pd
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time
from services.synthesizer import Synthesizer
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
# Initialize VectorStore
vec = VectorStore()

# Streamlit app
st.title("Job Description and Resume Analyzer")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel or PDF file", type=["xlsx", "pdf"])

if uploaded_file is not None:
    # Read the Excel file
    try:
        df = pd.read_excel(uploaded_file)
        st.write("File uploaded successfully.")
    except FileNotFoundError:
        st.error("Error: The specified file was not found.")
        st.stop()

    # Prepare data for insertion
    def prepare_record(row):
        content = f"JD NAME: {row.get('JD NAME', '')}\nJob Description: {row.get('JD', '')}\nRESUME: {row.get('RESUME', '')}\nInterview_Details: {row.get('Q AND A', '')}"
        embedding_1 = vec.get_embedding(content)
        return {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "Acceptance": row.get("TAG", "Unknown"),
                "Category": row.get("JD NAME", "Unknown"),
                "created_at": datetime.now().isoformat(),
            },
            "contents": content,
            "embedding": embedding_1,
        }

    records_df = pd.DataFrame([prepare_record(row) for _, row in df.iterrows()])

    # Create tables and insert data
    vec.create_tables()
    vec.create_index()
    vec.upsert(records_df)

    st.write("Data insertion complete.")

    # Define the question
    relevant_question = st.text_input("Enter the relevant question:")

    if relevant_question:
        # Perform search
        results = vec.search(relevant_question, limit=3)
        if not results:
            st.error("No relevant results found.")
        else:
            response = Synthesizer.generate_response(question=relevant_question, context=results)

            # Display response
            st.write(response.answer)

            # Generate PDF
            pdf_filename = "suitability_report.pdf"
            doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
            styles = getSampleStyleSheet()
            content = [Paragraph(response.answer, styles['Normal'])]
            doc.build(content)

            st.write(f"PDF generated: {pdf_filename}")
            with open(pdf_filename, "rb") as pdf_file:
                st.download_button("Download PDF", data=pdf_file, file_name=pdf_filename)
else:
    st.write("Upload an Excel file to get started.")