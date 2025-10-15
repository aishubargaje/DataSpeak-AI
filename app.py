from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import sqlite3
import pandas as pd
import google.generativeai as genai
from transformers import pipeline

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize local fallback model (Code Llama)
try:
    local_llm = pipeline("text-generation", model="codellama/CodeLlama-7b-Instruct-hf")
    fallback_ready = True
except Exception:
    fallback_ready = False

# ------------------ FUNCTIONS ------------------
def get_sql_query(question, table_name, columns):
    """
    Try Gemini first. If quota exceeded, fallback to local Code Llama.
    """
    prompt = f"""
    You are an expert in converting English questions to SQL queries.
    Table name: {table_name}
    Columns: {columns}
    Important:
    - Only return SQL query.
    - Do NOT include ``` or 'sql' tags.
    """

    # Try Gemini
    try:
        model = genai.GenerativeModel('models/gemini-2.5-pro')
        response = model.generate_content([prompt, question])
        return response.text.strip(), "Gemini"
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            if fallback_ready:
                st.warning("⚠️ Gemini quota exceeded — using Code Llama fallback.")
                llm_input = prompt + "\nQuestion: " + question
                local_resp = local_llm(llm_input, max_length=256, do_sample=True)[0]["generated_text"]
                return local_resp.strip(), "Code Llama"
            else:
                return "Error: Gemini quota exceeded and no fallback model found.", "None"
        else:
            return f"Error: {str(e)}", "None"

def run_query(sql, conn):
    try:
        df = pd.read_sql_query(sql, conn)
        return df
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

# ------------------ STREAMLIT ------------------
st.set_page_config(page_title="DataSpeak AI", layout="wide")
st.title("🧠 DataSpeak AI")
st.caption("Translate Natural Language into SQL Queries using Gemini + Code Llama")

uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.dataframe(df.head())

    # Create in-memory SQLite DB
    conn = sqlite3.connect(":memory:")
    df.to_sql("DATA", conn, index=False, if_exists="replace")

    columns = ", ".join(df.columns)
    question = st.text_input("💬 Ask your question in English:")

    if st.button("🔍 Generate SQL and Run"):
        with st.spinner("Generating SQL query..."):
            sql_query, model_used = get_sql_query(question, "DATA", columns)
            st.subheader(f"🧠 Generated SQL Query ({model_used}):")
            st.code(sql_query, language="sql")

            result_df = run_query(sql_query, conn)
            st.subheader("📊 Query Results:")
            st.dataframe(result_df)
else:
    st.info("⬆️ Upload a CSV file to begin.")

st.markdown("---")
st.caption("✨ Developed by Aishwarya Bargaje | Gemini Text-to-SQL")