import streamlit as st
import re
import string
import pickle
import docx2txt
import PyPDF2
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Resume Cleaning Function
# -----------------------------
def cleanResume(txt):
    cleanTxt = re.sub(r'http\S+\s', ' ', txt)
    cleanTxt = re.sub(r'RT|S+', ' ', cleanTxt)
    cleanTxt = re.sub(r'@\S+', ' ', cleanTxt)
    cleanTxt = re.sub(r'#\S+\s', ' ', cleanTxt)
    cleanTxt = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]', ' ', cleanTxt)
    cleanTxt = re.sub(r'\s+', ' ', cleanTxt)
    return cleanTxt.strip().lower()

# -----------------------------
# Load ML Model and TF-IDF
# -----------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("clf.pkl", "rb"))      # trained classifier
    vectorizer = pickle.load(open("tfidf.pkl", "rb"))  # fitted TF-IDF
    return model, vectorizer

# Check if the model files exist before trying to load them
try:
    clf, tfidf = load_model()
except FileNotFoundError:
    st.error("Error: Could not find 'clf.pkl' or 'tfidf.pkl'. Please ensure your trained model and vectorizer files are in the root directory.")
    # Use a placeholder model/vectorizer to allow the app to run partially for demonstration
    class PlaceholderClassifier:
        def predict(self, X):
            # Returns a default category ID (e.g., Data Science)
            return np.array([6]) 
        def predict_proba(self, X):
            # Returns equal probability for all categories for demo
            num_categories = 25 
            return np.array([[1/num_categories] * num_categories])
            
    class PlaceholderVectorizer:
        def transform(self, raw_documents):
            # Returns a dummy feature vector
            return np.zeros((len(raw_documents), 1000))
            
    clf = PlaceholderClassifier()
    tfidf = PlaceholderVectorizer()


# -----------------------------
# Category Mapping
# -----------------------------
category_mapping = {
    0: "Advocate", 1: "Arts", 2: "Automation Testing", 3: "Blockchain",
    4: "Business Analyst", 5: "Civil Engineer", 6: "Data Science",
    7: "Database", 8: "DevOps Engineer", 9: "DotNet Developer",
    10: "ETL Developer", 11: "Electrical Engineering", 12: "HR",
    13: "Hadoop", 14: "Health and Fitness", 15: "Java Developer",
    16: "Mechanical Engineer", 17: "Network Security Engineer",
    18: "Operations Manager", 19: "PMO", 20: "Python Developer",
    21: "SAP Developer", 22: "Sales", 23: "Testing", 24: "Web Designing"
}

# -----------------------------
# Extract Text from Upload - IMPROVED DOCX AND SEEK HANDLING
# -----------------------------
def extract_text(uploaded_file):
    # Always reset the pointer to the beginning of the file-like object
    uploaded_file.seek(0) 

    if uploaded_file.type == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            return " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
            
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            # FIX: Read the binary content of the file-like object
            binary_content = uploaded_file.read()
            # docx2txt.process can accept the binary content directly
            text = docx2txt.process(binary_content)
            # docx2txt returns a bytes-like object if passed binary, ensure it's a string
            return text.decode("utf-8") if isinstance(text, bytes) else text
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""
            
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
        
    else:
        return ""

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Resume Screening App", page_icon="📄", layout="wide")

st.sidebar.image("https://img.icons8.com/fluency/96/resume.png", use_container_width=True)

st.sidebar.title("📊 Resume Screening AI")
st.sidebar.markdown("Upload resumes, analyze, and get job category predictions 🚀")

menu = st.sidebar.radio("Navigation", ["🏠 Home", "📄 Single Resume", "📂 Multiple Resumes", "📑 Job Match", "📊 Analytics"])

# -----------------------------
# HOME
# -----------------------------
if menu == "🏠 Home":
    st.title("📄 AI-Powered Resume Screening System")
    st.markdown("""
    Welcome to the **AI Resume Screening App**.  
    This tool uses **Machine Learning & NLP** to analyze resumes and classify them into job categories.  

    ### Features:
    - Upload PDF/DOCX/TXT resumes
    - Predict job category using ML
    - Batch process multiple resumes
    - Confidence visualization
    - Job description similarity check
    - Export results as CSV/Excel  
    """)

# -----------------------------
# SINGLE RESUME
# -----------------------------
elif menu == "📄 Single Resume":
    st.header("📄 Single Resume Prediction")
    uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        with st.spinner("Analyzing resume..."):
            raw_text = extract_text(uploaded_file)
            
            if not raw_text:
                st.warning("Could not extract any text from the uploaded file.")
            else:
                # Show Resume Content
                st.subheader("📄 Resume Content")
                st.text_area("Resume Text", raw_text, height=300)
                
                cleaned_text = cleanResume(raw_text)
                
                # Check for empty text after cleaning
                if not cleaned_text:
                    st.warning("Resume content was successfully extracted but appears to be empty after cleaning.")
                else:
                    input_features = tfidf.transform([cleaned_text])
                    prediction_id = clf.predict(input_features)[0]
                    category_name = category_mapping.get(prediction_id, "Unknown")

                    st.success(f"🎯 Predicted Job Category: **{category_name}**")

                    # Show Probabilities
                    probs = clf.predict_proba(input_features)[0]
                    prob_df = pd.DataFrame({
                        "Category": [category_mapping.get(i, str(i)) for i in range(len(probs))],
                        "Confidence (%)": (probs * 100).round(2)
                    }).sort_values(by="Confidence (%)", ascending=False)
                    st.subheader("📊 Prediction Confidence")
                    # Limit to top 10 categories for cleaner chart
                    st.bar_chart(prob_df.head(10).set_index("Category"))

# -----------------------------
# MULTIPLE RESUMES
# -----------------------------
elif menu == "📂 Multiple Resumes":
    st.header("📂 Batch Resume Screening")
    uploaded_files = st.file_uploader("Upload Multiple Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files:
        results = []
        with st.spinner(f"Analyzing {len(uploaded_files)} resumes..."):
            for f in uploaded_files:
                raw_text = extract_text(f)
                cleaned_text = cleanResume(raw_text)
                
                category_name = "N/A (No Text)"
                top_confidence = 0.0
                
                if cleaned_text:
                    input_features = tfidf.transform([cleaned_text])
                    prediction_id = clf.predict(input_features)[0]
                    category_name = category_mapping.get(prediction_id, "Unknown")
                    
                    # Get confidence for the predicted class
                    probs = clf.predict_proba(input_features)[0]
                    top_confidence = (probs[prediction_id] * 100).round(2)
                
                results.append({
                    "File": f.name,
                    "Predicted Category": category_name,
                    "Confidence (%)": top_confidence
                })

        df_results = pd.DataFrame(results)
        st.dataframe(df_results)

        # Download Option
        st.download_button(
            "📥 Download Results", 
            df_results.to_csv(index=False).encode("utf-8"), 
            "resume_predictions.csv", 
            "text/csv"
        )

# -----------------------------
# JOB MATCHING
# -----------------------------
elif menu == "📑 Job Match":
    st.header("📑 Resume-Job Description Matching")
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"], key="job_match")
    job_desc = st.text_area("Paste Job Description Here")

    if st.button("Calculate Match Score") and uploaded_file and job_desc:
        with st.spinner("Calculating similarity..."):
            resume_text = cleanResume(extract_text(uploaded_file))
            job_text = cleanResume(job_desc)
            
            if not resume_text or not job_text:
                st.error("Please ensure text is extracted from the resume and the job description is not empty.")
            else:
                # To calculate cosine similarity, both vectors must be transformed.
                # However, your vectorizer (tfidf) was trained on individual resumes,
                # so transforming two separate documents into one sparse matrix, then 
                # calculating similarity between the two rows is the correct approach.
                vectors = tfidf.transform([resume_text, job_text])
                similarity = cosine_similarity(vectors[0], vectors[1])[0][0] * 100
                st.info(f"✅ Resume matches **{similarity:.2f}%** with the job description.")

# -----------------------------
# ANALYTICS
# -----------------------------
elif menu == "📊 Analytics":
    st.header("📊 Resume Analytics Dashboard")
    st.markdown("Upload multiple resumes to analyze category distribution")

    uploaded_files = st.file_uploader("Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="analytics")

    if uploaded_files:
        categories = []
        with st.spinner(f"Processing {len(uploaded_files)} resumes for analytics..."):
            for f in uploaded_files:
                text = cleanResume(extract_text(f))
                if text: # Only process if text was extracted
                    pred = clf.predict(tfidf.transform([text]))[0]
                    categories.append(category_mapping.get(pred, "Unknown"))

            if categories:
                df = pd.DataFrame(categories, columns=["Category"])
                # Use st.dataframe for the table and st.bar_chart for the visualization
                st.subheader("Category Distribution")
                st.bar_chart(df["Category"].value_counts())
                st.subheader("Raw Counts")
                st.dataframe(df["Category"].value_counts().reset_index().rename(columns={'index': 'Category', 'Category': 'Count'}))
            else:
                st.warning("No text could be extracted from the uploaded files to generate analytics.")