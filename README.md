# 📄 AI-Powered Resume Screening App

**🚀 Live Demo:** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://harshchoudhary2003-resume-screening-app-app-wv6fcl.streamlit.app/)

An interactive **Streamlit web app** that uses **Machine Learning and NLP** to analyze resumes and classify them into job categories. This tool facilitates the initial screening process by allowing batch processing of resumes, job description matching, and insightful analytics visualization.

---

## **Features** ✨

* Upload **PDF, DOCX, or TXT** resumes for processing.
* **Single Resume Prediction:** Classify a resume into a job category and display the model's confidence scores.
* **Batch Processing:** Upload multiple resumes and receive a CSV/Excel exportable table of predictions.
* **Job Description Matching:** Calculate a **cosine similarity percentage** between a resume and a given job description.
* **Analytics Dashboard:** Visualize the category distribution of a batch of uploaded resumes using bar charts.

---

## **Job Categories Supported** 📊

| ID | Category |
| :--- | :--- |
| 0 | Advocate |
| 1 | Arts |
| 2 | Automation Testing |
| 3 | Blockchain |
| 4 | Business Analyst |
| 5 | Civil Engineer |
| 6 | **Data Science** |
| 7 | Database |
| 8 | DevOps Engineer |
| 9 | DotNet Developer |
| 10 | ETL Developer |
| 11 | Electrical Engineering |
| 12 | HR |
| 13 | Hadoop |
| 14 | Health and Fitness |
| 15 | **Java Developer** |
| 16 | Mechanical Engineer |
| 17 | Network Security Engineer |
| 18 | Operations Manager |
| 19 | PMO |
| 20 | **Python Developer** |
| 21 | SAP Developer |
| 22 | Sales |
| 23 | Testing |
| 24 | Web Designing |

---

## **Installation and Setup** ⚙️

To run this application locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/HarshChoudhary2003/resume-screening-app.git](https://github.com/HarshChoudhary2003/resume-screening-app.git)
    cd resume-screening-app
    ```

2.  **Install Dependencies:** Ensure you have Python installed, then use the provided `requirements.txt` file to install all necessary libraries, including `streamlit`, `scikit-learn`, `pandas`, and file-parsers.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App:** Launch the Streamlit application from your terminal.

    ```bash
    streamlit run app.py
    ```

    The app will automatically open in your web browser.

---

## **Project Structure**

| File/Folder | Description |
| :--- | :--- |
| `app.py` | **Main Streamlit application file.** Contains all UI logic, functions, model loading, and predictions. |
| `requirements.txt` | Lists all necessary Python packages for installation (e.g., `streamlit`, `scikit-learn`, `PyPDF2`, `docx2txt`). |
| `clf.pkl` | The **trained Machine Learning Classifier** model (used for category prediction). |
| `tfidf.pkl` | The **fitted TF-IDF Vectorizer** object (used to convert resume text into numerical features). |
| `README.md` | This file. |
