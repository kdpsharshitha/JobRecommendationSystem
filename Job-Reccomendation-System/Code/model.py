import re
import sys
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import streamlit as st
import pdfplumber
from pathlib import Path
from utils.cosine_similarity import cal_cosine_similarity
from utils.TfidfVectorizer import calculate_tfidf
from utils.IDF import calculate_idf

csv_path1 = Path(__file__).parents[1] / 'Cleaned_Datasets/IT_Dataset.csv'
csv_path2 = Path(__file__).parents[1] / 'Cleaned_Datasets/NonIT_Dataset.csv'
csv_path3 = Path(__file__).parents[1] / 'Jobs_Data/jobs_url.csv'

def extract_data(feed): 
    text = ''
    with pdfplumber.open(feed) as pdf:
        pages = pdf.pages
        for page in pages:
            text += page.extract_text(x_tolerance=2)
    return text

st.title("Job Recommender System")
st.markdown('<h4 style="color:blue;">User Information Form</h4>', unsafe_allow_html=True)

p1, p2 = st.columns((4, 4))
with p1:
    name = st.text_input("Enter your name:", key='name_input')

with p2:
    email = st.text_input("Enter your email:", key='email_input')

c1,c2 = st.columns((4,4))
with c1: option = st.selectbox(
    'Select the Industry',
    ('IT', 'NON-IT'))

if option == "IT":
    path = csv_path1
elif option == "NON-IT":
    path = csv_path2
else:
    st.error("Invalid value for option")
    st.stop()

jobs = pd.read_csv(path)
labels = jobs["Query"].unique()

with c2: interest = st.selectbox("Select your interests",(labels))
if option not in ['IT', 'NON-IT']:
    st.error("Invalid value for option. Please select a valid industry.")
    st.stop()

st.markdown('---')
st.subheader("Options")
st.write("Select your industry and interest.")

st.sidebar.subheader("USER INFORMATION")
st.sidebar.write(f"Name: {name}")
st.sidebar.write(f"Email: {email}")


st.sidebar.subheader("OPTIONS")
st.sidebar.write(f"Industry: {option}")
st.sidebar.write(f"Intrest: {interest}")

all_stopwords = stopwords.words('english')
ps = PorterStemmer()



cv = st.file_uploader('Upload your Resume', type='pdf')

if not name or not email or not option or not interest or not cv:
    st.error("Please fill all the required information.")
    st.stop()
with st.spinner('Analyzing your resume...'):
    cvtext = extract_data(cv)
    prediction_text = str(cvtext)
    prediction_text = re.sub('[^a-zA-Z]', ' ', prediction_text)
    prediction_text = prediction_text.lower()
    prediction_text = prediction_text.split()
    prediction_text = [ps.stem(word) for word in prediction_text if not word in set(all_stopwords)]
    prediction_text = ' '.join(prediction_text)
    all_words = ' '.join(jobs['Description']).lower().split()
    unique_words = set(all_words)
    print(unique_words)
    idf_values = calculate_idf(unique_words, jobs['Description'])
    updated_unique_words = [term for term in unique_words if idf_values[term] != 0]
    tfidf_jobs = []
    for document in jobs['Description']:
        tfidf_vector = calculate_tfidf(document, updated_unique_words, idf_values)
        tfidf_jobs.append(tfidf_vector)
    tfidf_prediction_text = calculate_tfidf(prediction_text, updated_unique_words, idf_values)
    similarity_measure = cal_cosine_similarity(tfidf_prediction_text, tfidf_jobs)
    print("similarity measure: \n", similarity_measure)

    similarity_scores = {label: {"sum": 0, "count": 0} for label in labels} 
    for i in range(len(similarity_measure)):
        similarity_scores[jobs["Query"][i]]["sum"] += similarity_measure[i]
        similarity_scores[jobs["Query"][i]]["count"] += 1

    predictions = []
    for label in similarity_scores:
        avg = similarity_scores[label]["sum"]/similarity_scores[label]["count"]
        predictions.append([avg, label])
    predictions.sort(key = lambda key: -key[0])
    output_text = f"\nTop 3 predictions for you in {option} Industry:\n" + '\n'.join("\n" + x[1] for x in predictions[:3])
    print(output_text)

    st.success(f"Hey {name}! Here are your job recommendations:")
    st.markdown('<h6 style="color:blue;">Your job recommendations are:</h6>', unsafe_allow_html=True)
    st.write(output_text)

    joburls = pd.read_csv(csv_path3)
    st.success(f"Top 5 Job hirings open for you in {option} Industry:")
    for word in predictions[:3]:
        temp = joburls[joburls['Title'].str.contains(word[1])]
        if temp['URL'].size != 0:
            st.write(f"\nFor {word[1]}: \n")
            st.write(temp['URL'].head(5)) 
    matchi = joburls[joburls['Title'].str.contains(interest)]
    if matchi['URL'].size != 0:
        st.success(f"Top 5 job hirings open for you in your interest area ({interest}):")
        st.write(f"\nFor {interest}: \n")
        st.write(matchi['URL'].head(5)) 

st.success('Done!Land in your dream job with ease!')