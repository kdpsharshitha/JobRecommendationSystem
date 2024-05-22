# Job-Recommender

# Set up Environment for the project

- Download the required dependencies used in the project using, pip install -r requirements.txt

# Project Hierarchy
- The  Datasets/ directory contains all the required datasets for the project. It mainly consists of two job datasets one for IT and other for   Non-IT. 
- The Jobs_Data\ directory consists urls's to apply for jobs
- Data preprocessing and natural language processing is done in code/data_cleaning.py.
- Saved cleaned data to a new directory Cleaned_Datasets/ 
- Job recommender System is built in Code/model.py

# Data cleaning
nltk library used. The following pipeline has been executed.
- remove all non alphabets regex = [^a-zA-Z], 
- remove whitespaces
- convert case to lowercase 
- tokenize words
- remove stopwords
- stemming
The result saved to Cleaned_Dataset/ directory.

# How to Use:
- run below commands

# Series of commands
- cd Code/
- python data_cleaning.py
- streamlit run model.py 
