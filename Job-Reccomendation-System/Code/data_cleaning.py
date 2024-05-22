import re
import nltk
from nltk import data
import numpy as np
import pandas as pd
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

print("Starting Data preprocessing phase ...")

# Reading IT Dataset
dataset_IT = pd.read_csv("../Datasets/IT_Dataset.csv")
dataset_IT = dataset_IT.loc[:, ["Query", "Description"]]

all_stopwords = stopwords.words('english')

print("Processing IT dataset ...")

for i in range(len(dataset_IT)):
    # Remove non-alphabetic characters and replace them with spaces.
    description = re.sub('[^a-zA-Z]', ' ', dataset_IT["Description"][i]) 
    # Converting to lower case
    description = description.lower()
    # Tokenize the description into list of words
    description = description.split()
    # Apply stemming using the Porter Stemmer
    ps = PorterStemmer()
    description = [ps.stem(word) for word in description if not word in set(all_stopwords)]
    # join stemmed words back into single string
    description = ' '.join(description)
    # update the it data set 
    dataset_IT["Description"][i] = description

dataset_IT.to_csv("../Cleaned_Datasets/IT_Dataset.csv", index=False)

print("IT dataset successfully cleaned! ..")

dataset_nonIT = pd.read_csv("../Datasets/NonIT_Dataset.csv")
dataset_nonIT = dataset_nonIT.loc[:, ["Query", "Description"]]

print("Processing NONIT Dataset ...")

for i in range(len(dataset_nonIT)):
    description = re.sub('[^a-zA-Z]', ' ', dataset_nonIT['Description'][i])
    description = description.lower()
    description = description.split()
    ps = PorterStemmer()
    description = [ps.stem(word) for word in description if not word in set(all_stopwords)]
    description = ' '.join(description)
    dataset_nonIT["Description"][i] = description
    
dataset_nonIT.to_csv("../Cleaned_Datasets/NonIT_Dataset.csv",index=False)

print("NONIT dataset successfully cleaned! ..")

print("Preprocessing phase finished!")