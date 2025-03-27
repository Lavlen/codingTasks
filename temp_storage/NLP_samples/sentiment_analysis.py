# Import libraries
import pandas as pd
import spacy
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#% matplotlib inline

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords


# Load spaCy model
nlp = spacy.load("en_core_web_md")
nlp.add_pipe('spacytextblob')

# Load dataset
data = pd.read_csv("amazon_product_reviews.csv")
data.head()




