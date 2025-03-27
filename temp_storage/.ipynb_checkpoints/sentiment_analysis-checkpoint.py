
import numpy as np
import pandas as pd
import spacy #This statement should work if you have spaCy installed 
nlp = spacy.load('en_core_web_sm')
sample = u"Build your data science skills to launch an in-demand, valuable career in six months."
doc = nlp(sample)

# Load file
df = pd.read_csv('amazon_product_reviews.csv')

