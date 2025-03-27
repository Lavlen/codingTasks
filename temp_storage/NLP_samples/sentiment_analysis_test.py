import pandas as pd
import numpy as np
import spacy
import en_core_web_sm
from spacytextblob.spacytextblob import SpacyTextBlob

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')

# Load dataset
data = pd.read_csv("amazon_product_reviews.csv")

# Preprocess text data
reviews_data = data['reviews.text']
clean_data = reviews_data.dropna()  # Remove missing reviews

# Define sentiment analysis function
def analyze_sentiment(review):
    doc = nlp(review)
    return doc._.polarity  # Extract sentiment polarity

# Test sentiment analysis function on sample reviews
sample_reviews = clean_data.head()
for review in sample_reviews:
    sentiment = analyze_sentiment(review)
    print(f"Review: {review}\nSentiment Polarity: {sentiment}\n")
