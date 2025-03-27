import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Load spaCy model
import en_core_web_md
nlp = en_core_web_md.load()
#nlp = spacy.load("en-core-web-sm")
print("Model loaded successfully!")

# Add the SpacyTextBlob pipeline
nlp.add_pipe('spacytextblob')

# Load dataset
data = pd.read_csv("amazon_product_reviews.csv")

# Preprocess text data
reviews_data = data['reviews.text']
clean_data = reviews_data.dropna()  # Remove missing reviews

# Define sentiment analysis function
def analyze_sentiment(review):
    doc = nlp(review)
    return doc._.blob.polarity  # Correct way to access sentiment polarity

# Test sentiment analysis function on sample reviews
sample_reviews = clean_data.head()
for review in sample_reviews:
    sentiment = analyze_sentiment(review)
    print(f"Review: {review}\nSentiment Polarity: {sentiment}\n")

# Select two reviews by indexing
review_1 = clean_data.iloc[0]  # First review
review_2 = clean_data.iloc[1]  # Second review

# Convert reviews to spaCy Doc objects
doc1 = nlp(review_1)
doc2 = nlp(review_2)

# Calculate similarity
similarity_score = doc1.similarity(doc2)

# Output the reviews and their similarity score
print(f"\nReview 1: {review_1}")
print(f"\nReview 2: {review_2}")
print(f"\nSimilarity Score: {similarity_score}")

    
