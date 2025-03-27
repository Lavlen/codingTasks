import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Load spaCy model
import en_core_web_md
nlp = en_core_web_md.load()
print("Model loaded successfully!")

# Add the SpacyTextBlob pipeline
nlp.add_pipe('spacytextblob')

# Load dataset
data = pd.read_csv("amazon_product_reviews.csv")

# Preprocess text data function
def preprocess_text(review):
    """
    Cleans and processes the review text by:
    - Converting to lowercase
    - Stripping leading and trailing whitespace
    - Removing stop words
    - Ensuring the text is treated as a string
    """
    doc = nlp(str(review).lower().strip())  # Convert to lowercase, strip spaces, and ensure it's a string
    processed_tokens = [token.text for token in doc if not token.is_stop]  # Remove stop words
    return " ".join(processed_tokens)  # Return cleaned text

# Preprocess all reviews
reviews_data = data['reviews.text']
clean_data = reviews_data.dropna().apply(preprocess_text)  # Remove missing reviews and apply preprocessing

'''
# Define sentiment analysis function
def analyze_sentiment(review):
    doc = nlp(review)  # Process the review text
    return doc._.blob.polarity  # Access the sentiment polarity
'''
# ==================================================================
# Function for sentiment analysis
def analyze_sentiment(review):
    try:
        # Process the review using spaCy
        doc = nlp(review)
        # Get sentiment using the sentiment attribute
        sentiment = doc._.blob.polarity
                # Determine sentiment category (positive, negative, or neutral)
        if sentiment > 0:
            return 'Positive'
        elif sentiment < 0:
            return 'Negative'
        else:
            return 'Neutral'

# ==================================================================

# Test sentiment analysis function on sample reviews
sample_reviews = clean_data.head()  # Take the first few cleaned reviews
for review in sample_reviews:
    sentiment = analyze_sentiment(review)
    print(f"Review: {review}\nSentiment Polarity: {sentiment}\n")

# Select two preprocessed reviews by indexing
review_1 = clean_data.iloc[0]  # First cleaned review
review_2 = clean_data.iloc[1]  # Second cleaned review

# Convert reviews to spaCy Doc objects
doc1 = nlp(review_1)
doc2 = nlp(review_2)

# Calculate similarity
similarity_score = doc1.similarity(doc2)

# Output the cleaned reviews and their similarity score
print(f"\nCleaned Review 1: {review_1}")
print(f"\nCleaned Review 2: {review_2}")
print(f"\nSimilarity Score: {similarity_score}")
