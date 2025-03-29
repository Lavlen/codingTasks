import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Load spaCy model
import en_core_web_md
nlp = en_core_web_md.load()

# Add the SpacyTextBlob pipeline
try:
    nlp.add_pipe('spacytextblob', last=True)
    print("Pipeline components:", nlp.pipe_names)  # Verify that 'spacytextblob' was added
except ValueError as e:
    print(f"Error adding spacytextblob to pipeline: {e}")
    exit()

# Load dataset
try:
    data = pd.read_csv("amazon_product_reviews.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'amazon_product_reviews.csv' file not found!")
    exit()

# Function to preprocess text
def preprocess_text(review):
    # Convert to lowercase, remove spaces and check that review's a string
    doc = nlp(str(review).lower().strip()) 
    # Remove stop words
    processed_tokens = [token.text for token in doc if not token.is_stop] 
    # Return cleaned text 
    return " ".join(processed_tokens) 

# Preprocess all reviews
if 'reviews.text' in data.columns:
    reviews_data = data['reviews.text']
    clean_data = reviews_data.dropna().apply(preprocess_text)  # Remove missing reviews and apply preprocessing
else:
    print("Error: 'reviews.text' column not found in the dataset!")
    exit()

# Function for sentiment analysis
def analyze_sentiment(review):
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

# Perform test sentiment analysis on sample reviews
sample_reviews = clean_data.head()  # Take the first few cleaned reviews
for review in sample_reviews:
    sentiment = analyze_sentiment(review)
    if sentiment is not None:
        print(f"Review: {review}\nSentiment Polarity: {sentiment}\n")
    else:
        print(f"Sentiment analysis failed for review: {review}\n")

# Select two preprocessed reviews by indexing
if len(clean_data) > 1:  # Ensure there are at least two reviews
    review_1 = clean_data.iloc[0]  # First cleaned review
    review_2 = clean_data.iloc[1]  # Second cleaned review

    # Convert reviews to spaCy Doc objects
    doc1 = nlp(review_1)
    doc2 = nlp(review_2)

    # Calculate similarity
    similarity_score = doc1.similarity(doc2)

    # Output the cleaned reviews and similarity score
    print(f"\nCleaned Review 1: {review_1}")
    print(f"\nCleaned Review 2: {review_2}")
    print(f"\nSimilarity Score: {similarity_score}")
else:
    print("Not enough reviews available for comparison.")
