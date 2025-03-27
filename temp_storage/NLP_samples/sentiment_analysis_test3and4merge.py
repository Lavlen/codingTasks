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

# Function for sentiment analysis
def analyze_sentiment(review):
    try:
        # Process the review using spaCy
        doc = nlp(review)
        # Get sentiment using the sentiment attribute
        sentiment = doc._.polarity
        # Determine sentiment category (positive, negative, or neutral)
        if sentiment > 0:
            return 'Positive'
        elif sentiment < 0:
            return 'Negative'
        else:
            return 'Neutral'
    except Exception as e:
        print(f"Error analyzing sentiment for review: {review}\nError: {e}")
        return 'Neutral'

# Test the model for Sample Model Reviews
def test_sentiment_analysis(review):
    sentiment_result = analyze_sentiment(review)
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment_result}")
    print("=" * 30)

# Ensure valid indices for testing reviews
if len(data) > 1:
    # Retrieve the reviews using indexing
    try:
        review_1 = data['reviews.text'].iloc[0]  # Safe indexing with .iloc
        review_2 = data['reviews.text'].iloc[1]
    except IndexError as e:
        print(f"Error accessing reviews for testing: {e}")
        exit()

    # Test the sentiment analysis function on the selected reviews
    test_sentiment_analysis(review_1)
    test_sentiment_analysis(review_2)

    # Compare the similarity of the two reviews using spaCy
    if review_1 and review_2:
        try:
            similarity_score = nlp(review_1).similarity(nlp(review_2))
            print(f"Similarity Score: {similarity_score}")
        except Exception as e:
            print(f"Error calculating similarity: {e}")
    else:
        print("One or both reviews are empty. Cannot calculate similarity.")
else:
    print("Not enough reviews in the dataset for testing.")

