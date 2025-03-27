import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Load spaCy model
import en_core_web_md
nlp = en_core_web_md.load()
print("Model loaded successfully!")

# Add the SpacyTextBlob pipeline
try:
    nlp.add_pipe('spacytextblob', last=True)
    print("Pipeline components:", nlp.pipe_names)  # Verify components
except ValueError as e:
    print(f"Error adding spacytextblob to pipeline: {e}")

# Load dataset
try:
    data = pd.read_csv("amazon_product_reviews.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'amazon_product_reviews.csv' file not found!")
    exit()

# Drop rows with missing values in 'reviews.text' column
if 'reviews.text' in data.columns:
    data.dropna(subset=['reviews.text'], inplace=True)
else:
    print("Error: 'reviews.text' column not found in dataset!")
    exit()

# Function to preprocess text
def preprocess_text(text):
    # Use spaCy to tokenize and remove stopwords
    doc = nlp(text)
    tokens = [token.text.lower().strip() for token in doc if not token.is_stop]
    return ' '.join(tokens)

# Apply preprocessing to 'reviews.text' column
try:
    data['reviews.text'] = data['reviews.text'].apply(preprocess_text)
except Exception as e:
    print(f"Error during preprocessing: {e}")
    exit()

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

