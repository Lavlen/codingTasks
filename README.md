# codingTasks
# Coding Task Name: Sentiment Analysis and Text Similarity with spaCy

## Description
This project demonstrates sentiment analysis and text similarity calculations using the spaCy NLP library and SpacyTextBlob pipeline. The script processes Amazon product reviews, performs sentiment analysis, and calculates similarity scores between reviews. Learning these skills is crucial for understanding how to work with Natural Language Processing (NLP) tools to extract insights from text data.

---

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)

---

## Installation
1. Install Python:
- Install Python 3.8-10 on your system. Download it from python.org.

2. Set Up a Virtual Environment (Optional but Recommended):
   Create a virtual environment to manage dependencies by typing the following code in your terminal
   - python -m venv myenv
   - source myenv/bin/activate (Unix-based platforms)
   - myenv\Scripts\activate (Windows platforms)

3. Install Required Libraries:
   Install the required Python packages using pip:
   - pip install pandas spacy spacytextblob

4. Download spaCy's Model:
   Download the spaCy model en_core_web_md required for text processing:
   - python -m spacy download en_core_web_md

5. Add Dataset File:
- Place the dataset file amazon_product_reviews.csv in the project directory.
- Ensure the file contains a reviews.text column to avoid errors.

6. Verify the Pipeline Configuration:
- The SpacyTextBlob pipeline should be added as described in the code. 

---

## Usage
The project enables sentiment analysis and similarity scoring for product reviews using pandas, spaCy, and SpacyTextBlob. Steps to run the code are as follows:

1. Prepare Your Dataset:
- Place the dataset amazon_product_reviews.csv in the root directory of the project.
- Ensure that the dataset contains a reviews.text column with customer reviews.

2. Import Python Libraries as per the code:
- import pandas as pd
- import spacy
- from spacytextblob.spacytextblob import SpacyTextBlob

3. Load spaCy Model:
- import en_core_web_md
- nlp = en_core_web_md.load()

4. Add the Sentiment Analysis Pipeline:
   The SpacyTextBlob pipeline is added to the spaCy model for sentiment analysis:
   - nlp.add_pipe('spacytextblob', last=True)

5. Run the code:
The code processes product reviews to:
a. Preprocess text (convert to lowercase, remove stop words, etc.).
b. Perform sentiment analysis to classify reviews as Positive, Negative, or Neutral.
c. Calculate similarity between two preprocessed reviews.

To execute the code, type the following in your terminal:
- python sentimental_analysis.py

Outputs:
The script performs the following tasks:
- Verifies the spacytextblob pipeline is added.
- Loads and preprocesses the dataset.
- Analyzes the sentiment polarity of sample reviews.
- Computes a similarity score between two preprocessed reviews.
- ![Alt text](images/Reviewallresults2.jpg "Project Flowchart")
---

## Credit
