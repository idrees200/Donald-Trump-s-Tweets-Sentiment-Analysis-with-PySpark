# Donald-Trump-s-Tweets-Sentiment-Analysis-with-PySpark


This project focuses on sentiment analysis of tweets from the dataset `realdonaldtrump.csv`. Using PySpark, the project involves data cleaning, sentiment classification, and visualizations.

## Table of Contents

1. [Setup](#setup)
2. [Data Cleaning](#data-cleaning)
3. [Sentiment Analysis](#sentiment-analysis)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Evaluation](#evaluation)
6. [Visualizations](#visualizations)
7. [Results](#results)

## Setup

To begin with, ensure that you have PySpark installed in your environment. The necessary libraries for sentiment analysis, data manipulation, and visualization include PySpark, NLTK, Matplotlib, WordCloud, and Plotly.

## Data Cleaning

The data cleaning process involves the following steps:

1. **Reading the Data**: Load the dataset into a PySpark DataFrame.
2. **Text Cleaning**: Define a function to clean the tweet content by removing URLs and punctuation.
3. **Apply Cleaning Function**: Apply the text cleaning function to the tweet content.

## Sentiment Analysis

The sentiment analysis process includes:

1. **Define Sentiment Analysis Function**: Use NLTK's VADER sentiment analysis tool to classify the cleaned text into Positive, Negative, or Neutral.
2. **Apply Sentiment Function**: Apply the sentiment analysis function to the cleaned text to generate sentiment labels.

## Machine Learning Pipeline

The machine learning pipeline consists of the following stages:

1. **StringIndexer**: Convert sentiment labels into numerical indices.
2. **Tokenizer**: Split the cleaned text into words.
3. **StopWordsRemover**: Remove common stop words from the tokenized words.
4. **HashingTF**: Convert the filtered words into numerical feature vectors.
5. **Logistic Regression**: Train a logistic regression model on the feature vectors and sentiment labels.

## Evaluation

The evaluation process involves:

1. **Splitting Data**: Split the data into training and testing sets.
2. **Training the Model**: Fit the machine learning pipeline on the training data.
3. **Making Predictions**: Use the trained model to make predictions on the testing data.
4. **Calculating Metrics**: Calculate accuracy, precision, recall, and F1-score to evaluate the model's performance.

## Visualizations

Several visualizations are created to understand the sentiment distribution and the frequency of tweets over time:

1. **Sentiment Distribution**: A bar chart showing the frequency of each sentiment label.
2. **Sentiment Scores Distribution**: A histogram showing the distribution of sentiment scores.
3. **Word Cloud**: A word cloud visualizing the most frequent terms in the tweets.
4. **Tweet Volume Over Time**: A line chart showing the tweet volume over time.

## Results

After training and evaluating the machine learning model, the following results are obtained:

- **Accuracy**: The overall accuracy of the sentiment classification.
- **Precision**: The weighted precision of the model.
- **Recall**: The weighted recall of the model.
- **F1-score**: The weighted F1-score of the model.

These metrics provide an understanding of how well the model performs in classifying the sentiments of tweets.

---

This project demonstrates the power of PySpark in handling large datasets and performing complex machine learning tasks efficiently. The visualizations provide insights into the sentiment trends and the volume of tweets over time, offering a comprehensive view of the data.
