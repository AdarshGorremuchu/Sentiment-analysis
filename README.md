# Sentiment-analysis

Twitter Sentiment Analysis with Selenium and Python Overview: This Python script performs sentiment analysis on tweets related to a specified subject using Selenium for web scraping, translation with Googletrans, and analysis with TextBlob. The script fetches tweets, translates them to English, cleans and analyzes the sentiment, and visualizes the results.

## Features: Web Scraping

Utilizes Selenium for automated web scraping of tweets from Twitter. Performs login to Twitter using a provided username and password. Translation:

Uses Googletrans to translate tweets to English for uniform analysis. Data Cleaning:

Removes Twitter handles, hashtags, and stopwords for cleaner text. Tokenizes and cleans the tweets for further analysis. Sentiment Analysis:

Applies sentiment analysis using TextBlob to calculate polarity and subjectivity. Segments tweets into positive, negative, and neutral categories based on polarity. Data Visualization:

Generates visualizations, including a Word Cloud and scatter plot, to represent sentiment distribution. Displays count plots for different sentiment categories. Data Export:

Saves the cleaned and analyzed data to a CSV file ('trump_300_twitter_tweets.csv'). Dependencies: Selenium Googletrans Pandas NLTK TextBlob WordCloud Matplotlib Seaborn Usage: Ensure you have the required dependencies installed. Set the 'website' variable to the desired Twitter search page. Update the 'subject' variable with the desired hashtag or keyword. Run the script, and it will scrape, translate, clean, analyze, and visualize the tweets. Additional Notes: The script is designed to scrape a specified number of tweets (default: 1000) and save the results to a CSV file. Feel free to customize the script to suit your specific needs or integrate it into larger projects. Enjoy exploring Twitter sentiments with this Python script!
