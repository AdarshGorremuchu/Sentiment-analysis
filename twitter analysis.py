import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
import pandas as pd
from deep_translator import GoogleTranslator
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize variables
website = 'https://twitter.com/login'
subject = "#KCR"
all_tweets = set()
translator = Translator()

# Set up the web driver and log in to Twitter
driver = webdriver.Chrome()
driver.get(website)

# Login
sleep(3)
wait = WebDriverWait(driver, 20)
username = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@name='text']")))
username.send_keys("pranavdola1054@gmail.com")
next_button = driver.find_element(By.XPATH, "//span[contains(text(),'Next')]")
next_button.click()

sleep(3)
wait = WebDriverWait(driver, 20)
username = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@name='text']")))
username.send_keys("+917674884904")
next_button = driver.find_element(By.XPATH, "//span[contains(text(),'Next')]")
next_button.click()

sleep(3)
password = driver.find_element(By.XPATH, "//input[@name='password']")
password.send_keys('Pranav@123')
log_in = driver.find_element(By.XPATH, "//span[contains(text(),'Log in')]")
log_in.click()

# Search for the subject
sleep(3)
search_box = driver.find_element(By.XPATH, "//input[@data-testid='SearchBox_Search_Input']")
search_box.send_keys(subject)
search_box.send_keys(Keys.ENTER)

# Fetch tweets and translate them
scroll_attempts = 0
max_scroll_attempts = 10
previous_tweet_count = 0

while len(all_tweets) < 600 and scroll_attempts < max_scroll_attempts:
    tweets = driver.find_elements(By.XPATH, "//div[@data-testid='tweetText']")
    new_tweets_collected = 0

    for tweet in tweets:
        try:
            tweet_text = tweet.text
            # Translate tweet text to English
            translation = translator.translate(tweet_text, dest='en')
            if translation and translation.text:  # Ensure translation is not None
                if translation.text not in all_tweets:
                    all_tweets.add(translation.text)
                    new_tweets_collected += 1
                    print(translation.text)
                else:
                    print("Duplicate tweet found, skipping:", tweet_text)
            else:
                print("Translation failed for tweet:", tweet_text)
        except selenium.common.exceptions.StaleElementReferenceException:
            # Handle the stale element exception by re-fetching the tweets
            tweets = driver.find_elements(By.XPATH, "//div[@data-testid='tweetText']")
            continue
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

    driver.execute_script('window.scrollTo(0,document.body.scrollHeight);')
    sleep(5)  # Increase sleep time to ensure more tweets are loaded

    # Check if new tweets were collected in this scroll
    if len(all_tweets) == previous_tweet_count:
        scroll_attempts += 1
    else:
        scroll_attempts = 0

    previous_tweet_count = len(all_tweets)

# Convert the set of translated tweets to a list and then create a DataFrame
all_tweets = list(all_tweets)
df = pd.DataFrame({'tweets': all_tweets})
df.to_csv('trump_300_twitter_tweets.csv', index=False)

# Define stopwords and functions for cleaning tweets, calculating polarity and subjectivity, and segmentation
stp_words = stopwords.words('english')

def TweetCleaning(tweet):
    cleanTweet = re.sub(r"@[a-zA-Z0-9]+","",tweet)
    cleanTweet = re.sub(r"#[a-zA-Z0-9\s]+","",cleanTweet)
    cleanTweet = ' '.join(word for word in cleanTweet.split() if word not in stp_words)
    return cleanTweet

def calPolarity(tweet):
    return TextBlob(tweet).sentiment.polarity

def calSubjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity

def segmentation(tweet):
    if tweet > 0:
        return "positive"
    if tweet == 0:
        return "neutral"
    else:
        return "negative"

# Clean tweets and calculate sentiment
df['cleanedTweets'] = df['tweets'].apply(TweetCleaning)
df['tPolarity'] = df['cleanedTweets'].apply(calPolarity)
df['tSubjectivity'] = df['cleanedTweets'].apply(calSubjectivity)
df['segmentation'] = df['tPolarity'].apply(segmentation)

# Analysis and Visualization
print("Pivot table by segmentation:")
print(df.pivot_table(index=['segmentation'], aggfunc={'segmentation': 'count'}))

# Top 3 most positive, negative, and neutral tweets
print("Top 3 most positive tweets:")
print(df.sort_values(by=['tPolarity'], ascending=False).head(3))
print("Top 3 most negative tweets:")
print(df.sort_values(by=['tPolarity'], ascending=True).head(3))
print("Neutral tweets:")
print(df[df.tPolarity == 0])

# Generate and display the word cloud
consolidated = ' '.join(word for word in df['cleanedTweets'])
wordCloud = WordCloud(width=400, height=200, random_state=20, max_font_size=119, font_path=r"C:\\Windows\\Fonts\\arial.ttf").generate(consolidated)

plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Plot sentiment distribution
plt.figure(figsize=(10,5))
sns.set_style("whitegrid")
sns.scatterplot(data=df, x='tPolarity', y='tSubjectivity', s=100, hue='segmentation')
sns.countplot(data=df, x='segmentation')
plt.show()

# Calculate and display response percentages
positive = round(len(df[df.segmentation == 'positive'])/len(df)*100,1)
negative = round(len(df[df.segmentation == 'negative'])/len(df)*100,1)
neutral = round(len(df[df.segmentation == 'neutral'])/len(df)*100,1)

responses = [positive, negative, neutral]

response = {'resp': ['mayWin', 'mayLoose', 'notSure'], 'pct': [positive, negative, neutral]}
print(pd.DataFrame(response))

print(f"Positive: {positive}%, Negative: {negative}%, Neutral: {neutral}%")
