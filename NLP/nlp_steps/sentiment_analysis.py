import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

data = {
  "text": [
    "I love this product! It's amazing!",
    "I hate waiting in long lines. It's so frustrating.",
    "The movie was okay, not great but not terrible.",
    "I am so happy with my new phone!",
    "I feel really sad today, everything is going wrong.",
    "I am super excited about the upcoming vacation!",
    "The service at the restaurant was terrible. Very slow.",
    "I just had the best coffee ever. Highly recommend it!",
    "The weather today is gloomy and rainy.",
    "I had a really good time at the party last night!"
  ]
}

df = pd.DataFrame(data)

def textblob_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def vader_sentiment(text):
    vader = SentimentIntensityAnalyzer()
    score = vader.polarity_scores(text)
    return score['compound']

df['textblob_sentiment'] = df['text'].apply(textblob_sentiment)
df['vader_sentiment'] = df['text'].apply(vader_sentiment)

print(df)

# Transformers // Add transformers to the same sample

sentiment_pipeline = pipeline("sentiment-analysis")

def transformer_sentiment(text):
    result = sentiment_pipeline(text)[0]
    # Convert 'LABEL_0' and 'LABEL_1' to 'negative' and 'positive'
    if result['label'].lower() in ['positive', 'label_1']:
        return result['score']
    else:
        return -result['score']

# Apply all sentiment functions
df['textblob_sentiment'] = df['text'].apply(textblob_sentiment)
df['vader_sentiment'] = df['text'].apply(vader_sentiment)
df['transformer_sentiment'] = df['text'].apply(transformer_sentiment)

print(df)

