from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

df = pd.read_csv('NLP\project\IMDBDataset.csv')

X = df['review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

def analyze_sentiment(review:str)->str:
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)
    return prediction[0]

class Review(BaseModel):
    review: str



# Predict of the test set
y_pred = model.predict(X_test_tfidf)
# Calculate accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy}\n Classification Report:\n{class_report}')

# Try with SVM as well
svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)

# Prediction and metrics for SVM
y_pred_svm = svm_model.predict(X_test_tfidf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
class_report_svm = classification_report(y_test, y_pred_svm)
print(f'SVM Accuracy: {accuracy_svm}\n Classification Report:\n{class_report_svm}')


# Analyze sentiment using TextBlob

def analyze_textblob(review: str) -> str:
    polarity = TextBlob(review).sentiment.polarity
    return "positive" if polarity > 0 else "negative" 

sia = SentimentIntensityAnalyzer()

def analyze_vader(review: str) -> str:
    score = sia.polarity_scores(review)['compound']
    return "positive" if score > 0 else "negative"


@app.post("/analyze/")

def analyze_review(review: Review):
    logistic_sentiment = analyze_sentiment(review.review)
    svm_sentiment = svm_model.predict(vectorizer.transform([review.review]))[0]
    textblob_sentiment = analyze_textblob(review.review)
    vader_sentiment = analyze_vader(review.review)

    return {
        "review": review.review,
        "logistic_regression": logistic_sentiment,
        "svm": svm_sentiment,
        "textblob": textblob_sentiment,
        "vader": vader_sentiment
    }


# Run the FastAPI server with the command:
## uvicorn NLP.project.project:app --reload 
# Take the ip adress and add /docs to the end After that run the server in browser to see the API documentation and test the endpoint.
