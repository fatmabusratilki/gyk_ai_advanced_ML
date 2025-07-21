# Cleaning the dataset

import re

text = "Hello!, I am learning NLP. NLP is great area :)"

text = text.lower()  # Convert to lowercase

text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
print(text)

# Tokenization

from nltk.tokenize import word_tokenize

tokens = word_tokenize(text)  # Tokenize the text

print(tokens)

# Stopword removal
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))

filtered_tokens = [word for word in tokens if word not in stopwords]  # Remove stopwords
print(filtered_tokens)

# Steamming/ Lemmatization

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

words = ["running", "flies", "better", "easily"]

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("Stemmed words: ", [stemmer.stem(word) for word in words])
print("Lemmatized words: ", [lemmatizer.lemmatize(word) for word in words])


# Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

data = ["Natural Language is fun.", 
        "Leaning NLP opens many doors.", 
        "Studyin NLp makes you feel happy."]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)
print("Feature names:", vectorizer.get_feature_names_out())
print("TF-IDF matrix:\n", X.toarray())