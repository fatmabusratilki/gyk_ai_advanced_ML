# summary_api.py

import nltk
import numpy as np
import re
import heapq
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer
import uvicorn

nltk.download('punkt')
nltk.download('stopwords')

app = FastAPI()

class SummaryRequest(BaseModel):
    text: str
    sentence_count: Optional[int] = 3
    language: Optional[str] = "english"
    reference_summary: Optional[str] = None  # for ROUGE evaluation

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', '', text)
    return text

def generate_summary(text, sentence_count=3, language="english"):
    cleaned = clean_text(text)
    sentences = sent_tokenize(cleaned, language=language)
    stop_words = stopwords.words(language)
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(sentences)
    scores = {sentences[i]: X[i].toarray().sum() for i in range(len(sentences))}
    top_sentences = heapq.nlargest(sentence_count, scores, key=scores.get)
    return ' '.join(top_sentences)

def evaluate_rouge(summary, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return {metric: round(score.fmeasure, 3) for metric, score in scores.items()}

@app.post("/summarize")
def summarize(request: SummaryRequest):
    summary = generate_summary(request.text, request.sentence_count, request.language)
    response = {"summary": summary}

    if request.reference_summary:
        rouge_scores = evaluate_rouge(summary, request.reference_summary)
        response["rouge_evaluation"] = rouge_scores

    return response


# To run locally, use:
# uvicorn summary_api:app --reload