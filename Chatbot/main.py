import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import joblib

def get_response(utterance):
    data = pd.read_csv('dataset.csv')
    X = data['utterance']
    Y = data['response']
    vectorizer = TfidfVectorizer()
    x_tfid = vectorizer.fit_transform(X)
    model = SGDClassifier(loss='log_loss')
    loaded_model = joblib.load('model.pkl')
    new_utterance = [utterance]
    new_utterance_tfidf = vectorizer.transform(new_utterance)
    predicted_probability = loaded_model.predict_proba(new_utterance_tfidf)
    predictions = loaded_model.predict(new_utterance_tfidf)

    max_confidence = -1
    predicted_class = None

    for i in range(len(predictions)):
        for j, class_prob in enumerate(predicted_probability[i]):
            class_name = loaded_model.classes_[j]
            confidence = class_prob*100
            if confidence>max_confidence:
                max_confidence=confidence
                predicted_class=class_name
    
    return {"response":predicted_class, "confidence":max_confidence}
