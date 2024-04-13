import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import joblib

df=pd.read_csv("dataset.csv")
# print("Data Sample")
# print(df.head())
# print("Unique responses : ", len(df['response'].unique()))

dataset = df.groupby('utterance').apply(lambda x: x.reindex(x.index.repeat(2))).reset_index(drop=True)

X = dataset['utterance']
Y = dataset['response']

#transform x
vectorizer = TfidfVectorizer()
x_tfid = vectorizer.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(x_tfid, Y, test_size=0.2, random_state=42)

model = SGDClassifier(loss='log_loss')
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test,Y_pred)
print("Accuracy : ", accuracy*100)

joblib.dump(model, "model.pkl")