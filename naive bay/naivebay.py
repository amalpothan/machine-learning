import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

dataset = pd.read_csv(r"iris.csv")

x = dataset.drop('species', axis=1)
y = dataset['species']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=10)

classifier = GaussianNB()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)

print("Accuracy score :", accuracy)


new_observation = np.array([[5.1,3.5,1.4,0.2]])
predicted_class = classifier.predict(new_observation)
print("predicted class : ", predicted_class)