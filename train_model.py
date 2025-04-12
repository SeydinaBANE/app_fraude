import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


df= pd.read_csv("creditcard.csv")
X= df.drop(columns= "Class")
y =df["Class"]
X_train, X_test,y_train, y_test =train_test_split(X, y,
                                    random_state=42, test_size=0.2              )
model =RandomForestClassifier(n_estimators=100,
                              random_state=42)
pipeline = Pipeline([
    ("model",model)
])

score_cv =cross_val_score(pipeline, X_train, y_train,cv=3)
mean_accuracy =score_cv.mean()
mean_accuracy

pipeline.fit(X_train,y_train)
y_pred =pipeline.predict(X_test)
acccuracy =accuracy_score(y_test, y_pred)
acccuracy
f_1 =f1_score(y_test, y_pred)
f_1



recall =recall_score(y_test, y_pred)
recall

#Sauvegarder le model
pickle.dump(pipeline, open("fraud_model.pkl", "wb"))
X.shape
df.shape
