import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("Training.csv")

x = df.drop("prognosis",axis=1)
y = df["prognosis"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=40)

model = RandomForestClassifier()
model.fit(x_train,y_train)

joblib.dump(model,"symptom_model.pkl")


test_df = pd.read_csv("Testing.csv")
x_test = test_df.drop("prognosis",axis=1)
y_test = test_df["prognosis"]

accuracy = model.score(x_test,y_test)
print(f"accuray: {accuracy * 100:.2f}%")