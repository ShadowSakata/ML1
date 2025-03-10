import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("C:\\Users\\shift\\python-projects\\project1\\venv\\datasets\\titanic.csv")
print(df.head())
df = df.drop(df.columns[[5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,22,23,25,26]], axis=1) #удаление мёртвых столбцов
print(df.head())
df.fillna({"Age" : df["Age"].median()}, inplace=True)
df.fillna({"Embarked" : df["Embarked"].mode()[0]}, inplace=True)
scaler = MinMaxScaler()

df["Age"] = scaler.fit_transform(df[["Age"]])
print(df.head())

df = pd.get_dummies(df,columns=["Embarked"], drop_first=True)
print(df.head())
df.to_csv("../datasets/processed_titanic.csv", index=False)