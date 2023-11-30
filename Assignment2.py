import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model,metrics

print("Multi variable linear Regression")
print("read data")
df=pd.read_csv("./Assignment2.csv")
print("Head of data")
print(df.head())
X = df[['area', 'room','window']]
y = df['price']


regr = linear_model.LinearRegression()
regr.fit(X, y)

print("Predicted value of area=20, room=2, window=2")
predicted= regr.predict([[110,5,4]])

print(predicted)
plt.scatter(df.area,df.price,color="red",marker="+")
plt.plot(df.area,regr.predict(df[['area', 'room','window']]),color="blue")
print(plt.show())
print("coeficent " )
print(regr.coef_)
print("\nintercept")
print(regr.intercept_)
#print("score")
#print(regr.score())

