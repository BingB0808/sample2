# Importing necessary libraries
import numpy as np
import pandas as pd 
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
import pickle

# Reading the data
# Preparing the NumPy array of the inputs.
iris = pd.read_csv("C:/Users/avllnvmnnsl/Downloads/datasets/iris.csv")

iris["variety"] = iris["variety"].map({"Setosa":0,"Virginica":1,"Versicolor":2})

data_inputs = iris[['sepal.length','sepal.width','petal.length','petal.width']]
X = data_inputs.to_numpy()

y = iris['variety']



# Training the model
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.3)
model = LogisticRegression()
model.fit(x_train,y_train)

pickle.dump(model,open('irisReg-model.pkl','wb'))

model = pickle.load(open('irisReg-model.pkl', 'rb'))
prediction = model.predict([[5.9,2,3,1]])

print(prediction )