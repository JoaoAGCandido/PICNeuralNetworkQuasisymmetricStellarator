import pandas as pd
import numpy as np
from scipy.io.arff import loadarff
from sklearn import metrics, neural_network
from sklearn.model_selection import train_test_split
import joblib


#load neural network
neuralNetwork2 = joblib.load("NeuralNetwork/neuralNetwork.pkl")

testData = {
        'RotTrans': [4, 3],
        'axLenght': [1.1, 1.5],
        'max_elong': [6, 5]
}
df = pd.DataFrame(testData)

predicted = neuralNetwork2.predict(df)

print("for the following data:\n", df)

print("The Result was:\n      nfp          rc1          zs1         eta\n", predicted)
