import pandas as pd
import numpy as np
from sklearn import preprocessing, neural_network
from sklearn.model_selection import train_test_split
import os


def saveData(out):
    df = pd.DataFrame(out)
    file_exists = os.path.isfile("NeuralNetwork/HyperParameter/loss.csv")
    if file_exists:
        df.to_csv("NeuralNetwork/HyperParameter/loss.csv", index=False, header=False, mode="a")
    else:
        df.to_csv("NeuralNetwork/HyperParameter/loss.csv", index=False)
    # clear out
    out = {
        'hiddenLayerSize': [],
        'loss': [],
    }
    return out


# Load and partition data
df = pd.read_csv("scans/scan4/scan4.csv.zip")
X = df.loc[:, ['RotTrans', 'axLenght', 'max_elong']]
X_scaler = preprocessing.StandardScaler()
X_scaler.fit(X)
X_scaled = X_scaler.transform(X)
y = df.loc[:, ['nfp', 'rc1', 'zs1', 'eta']]
y_scaler = preprocessing.StandardScaler()
y_scaler.fit(y)
y_scaled = y_scaler.transform(y)

# Split training and testing sets
X_train, X_test, y_train, y_test = \
    train_test_split(X_scaled, y_scaled, test_size=0.3, train_size=0.7,
                     random_state=0)

# arrays to store output
out = {
    'hiddenLayerSize': [],
    'loss': [],
}


for layerSize in np.arange(5, 40, 5):
    hiddenLayerSize = [layerSize]
    for numLayers in range(4):
        hiddenLayerSize.append(layerSize)

        neuralNetwork = neural_network.MLPRegressor(hidden_layer_sizes=(hiddenLayerSize),
                                                    activation='tanh',
                                                    solver='adam',
                                                    alpha=0.0001,
                                                    batch_size='auto',
                                                    # learning_rate='constant', (sgd)
                                                    learning_rate_init=0.001,
                                                    # power_t=0.5, (sgd)
                                                    max_iter=1000,
                                                    shuffle=True,
                                                    random_state=None,
                                                    tol=0.0001,
                                                    verbose=False,
                                                    warm_start=False,
                                                    # momentum=0.9, (sgd)
                                                    # nesterovs_momentum=True, (sgd)
                                                    early_stopping=False,
                                                    validation_fraction=0.1,
                                                    beta_1=0.9,  # (adam)
                                                    beta_2=0.999,  # (adam)
                                                    epsilon=1e-08,  # (adam)
                                                    n_iter_no_change=10,
                                                    )

        # Train
        neuralNetwork.fit(X_train, y_train)

        out['hiddenLayerSize'].append(str(hiddenLayerSize))
        out['loss'].append(neuralNetwork.loss_)

saveData(out)
