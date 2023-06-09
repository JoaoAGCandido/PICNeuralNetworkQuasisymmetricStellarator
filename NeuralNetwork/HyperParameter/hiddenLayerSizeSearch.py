import pandas as pd
from sklearn import neural_network, preprocessing
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import os
import time
import datetime

parser = argparse.ArgumentParser(
    description="Returns a csv with various losses from neural networks with different hidden layer sizes\nexample:\npython3 NeuralNetwork/HyperParameter/hiddenLayerSizeSearch.py -v -nfp=3 -ds=\"scans/scan7/scan7.csv.zip\" -f=\"hiddenLayerLoss.csv\"")
parser.add_argument(
    "-nfp", "--nfp", help="Train neural networks for a specific nfp (1 to 8), default = 0 (all nfp)", type=int, default=0, choices=range(0, 9))
parser.add_argument(
    "-ds", "--dataSet", help="Data set to train Network with, default=\"scans/scan7/scan7Clean.csv.zip\"", type=str, default="scans/scan7/scan7Clean.csv.zip")
parser.add_argument(
    "-es", "--noEarlyStop", help="Disable early_stopping", action="store_false")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument(
    "-s", "--seed", help="Seed for neural network training, default = None", type=int, default=None)
parser.add_argument(
    "-f", "--fileName", help="Name of the created file, ex \"NeuralNetwork/HyperParameter/hiddenLayerLoss.csv\"", type=str, default="NeuralNetwork/HyperParameter/hiddenLayerLoss.csv")
args = parser.parse_args()


def saveData(out):
    df = pd.DataFrame(out)
    file_exists = os.path.isfile(args.fileName)
    df.sort_values("bestValidationScore", ascending=True)
    if file_exists:
        df.to_csv(args.fileName, index=False, header=False, mode="a")
    else:
        df.to_csv(args.fileName, index=False)
    # clear out
    out = {
        'hiddenLayerSize': [],
        'trainTime' : [],
        'loss': [],
        'bestValidationScore': []
    }
    return out


# Load and partition data
df = pd.read_csv(args.dataSet)
# select nfp
if (args.nfp != 0):
    df = df[df['nfp'] == args.nfp]
    y = df.loc[:, ['rc1', 'zs1', 'eta']]
else:
    y = df.loc[:, ['nfp', 'rc1', 'zs1', 'eta']]

X = df.loc[:, ['RotTrans', 'axLenght', 'max_elong']]


# Split training and testing sets
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.1, train_size=0.9,
                     random_state=0)

#scale
X_scaler = preprocessing.StandardScaler()
X_scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

y_scaler = preprocessing.StandardScaler()
y_scaler.fit(y_train)
y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# arrays to store output
out = {
    'hiddenLayerSize': [],
    'trainTime': [],
    'loss': [],
    'bestValidationScore': [],
}


for layerSize in np.arange(15, 55, 5):
    hiddenLayerSize = [layerSize]
    for numLayers in range(4):
        hiddenLayerSize.append(layerSize)
        # Setup regressors
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
                                                    random_state=args.seed,
                                                    tol=0.0001,
                                                    verbose=args.verbose,
                                                    warm_start=False,
                                                    # momentum=0.9, (sgd)
                                                    # nesterovs_momentum=True, (sgd)
                                                    early_stopping=args.noEarlyStop,
                                                    validation_fraction=0.1,
                                                    beta_1=0.9,  # (adam)
                                                    beta_2=0.999,  # (adam)
                                                    epsilon=1e-08,  # (adam)
                                                    n_iter_no_change=10,
                                                    )

        # Train
        startTime = time.time()
        neuralNetwork.fit(X_train, y_train)
        out['hiddenLayerSize'].append(str(hiddenLayerSize))
        endTime = time.time() - startTime
        out['trainTime'].append(str(datetime.timedelta(seconds=int(endTime))))
        out['loss'].append(neuralNetwork.loss_)
        out['bestValidationScore'].append(neuralNetwork.best_validation_score_)


if args.verbose:
    print(out)
saveData(out)
