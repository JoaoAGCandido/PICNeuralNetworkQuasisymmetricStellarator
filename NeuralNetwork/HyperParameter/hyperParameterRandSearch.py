import pandas as pd
from sklearn import neural_network, preprocessing
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import os
import random as rnd
import time
import datetime


parser = argparse.ArgumentParser(
    description="Returns a csv with various losses from neural networks with random hyper parameters\nexample:\npython3 NeuralNetwork/HyperParameter/hyperParameterRandSearch.py 500 -v -nfp=3 -ds=\"scans/scan7/scan7.csv.zip\" -f=\"NeuralNetwork/HyperParameter/randLoss.csv\"")
parser.add_argument(
    "num", help="Number of scans", type=int)
parser.add_argument(
    "-nfp", "--nfp", help="Train neural networks for a specific nfp (1 to 8), default = 0 (all nfp)", type=int, default=0, choices=range(0, 9))
parser.add_argument(
    "-ds", "--dataSet", help="Data set to train Network with, default=\"scans/scan7/scan7Clean.csv.zip\"", type=str, default="scans/scan7/scan7Clean.csv.zip")
parser.add_argument(
    "-es", "--noEarlyStop", help="Disable early_stopping", action="store_false")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Prints verbose including the predicted duration in seconds")
parser.add_argument(
    "-s", "--seed", help="Seed for neural network training, default = None", type=int, default=None)
parser.add_argument(
    "-f", "--fileName", help="Name of the created file, ex \"NeuralNetwork/HyperParameter/randLoss.csv\"", type=str, default="NeuralNetwork/HyperParameter/randLoss.csv")
args = parser.parse_args()


def estimatedTime():
    estimatedTime = (time.time() - startTime) / 10 * args.num
    if args.verbose:
        print("Estimated time: ", str(
            datetime.timedelta(seconds=int(estimatedTime))))


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
        'alpha': [],
        'learning_rate_init': [],
        'beta_1': [],
        'beta_2': [],
        'epsilon': [],
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
    'alpha': [],
    'learning_rate_init': [],
    'beta_1': [],
    'beta_2': [],
    'epsilon': [],
    'loss': [],
    'bestValidationScore': []
}


startTime = time.time()
for i in range(args.num):
    alpha = abs(rnd.gauss(0.0001, 0.0002))
    learning_rate_init = abs(rnd.gauss(0.001, 0.002))
    beta_1 = rnd.uniform(0.7, 0.99)
    beta_2 = rnd.uniform(0.98, 0.999999)
    epsilon = abs(rnd.gauss(1e-8, 2e-08))
    neuralNetwork = neural_network.MLPRegressor(hidden_layer_sizes=(35,35,35,35),
                                                activation='tanh',
                                                solver='adam',
                                                alpha=alpha,
                                                batch_size='auto',
                                                # learning_rate='constant', (sgd)
                                                learning_rate_init=learning_rate_init,
                                                # power_t=0.5, (sgd)
                                                max_iter=1000,
                                                shuffle=True,
                                                random_state=args.seed,
                                                tol=0.0001,
                                                verbose=False,
                                                warm_start=False,
                                                # momentum=0.9, (sgd)
                                                # nesterovs_momentum=True, (sgd)
                                                early_stopping=args.noEarlyStop,
                                                validation_fraction=0.1,
                                                # (adam)
                                                beta_1=beta_1,
                                                # (adam)
                                                beta_2=beta_2,
                                                # (adam)
                                                epsilon=epsilon,
                                                n_iter_no_change=10,
                                                )

    # Train
    neuralNetwork.fit(X_train, y_train)
    out['alpha'].append(alpha)
    out['learning_rate_init'].append(learning_rate_init)
    out['beta_1'].append(beta_1)
    out['beta_2'].append(beta_2)
    out['epsilon'].append(epsilon)
    out['loss'].append(neuralNetwork.loss_)
    out['bestValidationScore'].append(neuralNetwork.best_validation_score_)

    if (i % 15 == 0):
        out = saveData(out)

    if (i == 9):
        estimatedTime()


saveData(out)
endTime = time.time() - startTime
if args.verbose:
    print("\nEnd Time: ", str(datetime.timedelta(seconds=int(endTime))))
