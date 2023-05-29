import pandas as pd
from sklearn import preprocessing, neural_network
from sklearn.model_selection import train_test_split
import os
import random as rnd
import time
import datetime
import argparse


parser = argparse.ArgumentParser(
    description="Train neural networks with random hyperparameters returning csv with losses\nexample:\npython3 NeuralNetwork/HyperParameter/hyperParameterRandSearch.py 500 loss.csv --nfp=5")
parser.add_argument(
    "num", help="Number of scans", type=int)
parser.add_argument(
    "fileName", help="Name of the file to be created")
parser.add_argument(
    "--nfp", help="Train neural networks for a specific nfp (2 to 8) instead of all", type=int, default=0, choices=range(2, 9))
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Prints the predicted duration in seconds")
args = parser.parse_args()


def estimatedTime():
    estimatedTime = (time.time() - startTime) / 10 * args.num
    if args.verbose:
        print("Estimated time: ", str(
            datetime.timedelta(seconds=int(estimatedTime))))


def saveData(out):
    df = pd.DataFrame(out)
    file_exists = os.path.isfile(args.fileName)
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
        'loss': []
    }
    return out


# Load and partition data
df = pd.read_csv("scans/scan7/scan7.csv.zip")
# select nfp
if (args.nfp != 0):
    df = df[df['nfp'] == args.nfp]

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
    'alpha': [],
    'learning_rate_init': [],
    'beta_1': [],
    'beta_2': [],
    'epsilon': [],
    'loss': []
}

startTime = time.time()
for i in range(args.num):
    alpha = abs(rnd.gauss(0.0001, 0.0002))
    learning_rate_init = abs(rnd.gauss(0.001, 0.002))
    beta_1=rnd.uniform(0.7, 0.99)
    beta_2=rnd.uniform(0.98, 0.999999)
    epsilon=abs(rnd.gauss(1e-8, 2e-08))
    neuralNetwork = neural_network.MLPRegressor(hidden_layer_sizes=(40,40,40,40),
                                                activation='tanh',
                                                solver='adam',
                                                alpha=alpha,
                                                batch_size='auto',
                                                # learning_rate='constant', (sgd)
                                                learning_rate_init=learning_rate_init,
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

    if (i % 15 == 0):
        out = saveData(out)

    if (i == 9):
        estimatedTime()


saveData(out)
endTime = time.time() - startTime
if args.verbose:
    print("\nEnd Time: ", str(datetime.timedelta(seconds=int(endTime))))
