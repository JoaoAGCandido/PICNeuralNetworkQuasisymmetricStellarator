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
        'hiddenLayerSize': [],
        'loss': [],
    }
    return out


# Load and partition data
df = pd.read_csv("scans/scan4/scan4.csv.zip")
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
    out['alpha'].append(rnd.uniform(0.000001, 0.1))
    out['learning_rate_init'].append(rnd.uniform(0.000001, 0.1))
    out['beta_1'].append(rnd.uniform(0.1, 0.99))
    out['beta_2'].append(rnd.uniform(0.8, 0.999999))
    out['epsilon'].append(rnd.uniform(1e-10, 1e-07))
    neuralNetwork = neural_network.MLPRegressor(hidden_layer_sizes=(35, 35, 35, 35),
                                                activation='tanh',
                                                solver='adam',
                                                alpha=out['alpha'][i],
                                                batch_size='auto',
                                                # learning_rate='constant', (sgd)
                                                learning_rate_init=out['learning_rate_init'][i],
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
                                                beta_1=out['beta_1'][i],
                                                # (adam)
                                                beta_2=out['beta_2'][i],
                                                # (adam)
                                                epsilon=out['epsilon'][i],
                                                n_iter_no_change=10,
                                                )

    # Train
    neuralNetwork.fit(X_train, y_train)
    out['loss'].append(neuralNetwork.loss_)

    if (i % 15 == 0):
        out = saveData(out)

    if (i == 9):
        estimatedTime()


saveData(out)
endTime = time.time() - startTime
if args.verbose:
    print("\nEnd Time: ", str(datetime.timedelta(seconds=int(endTime))))
