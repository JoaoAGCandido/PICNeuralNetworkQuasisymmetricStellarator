import pandas as pd
from sklearn import neural_network, preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import scipy.stats
import argparse
import os
import time
import datetime
import joblib


parser = argparse.ArgumentParser(
    description="Returns a pkl with parameters optimization using sklearn.model_selection.RandomizedSearchCV\nexample:\npython3 NeuralNetwork/HyperParameter/hyperParameterRandSearch.py -num=10 -v -nfp=3 -hp=\"activation\" -ds=\"scans/scan7/scan7Clean.csv.zip\" -f=\"NeuralNetwork/HyperParameter/randSearch.pkl\"")
parser.add_argument(
    "-num", help="Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.", type=int, default=10)
parser.add_argument(
    "-nfp", "--nfp", help="Train neural networks for a specific nfp (1 to 8), default = 0 (all nfp)", type=int, default=0, choices=range(0, 9))
parser.add_argument(
    "-hp", "--hyperParameter", help="hyperparameter to optimize, \"all\" fixes activation to tanh", type=str, choices=["batch_size", "alpha", "learning_rate_init", "activation", "hiddenLayer", "all"], default="all")
parser.add_argument(
    "-ds", "--dataSet", help="Data set to train Network with, default=\"scans/scan7/scan7Clean.csv.zip\"", type=str, default="scans/scan7/scan7Clean.csv.zip")
parser.add_argument(
    "-es", "--noEarlyStop", help="Disable early_stopping", action="store_false")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Prints verbose including the predicted duration in seconds")
parser.add_argument(
    "-f", "--fileName", help="Name of the created file, ex \"NeuralNetwork/HyperParameter/randSearch.pkl\"", type=str, default="NeuralNetwork/HyperParameter/randSearch.pkl")
args = parser.parse_args()


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

# scale
X_scaler = preprocessing.StandardScaler()
X_scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

y_scaler = preprocessing.StandardScaler()
y_scaler.fit(y_train)
y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)


startTime = time.time()
neuralNetwork = neural_network.MLPRegressor(hidden_layer_sizes=(35, 35, 35),
                                            activation='tanh',
                                            solver='adam',
                                            alpha=0.0001,
                                            batch_size=90,  # 'auto',
                                            # learning_rate='constant', (sgd)
                                            learning_rate_init=0.001,
                                            # power_t=0.5, (sgd)
                                            max_iter=1000,
                                            shuffle=True,
                                            random_state=0,
                                            tol=0.0001,
                                            verbose=False,
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

if args.hyperParameter == "batch_size":
    distributions = dict(
        batch_size=range(20, 200),
    )
elif args.hyperParameter == "alpha":
    distributions = dict(
        alpha=scipy.stats.uniform(0.00001, 0.0005),
    )
elif args.hyperParameter == "learning_rate_init":
    distributions = dict(
        learning_rate_init=scipy.stats.uniform(0.0001, 0.005),
    )
elif args.hyperParameter == "activation":
    distributions = dict(
        activation=["identity", "logistic", "tanh", "relu"],
    )
elif args.hyperParameter == "hiddenLayer":
    hiddenLayerList = [
        [15, 15], [15, 15, 15], [15, 15, 15, 15], [15, 15, 15, 15, 15],
        [20, 20], [20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20, 20],
        [25, 25], [25, 25, 25], [25, 25, 25, 25], [25, 25, 25, 25, 25],
        [30, 30], [30, 30, 30], [30, 30, 30, 30], [30, 30, 30, 30, 30],
        [35, 35], [35, 35, 35], [35, 35, 35, 35], [35, 35, 35, 35, 35],
        [40, 40], [40, 40, 40], [40, 40, 40, 40], [40, 40, 40, 40, 40],
        [45, 45], [45, 45, 45], [45, 45, 45, 45], [45, 45, 45, 45, 45],
        [50, 50], [50, 50, 50], [50, 50, 50, 50], [50, 50, 50, 50, 50],
    ]
    distributions = dict(
        hidden_layer_sizes=hiddenLayerList,
    )
elif args.hyperParameter == "all":
    hiddenLayerList = [
        [30, 30, 30], [30, 30, 30, 30],
        [31, 31, 31], [31, 31, 31, 31],
        [32, 32, 32], [32, 32, 32, 32],
        [33, 33, 33], [33, 33, 33, 33],
        [34, 34, 34], [34, 34, 34, 34],
        [35, 35, 35], [35, 35, 35, 35],
        [36, 36, 36], [36, 36, 36, 36],
        [37, 37, 37], [37, 37, 37, 37],
        [38, 38, 38], [38, 38, 38, 38],
        [39, 39, 39], [39, 39, 39, 39],
        [40, 40, 40], [40, 40, 40, 40],
        [41, 41, 41], [41, 41, 41, 41],
        [42, 42, 42], [42, 42, 42, 42],
        [43, 43, 43], [43, 43, 43, 43],
        [44, 44, 44], [44, 44, 44, 44],
        [45, 45, 45], [45, 45, 45, 45],
    ]
    alphaLoc=0.0001
    alphaScale=0.00005
    a, b = (0 - alphaLoc) / alphaScale, (0.0505 - alphaLoc) / alphaScale
    distributions = dict(
        batch_size=range(50, 150),
        alpha=scipy.stats.truncnorm(loc=alphaLoc, scale=alphaScale, a=a, b=b),
        learning_rate_init=scipy.stats.norm(0.001, 0.0002),
        hidden_layer_sizes=hiddenLayerList,
    )

clf = RandomizedSearchCV(neuralNetwork, distributions, random_state=0,
                         verbose=args.verbose, n_jobs=-1, n_iter=args.num)
search = clf.fit(X_train_scaled, y_train_scaled)
joblib.dump(search, args.fileName)

endTime = time.time() - startTime
if args.verbose:
    print("\nsearch:\n", search)
    print("\nbest_params: ", search.best_params_)
    print("\ncv_results:\n", search.cv_results_)
    print("\nEnd Time: ", str(datetime.timedelta(seconds=int(endTime))))
