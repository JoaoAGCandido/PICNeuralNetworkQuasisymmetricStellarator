import pandas as pd
from sklearn import neural_network, preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import scipy.stats
import argparse
import numpy as np
import time
import datetime
import joblib
from sklearn.linear_model import Ridge, LinearRegression
from sklearn import pipeline


parser = argparse.ArgumentParser(
    description="Returns a pkl with hyperparameters optimization using sklearn.model_selection.RandomizedSearchCV\nexample:\npython3 NeuralNetwork/HyperParameter/hyperParameterRandSearch.py -num=10 -v -nfp=3 -hp=\"activation\" -ds=\"scans/scan7/scan7Clean.csv.zip\" -f=\"OtherModels/randSearch.pkl\"")
parser.add_argument(
    "-num", help="Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.", type=int, default=10)
parser.add_argument(
    "-nfp", "--nfp", help="Train Ridge for a specific nfp (1 to 8), default = 0 (all nfp)", type=int, default=0, choices=range(0, 9))
parser.add_argument(
    "-ds", "--dataSet", help="Data set to train Network with, default=\"scans/scan7/scan7Clean.csv.zip\"", type=str, default="scans/scan7/scan7Clean.csv.zip")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Prints verbose including the predicted duration in seconds")
parser.add_argument(
    "-f", "--fileName", help="Name of the created file, ex \"OtherModels/randSearch.pkl\"", type=str, default="OtherModels/randSearch.pkl")
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


poly = preprocessing.PolynomialFeatures()
reg= LinearRegression()
model = pipeline.Pipeline([("poly", poly), ("reg", reg)])


distributions = {'poly__degree': range(2,20)}

clf = RandomizedSearchCV(model, distributions, random_state=0,
                         verbose=args.verbose, n_jobs=-1, n_iter=args.num)
search = clf.fit(X_train_scaled, y_train_scaled)
joblib.dump(search, args.fileName)

endTime = time.time() - startTime
if args.verbose:
    print("\nsearch:\n", search)
    print("\nbest_params: ", search.best_params_)
    print("\ncv_results:\n", search.cv_results_)
    print("\nbest_params: ", search.best_params_)
    print("\nEnd Time: ", str(datetime.timedelta(seconds=int(endTime))))
