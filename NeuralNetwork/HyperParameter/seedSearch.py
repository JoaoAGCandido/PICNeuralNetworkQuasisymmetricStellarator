import pandas as pd
from sklearn import neural_network, preprocessing
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import os
import random as rnd
import time
import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from qsc import Qsc


parser = argparse.ArgumentParser(
    description="Returns a csv with various losses from neural networks with different seeds\nexample:\npython3 NeuralNetwork/HyperParameter/seedSearch.py 30 -v -nfp=3 -ds=\"scans/scan7/scan7.csv.zip\" -f=\"NeuralNetwork/HyperParameter/seedLoss.csv\"")
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
    "-f", "--fileName", help="Name of the created file, ex \"NeuralNetwork/HyperParameter/randLoss.csv\"", type=str, default="NeuralNetwork/nfp3/seedSearchOptimized3.csv")
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
        'seed': [],
        'loss': [],
        'bestValidationScore': [],
	    'testR2': [],
        "testR2Real":[],
        "testR2RealNoOutiler":[],
        "mseRotTrans" : [],
        "mseAxLength" : [],
        "mseMElong" : [],
        "maeRotTrans" : [],
        "maeAxLength" : [],
        "maeMElong" : [],
        "mseMElongNoOutiler" : [],
        "maeMElongNoOutiler" : [],
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


# arrays to store output
out = {
    'seed': [],
    'loss': [],
    'bestValidationScore': [],
    "testR2": [],
    "testR2Real": [],
    "testR2RealNoOutiler": [],
    "mseRotTrans" : [],
    "mseAxLength" : [],
    "mseMElong" : [],
    "maeRotTrans" : [],
    "maeAxLength" : [],
    "maeMElong" : [],
    "mseMElongNoOutiler" : [],
    "maeMElongNoOutiler" : [],
}


startTime = time.time()
for i in range(args.num):
    seed = rnd.randrange(0, 10000)
    # Split training and testing sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.1, train_size=0.9,
                        random_state=seed)

    #scale
    X_scaler = preprocessing.StandardScaler()
    X_scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    y_scaler = preprocessing.StandardScaler()
    y_scaler.fit(y_train)
    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    # Setup regressors
    neuralNetwork = neural_network.MLPRegressor(hidden_layer_sizes=([45,45,45,45]),
                                                activation='tanh',
                                                solver='adam',
                                                alpha=7.910723593282952e-05,
                                                batch_size=87,
                                                # learning_rate='constant', (sgd)
                                                learning_rate_init=0.0009269282149794735,
                                                # power_t=0.5, (sgd)
                                                max_iter=1000,
                                                shuffle=True,
                                                random_state=0,
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
    neuralNetwork.fit(X_train_scaled, y_train_scaled)
    Y_NN = neuralNetwork.predict(X_test_scaled)
    Y_NN_unscaled = y_scaler.inverse_transform(Y_NN)

    out2 = {
        #"askedRotTrans" : [],
        #"askedAxLength" : [],
        #"askedMaxElong" : [],
        "RotTrans" : [],
        "axLenght" : [],
        "max_elong" : [],
    }
    OutlierList = []
    for i in range(len(Y_NN_unscaled)):
        rc1=Y_NN_unscaled[i][0]
        zs1=Y_NN_unscaled[i][1]
        eta=Y_NN_unscaled[i][2]
        stel = Qsc(rc=[1,rc1],zs=[0,zs1],nfp=args.nfp,etabar=eta)
        #out["askedRotTrans"].append(X_test.loc[index,"RotTrans"])
        #out["askedAxLength"].append(X_test.loc[index,"axLenght"])
        #out["askedMaxElong"].append(X_test.loc[index,"max_elong"])
        out2["RotTrans"].append(stel.iota)
        out2["axLenght"].append(stel.axis_length / 2. / np.pi)
        out2["max_elong"].append(stel.max_elongation)
        if stel.max_elongation > 15:
            print(i, " max elongation >15")
            OutlierList.append(i)

    predictedX=pd.DataFrame(out2)

    # Train
    test_predictions = neuralNetwork.predict(X_test_scaled)
    out['seed'].append(seed)
    out['loss'].append(neuralNetwork.loss_)
    out['bestValidationScore'].append(neuralNetwork.best_validation_score_)
    out["testR2"].append(r2_score(y_test_scaled, test_predictions))

    out["testR2Real"].append(r2_score(X_test, predictedX))

    predictedXNoOutlier = predictedX.drop(OutlierList)
    predictedX_scaledNoOutlier = X_scaler.transform(predictedXNoOutlier)
    X_test_scaledNoOutlier = X_test_scaled
    X_test_scaledNoOutlier=np.delete(X_test_scaledNoOutlier,OutlierList,axis=0)
    out["testR2RealNoOutiler"].append(r2_score(X_test_scaledNoOutlier, predictedX_scaledNoOutlier))
    out["mseRotTrans"].append(mean_squared_error(X_test["RotTrans"], predictedX["RotTrans"]))
    out["mseAxLength"].append(mean_squared_error(X_test["axLenght"], predictedX["axLenght"]))
    out["mseMElong"].append(mean_squared_error(X_test["max_elong"], predictedX["max_elong"]))
    out["maeRotTrans"].append(mean_absolute_error(X_test["RotTrans"], predictedX["RotTrans"]))
    out["maeAxLength"].append(mean_absolute_error(X_test["axLenght"], predictedX["axLenght"]))
    out["maeMElong"].append(mean_absolute_error(X_test["max_elong"], predictedX["max_elong"]))
    X_testNoOutlier = X_test.copy()
    X_testNoOutlier.reset_index(inplace=True)
    predictedXNoOutlier = predictedX.drop(OutlierList)
    X_testNoOutlier = X_testNoOutlier.drop(OutlierList)
    out["mseMElongNoOutiler"].append(mean_squared_error(X_testNoOutlier["max_elong"], predictedXNoOutlier["max_elong"]))
    out["maeMElongNoOutiler"].append(mean_absolute_error(X_testNoOutlier["max_elong"], predictedXNoOutlier["max_elong"]))

    if (i % 15 == 0):
        out = saveData(out)

    if (i == 9):
        estimatedTime()


saveData(out)
endTime = time.time() - startTime
if args.verbose:
    print("\nEnd Time: ", str(datetime.timedelta(seconds=int(endTime))))
