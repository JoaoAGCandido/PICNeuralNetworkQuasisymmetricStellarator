from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from sklearn import neural_network, preprocessing
from sklearn.model_selection import train_test_split
import joblib
import argparse


parser = argparse.ArgumentParser(
    description="Train a neural network,\nexample:\npython3 NeuralNetwork/createNeuralNetwork.py -v -nfp=3 -ds=\"scans/scan7/scan7.csv.zip\" -f=\"NeuralNetwork/neuralNetwork\"")
parser.add_argument(
    "-nfp", "--nfp", help="Train neural networks for a specific nfp (1 to 8), default = 0 (all nfp)", type=int, default=0, choices=range(0, 9))
parser.add_argument(
    "-ds", "--dataSet", help="Data set to train Network with, default=\"scans/scan7/scan7Clean.csv.zip\"", type=str, default="scans/scan7/scan7Clean.csv.zip")
parser.add_argument(
    "-es", "--noEarlyStop", help="Disable early_stopping", action="store_false")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument(
    "-f", "--fileName", help="Name of the created files, ex \"NeuralNetwork/neuralNetwork\"", type=str, default="NeuralNetwork/neuralNetwork")
args = parser.parse_args()


# Load and partition data
df = pd.read_csv(args.dataSet)
# select nfp
if (args.nfp != 0):
    df = df[df['nfp'] == args.nfp]

X = df.loc[:, ['RotTrans', 'axLenght', 'max_elong']]
X_scaler = preprocessing.StandardScaler()
X_scaler.fit(X)
X_scaled = X_scaler.transform(X)

if (args.nfp == 0):
    y = df.loc[:, ['nfp', 'rc1', 'zs1', 'eta']]
else:
    y = df.loc[:, ['rc1', 'zs1', 'eta']]
y_scaler = preprocessing.StandardScaler()
y_scaler.fit(y)
y_scaled = y_scaler.transform(y)

# Split training and testing sets
X_train, X_test, y_train, y_test = \
    train_test_split(X_scaled, y_scaled, test_size=0.1, train_size=0.9,
                     random_state=0)
print(X_train)
# Setup regressors
neuralNetwork = neural_network.MLPRegressor(hidden_layer_sizes=(35, 35, 35, 35),
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
neuralNetwork.fit(X_train, y_train)

Y_NN = neuralNetwork.predict(X_test)
if (args.verbose):
    print("\n\nFinal Results:")
    print("\ntest stats:")
    print("r2: ", r2_score(y_test, Y_NN))
    print("mse: ", mean_squared_error(y_test, Y_NN))
    print("\ntraining stats:")
    print("loss: ", neuralNetwork.loss_)
    print("validationScore: ", neuralNetwork.best_validation_score_)

# save neural network and scalers
joblib.dump(neuralNetwork, args.fileName + ".pkl")
joblib.dump(X_scaler, args.fileName + "X_scaler.pkl")
joblib.dump(y_scaler, args.fileName + "y_scaler.pkl")
