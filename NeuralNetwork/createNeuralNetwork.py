import pandas as pd
from sklearn import neural_network, preprocessing
from sklearn.model_selection import train_test_split
import joblib
import argparse


parser = argparse.ArgumentParser(
    description="Train a neural network\nexample:\npython3 NeuralNetwork/createNeuralNetwork.py --nfp=4 nn.pkl")
parser.add_argument(
    "--nfp", help="Train neural networks for a specific nfp (2 to 8), default = all nfp", type=int, default=0, choices=range(2, 9))
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()


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

# Setup regressors
neuralNetwork = neural_network.MLPRegressor(hidden_layer_sizes=(35,35,35,35),
                                            activation='tanh',
                                            solver='adam',
                                            alpha=0.0001,
                                            batch_size='auto',
                                            #learning_rate='constant', (sgd)
                                            learning_rate_init=0.001,
                                            #power_t=0.5, (sgd)
                                            max_iter=1000,
                                            shuffle=True,
                                            random_state=None,
                                            tol=0.0001,
                                            verbose=args.verbose,
                                            warm_start=False,
                                            #momentum=0.9, (sgd)
                                            #nesterovs_momentum=True, (sgd)
                                            early_stopping=False,
                                            validation_fraction=0.1,
                                            beta_1=0.9, #(adam)
                                            beta_2=0.999, #(adam)
                                            epsilon=1e-08, #(adam)
                                            n_iter_no_change=10,
                                            )

# Train
neuralNetwork.fit(X_train, y_train)

# save neural network and scalers
joblib.dump(neuralNetwork, "NeuralNetwork/neuralNetwork.pkl")
joblib.dump(X_scaler, "NeuralNetwork/X_scaler.pkl")
joblib.dump(y_scaler, "NeuralNetwork/y_scaler.pkl")
