import pandas as pd
from sklearn import neural_network
from sklearn.model_selection import train_test_split
import joblib


# Load and partition data
df = pd.read_csv("scans/scan4/scan4.csv")
X = df.loc[:, ['RotTrans', 'axLenght', 'max_elong']]
y = df.loc[:, ['nfp', 'rc1', 'zs1', 'eta']]

# Split training and testing sets
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, train_size=0.7,
                     random_state=0)

# Setup regressors
neuralNetwork = neural_network.MLPRegressor(hidden_layer_sizes=(20, 20, 20),
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
                                            verbose=False,
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

# save neural network
joblib.dump(neuralNetwork, "NeuralNetwork/neuralNetwork.pkl")
