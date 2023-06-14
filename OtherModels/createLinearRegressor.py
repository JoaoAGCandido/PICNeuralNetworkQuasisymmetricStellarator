import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# Load and partition data
df = pd.read_csv("../scans/scan7/scan7Clean.csv.zip")
# select nfp

df = df[df['nfp'] == 3]
y = df.loc[:, ['rc1', 'zs1', 'eta']]

X = df.loc[:, ['RotTrans', 'axLenght', 'max_elong']]


# Split training and testing sets
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.1, train_size=0.9,
                     random_state=0)

# Split training and validationsets
X_train, X_validation, y_train, y_validation = \
    train_test_split(X_train, y_train, test_size=0.1, train_size=0.9,
                     random_state=0)

#scale
X_scaler = preprocessing.StandardScaler()
#X_scaler = preprocessing.RobustScaler()
X_scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
X_validation_scaled =  X_scaler.transform(X_validation)

y_scaler = preprocessing.StandardScaler()
#y_scaler = preprocessing.RobustScaler()
y_scaler.fit(y_train)
y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)
y_validation_scaled =  y_scaler.transform(y_validation)


out={
    "index":[],
    "R2_validation":[]
}

#for i in range(2, 20):
#model = make_pipeline(preprocessing.SplineTransformer(n_knots=100, degree=100, extrapolation = "continue"), Ridge(alpha=1e-3))
#model = make_pipeline(preprocessing.PolynomialFeatures(degree=12), LinearRegression())
#model = Ridge(alpha=1)
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)

test_predictions = model.predict(X_test_scaled)
validation_predictions = model.predict(X_validation_scaled)
#print(test_predictions[:,0])
#print(y_test_scaled)
#print("\n",i)
print("test stats:")
print("r2: ", r2_score(y_test_scaled, test_predictions))
print("mse: ", mean_squared_error(y_test_scaled, test_predictions))
print("\nvalidation stats:")
print("r2: ", r2_score(y_validation_scaled, validation_predictions))
print("mse: ", mean_squared_error(y_validation_scaled, validation_predictions))
#out["index"].append(i)
#out["R2_validation"].append(r2_score(y_validation_scaled, validation_predictions))
#df = pd.DataFrame(out)
#print(df.sort_values("R2_validation"))
fig, ax = plt.subplots()

ax.scatter(test_predictions[:,2],y_test_scaled[:,2], c="blue", label="eta",s=2)
ax.scatter(test_predictions[:,0],y_test_scaled[:,0], c="orange", label="rc1", s=2)
ax.scatter(test_predictions[:,1],y_test_scaled[:,1], c="green", label="zs1", s=2)

plt.axline([0, 0], [1, 1], color='red')
plt.ylim(-3, 3)
plt.xlim(-3, 3)
plt.xlabel('Predicted')
plt.ylabel('Actual')
ax.legend()