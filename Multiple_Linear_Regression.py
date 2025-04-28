import pandas as pd
import numpy as np

# READ DATA
df = pd.read_csv("Football_players.csv", encoding="ISO-8859-1")
age = df["Age"].to_numpy()
height = df["Height"].to_numpy()
mental = df["Mental"].to_numpy()
skill = df["Skill"].to_numpy()
salary = df["Salary"].to_numpy()

bias = np.ones(len(age))
X = np.column_stack((bias, age, height, mental, skill))
y = salary

#print("X shape:", X.shape)   # (100, 5)
#print("y shape:", y.shape)   # (100,)

# FUNCTIONS
def compute_beta(X, y):
    XT = X.T
    beta = np.linalg.inv(XT @ X) @ XT @ y
    return beta

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

#y_actual = np.array([100, 200, 300])
#y_pred = np.array([110, 190, 310])
#print("MSE:", mean_squared_error(y_actual, y_pred)) MSE: 100.0

#80% EDUCATION, 20% TESTING
split_idx = int(len(X) * 0.8)
X_train = X[:split_idx]
y_train = y[:split_idx]
X_test = X[split_idx:]
y_test = y[split_idx:]

beta = compute_beta(X_train, y_train)
y_pred_test = X_test @ beta
mse_test = mean_squared_error(y_test, y_pred_test)

print("-------ORGİNAL DATA-------")
print("TesT ERROR (%80 - %20 split):")
print("MSE:", mse_test)

#WITH ALL DATA
beta_all = compute_beta(X, y)
y_pred_all = X @ beta_all
mse_all = mean_squared_error(y, y_pred_all)
print("all data test (train = test):")
print("MSE:", mse_all)

#ADD RANDOM COLUMN
random_col = np.random.randint(-1000, 1000, size=len(X))
X_rand = np.column_stack((X, random_col))

# 80-20 split
Xr_train = X_rand[:split_idx]
Xr_test = X_rand[split_idx:]
beta_rand = compute_beta(Xr_train, y_train)
y_pred_rand_test = Xr_test @ beta_rand
mse_rand_test = mean_squared_error(y_test, y_pred_rand_test)


beta_rand_all = compute_beta(X_rand, y)
y_pred_rand_all = X_rand @ beta_rand_all
mse_rand_all = mean_squared_error(y, y_pred_rand_all)

print("---------------DATA RANDOM COLN--------------")
print("Test ERROR (%80 - %20 split):")
print("MSE:", mse_rand_test)

print("T (train = test):")
print("MSE:", mse_rand_all)
