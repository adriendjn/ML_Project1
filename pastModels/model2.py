import numpy as np
from helpers import load_csv_data, create_csv_submission
from implementations import least_squares, ridge_regression, reg_logistic_regression, logistic_regression
from sklearn.metrics import f1_score, accuracy_score

print("Loading data")
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/dataset")

print("Cleaning the mostly empty features")

missing_treshold = 0.9
full_treshold = 0.2

missing_ratio = np.mean(np.isnan(x_train), axis=0)

full_features = np.where(missing_ratio < full_treshold)[0]

x_clean = x_train[:, full_features]
x_test = x_test[:, full_features]

print("Kept only the features :", full_features)
print("Cleaning the low variance features")

variance = np.nanvar(x_clean, axis=0)
low_var_treshold = 0.005
high_var_features = np.where(variance > low_var_treshold)[0]

x_clean = x_clean[:, high_var_features]
x_test = x_test[:, high_var_features]

print("Kept only the features :", high_var_features)
print("Cleaning highly correlated features")

x_filled = np.where(np.isnan(x_clean), np.nanmean(x_clean, axis=0), x_clean)

corr_matrix = np.corrcoef(x_filled, rowvar=False)

low_covar_treshold = 0.9
n_features = corr_matrix.shape[0]
to_remove = set()
for i in range(n_features):
    for j in range(i+1, n_features):
        if abs(corr_matrix[i,j]) > low_covar_treshold:
            to_remove.add(i)

keep_features = [i for i in range(n_features) if i not in to_remove]

x_clean = x_clean[:, keep_features]
x_test = x_test[:, keep_features]

# Normalisation
mean = np.nanmean(x_clean, axis=0)
x_clean = np.where(np.isnan(x_clean), mean, x_clean)
std = np.nanstd(x_clean, axis=0)

x_clean = (x_clean - mean) / std

x_test = np.where(np.isnan(x_test), mean, x_test)
x_test = (x_test - mean) / std

# Splitting the cleaned data

n = x_clean.shape[0]
indices = np.random.permutation(n)
split = int(n * 0.8)

x_train_split = x_clean[indices[:split]]
y_train_split = y_train[indices[:split]]
x_test_split = x_clean[indices[split:]]
y_test_split = y_train[indices[split:]]

gammas = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]
lambdas = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
best_score = 0
best_gamma = None
best_lambda = None
n_iter = 40

for gamma in gammas:
    for lambda_ in lambdas:
        w, loss = reg_logistic_regression(y_train_split, x_train_split, lambda_, np.zeros(x_train_split.shape[1]), n_iter, gamma)
        y_pred = x_test_split @ w
        y_pred_class = np.where(y_pred >= 0, 1, -1)
        score = f1_score(y_test_split, y_pred_class)
        if score > best_score:
            best_score = score
            best_gamma = gamma
            best_lambda = lambda_
            best_w = w

print("Meilleur gamma :", best_gamma)
print("Meilleur lambda :", best_lambda)
print("F1 score : ", best_score)

test_pred = x_test @ best_w
y_test_pred = np.where(test_pred >= 0, 1, -1)
create_csv_submission(test_ids, y_test_pred, "model2.csv")