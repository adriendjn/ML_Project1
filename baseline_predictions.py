import numpy as np
import time
from helpers import load_csv_data, create_csv_submission
from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    reg_logistic_regression,
    logistic_regression,
)
from sklearn.metrics import f1_score, accuracy_score

start_time = time.time()

print("Loading data")
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data\dataset")
print("Data loaded in %.3s secs" % (time.time() - start_time))

stime = time.time()
print("\nCleaning the mostly empty features")

missing_treshold = 0.9
full_treshold = 0.1

missing_ratio = np.mean(np.isnan(x_train), axis=0)

full_features = np.where(missing_ratio < full_treshold)[0]

x_clean = x_train[:, full_features]
x_test = x_test[:, full_features]

print("Kept only the features :", full_features)

print("\nCleaning the low variance features")

variance = np.nanvar(x_clean, axis=0)
low_var_treshold = 0.001
high_var_features = np.where(variance > low_var_treshold)[0]

x_clean = x_clean[:, high_var_features]
x_test = x_test[:, high_var_features]

print("Kept only the features :", high_var_features)

print("\nCleaning highly correlated features")

x_filled = np.where(np.isnan(x_clean), np.nanmean(x_clean, axis=0), x_clean)

corr_matrix = np.corrcoef(x_filled, rowvar=False)

low_covar_treshold = 0.9
n_features = corr_matrix.shape[0]
to_remove = set()
for i in range(n_features):
    for j in range(i + 1, n_features):
        if abs(corr_matrix[i, j]) > low_covar_treshold:
            to_remove.add(i)

keep_features = [i for i in range(n_features) if i not in to_remove]

x_clean = x_clean[:, keep_features]
x_test = x_test[:, keep_features]

print("Kept only the features :", keep_features)

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

print("Data cleaned in %.3s secs" % (time.time() - stime))

gammas = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]
N_LAMBDAS = 10
lambdas = [x / N_LAMBDAS for x in range(N_LAMBDAS + 1)]
initial_w = np.ones(x_train_split.shape[1]) / 1.0
n_iter = 20
best_score = 0
best_gamma = None
best_lambda = 0

# Linear regression with MSE and gradient descent
stime = time.time()
best_score_mse_gd = 0
for gamma in gammas:
    w, loss = mean_squared_error_gd(
        y_train_split, x_train_split, initial_w, n_iter, gamma
    )
    y_pred = x_test_split @ w
    y_pred_class = np.where(y_pred >= 0, 1, -1)
    score = accuracy_score(y_test_split, y_pred_class)
    if score > best_score_mse_gd:
        best_score_mse_gd = score
        best_gamma_mse_gd = gamma
        best_w_mse_gd = w
if best_score_mse_gd > best_score:
    best_score = best_score_mse_gd
    best_gamma = best_gamma_mse_gd
    best_w = best_w_mse_gd

print(
    "\nLinear regression GD highest acc: %.3f using gamma: %f (took %.3s secs)\n"
    % (best_score_mse_gd * 100, best_gamma_mse_gd, time.time() - stime)
)

# Linear regression with MSE and stochastic gradient descent
stime = time.time()
best_score_mse_sgd = 0
for gamma in gammas:
    w, loss = mean_squared_error_sgd(
        y_train_split, x_train_split, initial_w, n_iter, gamma
    )
    y_pred = x_test_split @ w
    y_pred_class = np.where(y_pred >= 0, 1, -1)
    score = accuracy_score(y_test_split, y_pred_class)
    if score > best_score_mse_sgd:
        best_score_mse_sgd = score
        best_gamma_mse_sgd = gamma
        best_w_mse_sgd = w
if best_score_mse_sgd > best_score:
    best_score = best_score_mse_sgd
    best_gamma = best_gamma_mse_sgd
    best_w = best_w_mse_sgd

print(
    "Linear regression SGD highest acc: %.3f using gamma: %f (took %.3s secs)\n"
    % (best_score_mse_sgd * 100, best_gamma_mse_sgd, time.time() - stime)
)

# Least Squares regression with MSE and normal equations
stime = time.time()
best_score_ls = 0
w, loss = least_squares(y_train_split, x_train_split)
y_pred = x_test_split @ w
y_pred_class = np.where(y_pred >= 0, 1, -1)
score = accuracy_score(y_test_split, y_pred_class)
best_score_ls = score
best_w_ls = w
if best_score_ls > best_score:
    best_score = best_score_ls
    best_w = best_w_ls

print(
    "Least Squares regression acc: %.3f (took %.3s secs)\n"
    % (best_score_ls * 100, time.time() - stime)
)

# Ridge regression with MSE and normal equations
stime = time.time()
best_score_ridge = 0
for lambda_ in lambdas:
    w, loss = ridge_regression(y_train_split, x_train_split, lambda_)
    y_pred = x_test_split @ w
    y_pred_class = np.where(y_pred >= 0, 1, -1)
    score = accuracy_score(y_test_split, y_pred_class)
    if score > best_score_ridge:
        best_score_ridge = score
        best_lambda_ridge = lambda_
        best_w_ridge = w
if best_score_ridge > best_score:
    best_score = best_score_ridge
    best_lambda = best_lambda_ridge
    best_w = best_w_ridge

print(
    "Ridge regression highest acc: %.3f using lambda: %f (took %.3s secs)\n"
    % (best_score_ridge * 100, best_lambda_ridge, time.time() - stime)
)

# Logistic regression with Log loss and gradient descent
stime = time.time()
best_score_log_gd = 0
for gamma in gammas:
    w, loss = logistic_regression(
        y_train_split, x_train_split, initial_w, n_iter, gamma
    )
    y_pred = x_test_split @ w
    y_pred_class = np.where(y_pred >= 0, 1, -1)
    score = accuracy_score(y_test_split, y_pred_class)
    if score > best_score_log_gd:
        best_score_log_gd = score
        best_gamma_log_gd = gamma
        best_w_log_gd = w
if best_score_log_gd > best_score:
    best_score = best_score_log_gd
    best_gamma = best_gamma_log_gd
    best_w = best_w_log_gd

print(
    "Logistic regression highest acc: %.3f using gamma: %f (took %.3s secs)\n"
    % (best_score_log_gd * 100, best_gamma_log_gd, time.time() - stime)
)

# Logistic regression with Log loss and gradient descent
stime = time.time()
best_score_rlog_gd = 0
for gamma in gammas:
    for lambda_ in lambdas:
        w, loss = reg_logistic_regression(
            y_train_split, x_train_split, lambda_, initial_w, n_iter, gamma
        )
        y_pred = x_test_split @ w
        y_pred_class = np.where(y_pred >= 0, 1, -1)
        score = accuracy_score(y_test_split, y_pred_class)
        if score > best_score_rlog_gd:
            best_score_rlog_gd = score
            best_gamma_rlog_gd = gamma
            best_lambda_rlog_gd = lambda_
            best_w_rlog_gd = w
if best_score_rlog_gd > best_score:
    best_score = best_score_rlog_gd
    best_gamma = best_gamma_rlog_gd
    best_lambda = best_lambda_rlog_gd
    best_w = best_w_rlog_gd

print(
    "Regularized Logistic regression highest acc: %.3f using gamma: %f and lambda: %f (took %.3s secs)\n"
    % (
        best_score_rlog_gd * 100,
        best_gamma_rlog_gd,
        best_lambda_rlog_gd,
        time.time() - stime,
    )
)

test_pred = x_test @ best_w
y_test_pred = np.where(test_pred >= 0, 1, -1)
create_csv_submission(test_ids, y_test_pred, "best_submission.csv")
print("--- %.3s secs ---" % (time.time() - start_time))
