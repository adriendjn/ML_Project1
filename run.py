import numpy as np
from helpers import load_csv_data, create_csv_submission
from implementations import (
    least_squares,
    ridge_regression,
    reg_logistic_regression,
    logistic_regression,
    sigmoid,
)
from sklearn.metrics import f1_score, accuracy_score


def is_categorical(col, threshold=20):
    unique = np.unique(col[~np.isnan(col)])
    return len(unique) < threshold


def kFold(X, y, k=5):
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)

    res = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])
        res.append((X[train_idx], X[val_idx], y[train_idx], y[val_idx]))
    return res


np.random.seed(10)

print("Loading data")
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/dataset")

print("Cleaning the mostly empty features")

full_treshold = 0.4

missing_ratio = np.mean(np.isnan(x_train), axis=0)

full_features = np.where(missing_ratio < full_treshold)[0]

x_clean = x_train[:, full_features]
x_test_clean = x_test[:, full_features]  # Renommer pour éviter confusion

print("Kept only the features :", full_features)
print("Cleaning the low variance features")

categ = []
# Missing value handling
for i in range(x_clean.shape[1]):
    train_col = x_clean[:, i]
    test_col = x_test_clean[:, i]
    if is_categorical(train_col):
        vals, counts = np.unique(train_col[~np.isnan(train_col)], return_counts=True)
        mode = vals[np.argmax(counts)]
        train_col[np.isnan(train_col)] = mode
        test_col[np.isnan(test_col)] = mode
        categ.append(i)
    else:
        mean = np.nanmean(train_col)
        train_col[np.isnan(train_col)] = mean
        test_col[np.isnan(test_col)] = mean
    x_test_clean[:, i] = test_col
    x_clean[:, i] = train_col

variance = np.var(x_clean, axis=0)
low_var_treshold = 0.001
high_var_features = np.where(variance > low_var_treshold)[0]

x_clean = x_clean[:, high_var_features]
x_test_clean = x_test_clean[:, high_var_features]

print("Kept only the features :", high_var_features)
print("Cleaning highly correlated features")


corr_matrix = np.corrcoef(x_clean, rowvar=False)

low_covar_treshold = 0.9
n_features = corr_matrix.shape[0]
to_remove = set()
for i in range(n_features):
    for j in range(i + 1, n_features):
        if abs(corr_matrix[i, j]) > low_covar_treshold:
            to_remove.add(i)

keep_features = [i for i in range(n_features) if i not in to_remove]

x_clean = x_clean[:, keep_features]
x_test_clean = x_test_clean[:, keep_features]

print("Cleaning by One Hot Encoding :\n")


def one_hot_encode_manual(data, categorical_indices=None):

    if categorical_indices is None:
        categorical_indices = detect_categorical_columns(data)

    encoded_data = []
    feature_names = []

    for col_idx in range(data.shape[1]):
        if col_idx in categorical_indices:
            unique_values = np.unique(data[:, col_idx])
            unique_values = unique_values[~np.isnan(unique_values)]

            print(f"Colonne {col_idx}: {len(unique_values)} valeurs uniques")

            for val_idx, value in enumerate(unique_values):
                binary_column = (data[:, col_idx] == value).astype(float)
                encoded_data.append(binary_column)
                feature_names.append(f"col{col_idx}_val{val_idx}")
        else:
            encoded_data.append(data[:, col_idx])
            feature_names.append(f"col{col_idx}_num")

    return np.column_stack(encoded_data), feature_names


def detect_categorical_columns(data, max_unique_values=5):
    categorical_indices = []

    for col_idx in range(data.shape[1]):
        column = data[:, col_idx]
        non_nan_values = column[~np.isnan(column)]

        if len(non_nan_values) > 0:
            unique_values = np.unique(non_nan_values)
            if len(unique_values) <= max_unique_values:
                categorical_indices.append(col_idx)
                print(
                    f"Column {col_idx} detected as categorial : {len(unique_values)} unique values"
                )

    return categorical_indices


x_clean_encoded, feature_names = one_hot_encode_manual(x_clean)
x_test_encoded, _ = one_hot_encode_manual(x_test_clean)


mean = np.mean(x_clean_encoded, axis=0)
std = np.std(x_clean_encoded, axis=0)
std[std == 0] = 1

x_clean_final = (x_clean_encoded - mean) / std

x_test_final = (x_test_encoded - mean) / std

gammas = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]
lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]

best_f1 = 0
best_gamma = None
best_w = None
n_iter = 100

for gamma in gammas:
    for lambda_ in lambdas:
        print(gamma, lambda_)
        folds = kFold(x_clean_final, y_train)
        f1_scores = []
        accuracy_scores = []
        for x_train_fold, x_test_fold, y_train_fold, y_test_fold in folds:
            w, loss = reg_logistic_regression(
                y_train_fold,
                x_train_fold,
                lambda_,
                np.zeros(x_train_fold.shape[1]),
                n_iter,
                gamma,
            )
            y_pred = sigmoid(x_test_fold @ w)
            y_pred_class = np.where(y_pred >= 0.8, 1, -1)

            accuracy_scores.append(accuracy_score(y_test_fold, y_pred_class))
            f1_scores.append(f1_score(y_test_fold, y_pred_class))
        accuracy = np.mean(accuracy_scores)
        f1 = np.mean(f1_scores)
        if f1 > best_f1:
            best_f1 = f1
            best_accuracy = accuracy
            best_lambda = lambda_
            best_gamma = gamma

print("\Best gamma :", best_gamma)
print("Best lambda :", best_lambda)
print("Best accuracy : ", best_accuracy)
print("Best f1 score : ", best_f1)

print("Final Training")

best_w, final_loss = reg_logistic_regression(
    y_train,
    x_clean_final,
    best_lambda,
    np.zeros(x_clean_final.shape[1]),
    200,
    best_gamma,
)


test_pred = sigmoid(x_test_final @ best_w)
y_test_pred = np.where(test_pred >= 0.8, 1, -1)
create_csv_submission(test_ids, y_test_pred, "submission.csv")
