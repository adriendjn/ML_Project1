import numpy as np
from helpers import load_csv_data, create_csv_submission
from implementations import least_squares, ridge_regression, reg_logistic_regression, logistic_regression, sigmoid
from sklearn.metrics import f1_score, accuracy_score

np.random.seed(10)

def normalize(x_train, x_test):
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std[std==0] = 1

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return x_train, x_test

def is_categorical(col, threshold=20):
    unique = np.unique(col[~np.isnan(col)])
    return len(unique) < threshold

def stratified_split(X, y, test_size=0.2):
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == -1)[0]
    np.random.shuffle(idx_pos)
    np.random.shuffle(idx_neg)
    n_pos = int(len(idx_pos) * (1 - test_size))
    n_neg = int(len(idx_neg) * (1 - test_size))
    train_idx = np.concatenate([idx_pos[:n_pos], idx_neg[:n_neg]])
    test_idx = np.concatenate([idx_pos[n_pos:], idx_neg[n_neg:]])
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

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
                binary_column[np.isnan(data[:, col_idx])] = np.nan
                encoded_data.append(binary_column)
                feature_names.append(f"col{col_idx}_val{val_idx}")
        else:
            encoded_data.append(data[:, col_idx])
            feature_names.append(f"col{col_idx}_num")
    
    return np.column_stack(encoded_data), feature_names

def detect_categorical_columns(data, max_unique_values=20):
    categorical_indices = []
    
    for col_idx in range(data.shape[1]):
        column = data[:, col_idx]
        non_nan_values = column[~np.isnan(column)]
        
        if len(non_nan_values) > 0:
            unique_values = np.unique(non_nan_values)
            if len(unique_values) <= max_unique_values:
                categorical_indices.append(col_idx)
                print(f"Colonne {col_idx} détectée comme catégorielle: {len(unique_values)} valeurs uniques")
    
    return categorical_indices

sparse_feature_threshold = 0.6
corr_threshold = 0.95

print("Loading data")
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/dataset")

# Sparse feature deletion
missing_ratio = np.mean(np.isnan(x_train), axis=0)
full_features = np.where(missing_ratio < sparse_feature_threshold)[0]

x_train = x_train[:, full_features]
x_test = x_test[:, full_features]

categ = []
# Missing value handling
for i in range(x_train.shape[1]):
    train_col = x_train[:,i]
    test_col = x_test[:,i]
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
    x_test[:,i] = test_col
    x_train[:,i] = train_col

print("Cleaning by One Hot Encoding :\n")

x_train, feature_names = one_hot_encode_manual(x_train)
x_test, _ = one_hot_encode_manual(x_test)

# Variance cleaning
var = np.var(x_train, axis=0)
non_constant_features = np.where(var > 0)[0]
x_train = x_train[:, non_constant_features]
x_test = x_test[:, non_constant_features]

# Correlation cleaning
corr_matrix = np.corrcoef(x_train, rowvar=False)
to_remove = set()
n_features = corr_matrix.shape[0]
for i in range(n_features):
    for j in range(i+1, n_features):
        if abs(corr_matrix[i,j]) > corr_threshold:
            to_remove.add(j)
uncorr_features = [i for i in range(n_features) if i not in to_remove]
x_train = x_train[:, uncorr_features]
x_test = x_test[:, uncorr_features]

# x_train_split, x_test_split, y_train_split, y_test_split = stratified_split(x_train, y_train)
folds = kFold(x_train, y_train)

print("Cleaning finished")
# Optimization of the hyperparameters
sig_treshold = 0.8
gammas = np.arange(0, 1, 0.1)
lambdas = np.arange(0, 1, 0.1)
best_score = 0
n_iter = 100
lambda_ = 0.1
gamma = 0.2
for gamma in gammas:
    print(gamma)
    sig_scores = []
    for x_train_split, x_test_split, y_train_split, y_test_split in folds:
        x_train_split, x_test_split = normalize(x_train_split, x_test_split)
        w, loss = logistic_regression(y_train_split, x_train_split, np.zeros(x_train_split.shape[1]), n_iter, gamma)
        sig_y_pred = sigmoid(x_test_split @ w)
        sig_pred_class = np.where(sig_y_pred >= sig_treshold, 1, -1)
        sig_score = f1_score(y_test_split, sig_pred_class)
        sig_scores.append(sig_score)
    sig_mean = np.mean(sig_scores)
    if sig_mean > best_score:
        best_score = sig_mean
        best_gamma = gamma
        best_pred = sig_pred_class

print("Before threshold optimisation")

print("Meilleur gamma :", best_gamma)
print("F1 Score : ", best_score)
print(np.unique(best_pred, return_counts=True))


# Training the final model 
x_train, x_test = normalize(x_train, x_test)

w, loss = logistic_regression(y_train, x_train, np.zeros(x_train.shape[1]), 200, best_gamma)

sig_test_pred = sigmoid(x_test @ w)
y_test_pred = np.where(sig_test_pred > sig_treshold, 1, -1)
print("Valeurs uniques training set: ", np.unique(y_test_pred, return_counts=True))
create_csv_submission(test_ids, y_test_pred, "finalSubmission.csv")