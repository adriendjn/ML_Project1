import numpy as np
from helpers import load_csv_data, create_csv_submission
from implementations import (
    least_squares,
    ridge_regression,
    reg_logistic_regression,
    logistic_regression,
)
from sklearn.metrics import f1_score, accuracy_score

print("Loading data")
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/dataset")

print("Cleaning the mostly empty features")

missing_treshold = 0.9
full_treshold = 0.1

missing_ratio = np.mean(np.isnan(x_train), axis=0)

full_features = np.where(missing_ratio < full_treshold)[0]

x_clean = x_train[:, full_features]
x_test_clean = x_test[:, full_features]  # Renommer pour éviter confusion

print("Kept only the features :", full_features)
print("Cleaning the low variance features")

variance = np.nanvar(x_clean, axis=0)
low_var_treshold = 0.001
high_var_features = np.where(variance > low_var_treshold)[0]

x_clean = x_clean[:, high_var_features]
x_test_clean = x_test_clean[:, high_var_features]

print("Kept only the features :", high_var_features)
print("Cleaning highly correlated features")

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
                binary_column[np.isnan(data[:, col_idx])] = np.nan
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
                print(f"Colonne {col_idx} détectée comme catégorielle: {len(unique_values)} valeurs uniques")
    
    return categorical_indices

x_clean_encoded, feature_names = one_hot_encode_manual(x_clean)
x_test_encoded, _ = one_hot_encode_manual(x_test_clean)


mean = np.nanmean(x_clean_encoded, axis=0)
std = np.nanstd(x_clean_encoded, axis=0)

x_clean_final = np.where(np.isnan(x_clean_encoded), mean, x_clean_encoded)
x_clean_final = (x_clean_final - mean) / std

x_test_final = np.where(np.isnan(x_test_encoded), mean, x_test_encoded)
x_test_final = (x_test_final - mean) / std


# Splitting the cleaned data
n = x_clean_final.shape[0]
indices = np.random.permutation(n)
split = int(n * 0.8)

x_train_split = x_clean_final[indices[:split]]
y_train_split = y_train[indices[:split]]
x_test_split = x_clean_final[indices[split:]]
y_test_split = y_train[indices[split:]]

gammas = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]
lambdas  = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

best_score = 0
best_gamma = None
best_w = None
n_iter = 20

for gamma in gammas:
    for lamdba in lambdas :
        w, loss = reg_logistic_regression(
            y_train_split, x_train_split,lamdba, np.zeros(x_train_split.shape[1]), n_iter, gamma
        )
        y_pred = x_test_split @ w
        y_pred_class = np.where(y_pred >= 0, 1, -1)
        score = accuracy_score(y_test_split, y_pred_class)
        f1score =f1_score(y_test_split, y_pred_class) 
         
        if score > best_score:
        
            best_score = score
            best_f1score = f1score
            best_lambda = lamdba
            best_gamma = gamma
            best_w = w

print("\nMeilleur gamma :", best_gamma)
print("Meilleur lambda :", best_lambda)
print("Meilleur accuracy : ", best_score)
print("Meilleur f1 score : ", best_f1score)


test_pred = x_test_final @ best_w
y_test_pred = np.where(test_pred >= 0, 1, -1)
create_csv_submission(test_ids, y_test_pred, "first_submission.csv")

