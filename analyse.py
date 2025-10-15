import numpy as np
from helpers import load_csv_data

x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/dataset")

print("Forme des jeux de données")

print("x_train : ", x_train.shape, x_train.dtype)
print("y_train : ", y_train.shape, y_train.dtype)
print("x_test : ", x_test.shape, x_test.dtype)

print("Analyse du remplissage des Features")

missing_treshold = 0.9
full_treshold = 0.1

missing_ratio = np.mean(np.isnan(x_train), axis=0)

missing_features = np.where(missing_ratio > missing_treshold)[0]
full_features = np.where(missing_ratio < full_treshold)[0]

print("     Features principalement vides : ", missing_features)

print("     Features principalement pleines : ", full_features)

print("Analyse de la variance des features")

variance = np.nanvar(x_train, axis=0)
low_var_treshold = 0.001
low_var_features = np.where(variance < low_var_treshold)[0]

print("     Features avec variance très basse : ", low_var_features)

print("Analyse de covariance des features")
x_filled = np.where(np.isnan(x_train), np.nanmean(x_train, axis=0), x_train)

corr_matrix = np.corrcoef(x_filled, rowvar=False)

low_covar_treshold = 0.9
corr_pair = []
n_features = corr_matrix.shape[0]

for i in range(n_features):
    for j in range(i+1, n_features):
        if abs(corr_matrix[i,j]) > low_covar_treshold:
            corr_pair.append((i, j, corr_matrix[i,j]))
for i, j, corr in corr_pair:
    print(f"Colonne {i} et {j} : correlation = {corr:.2f}")
