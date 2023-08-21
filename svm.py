import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC


# Load a dataset into a Pandas DataFrame
train_proteins = pd.read_csv("train_proteins.csv")
train_peptides = pd.read_csv("train_peptides.csv")
train_clinical = pd.read_csv("train_clinical_data.csv")


# Function to prepare dataset with all the steps mentioned above:
def prepare_dataset(train_proteins, train_peptides):
    # Step 1: Grouping
    df_protein_grouped = train_proteins.groupby(['visit_id', 'UniProt'])['NPX'].mean().reset_index()
    df_peptide_grouped = train_peptides.groupby(['visit_id', 'Peptide'])['PeptideAbundance'].mean().reset_index()

    # Step 2: Pivoting
    df_protein = df_protein_grouped.pivot(index='visit_id', columns='UniProt', values='NPX').rename_axis(
        columns=None).reset_index()
    df_peptide = df_peptide_grouped.pivot(index='visit_id', columns='Peptide', values='PeptideAbundance').rename_axis(
        columns=None).reset_index()

    # Step 3: Merging
    pro_pep_df = df_protein.merge(df_peptide, on=['visit_id'], how='left')

    return pro_pep_df


pro_pep_df = prepare_dataset(train_proteins, train_peptides)

# Create an empty dictionary to store the models trained for each label.
model_dict = {}

# Create an empty dictionary to store the mse score of the models trained for each label.
mse_dict = {}

# Create an empty dictionary to store the sMAPE scores of the models trained for each label.
smape_dict = {}

FEATURES = [i for i in pro_pep_df.columns if i not in ["visit_id"]]
FEATURES.append("visit_month")


def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


# List of target labels to loop through and train models
target = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]
sum_mse = 0
sum_r2 = 0
sum_sma = 0

# Loop through each label
for label in target:
    # Merge the label 'visit_id', 'patient_id', 'visit_month' and label columns from `train_clinical`
    # data frame to `pro_prep_df` data frame on the `visit_id` column.
    dataset_df = pro_pep_df.merge(train_clinical[['visit_id', 'patient_id', 'visit_month', label]], on=['visit_id'],
                                  how='left')

    # Drop null value label rows
    dataset_df = dataset_df.dropna(subset=[label])
    # replace other null value with 0
    # col_means = round(dataset_df.mean().astype(float))
    col_means = 0
    dataset_filled = dataset_df.fillna(col_means)

    # Make a new copy of the FEATURES list we created previously. Add `label` to it.
    feature_list = FEATURES.copy()
    feature_list.append(label)

    feature_l = [i for i in feature_list if i not in target]

    # Split the dataset into train and validation datasets.
    X_train, X_test, y_train, y_test = train_test_split(dataset_filled[feature_l], dataset_filled[label], test_size=0.2,
                                                        random_state=10)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # get the best parameter
    # param_grid = {'C':[i for i in range(1,21)]}
    # rbf_svr_cg = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5)
    # rbf_svr_cg.fit(X_train_scaled, y_train)
    # best_c = rbf_svr_cg.best_params_.get('C')
    # best_g = rbf_svr_cg.best_params_.get('gamma')
    # print(f' best_c = {best_c}, best_g = {best_g}')

    model = SVR(kernel='rbf', C=10)

    # Train the model.
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    model_dict[label] = [y_pred, y_test]
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    sMAPE = smape(y_pred, y_test)
    print(f'{label}    mse: {mse}  r2 score: {r2}  sMAPE: {sMAPE} ')
    sum_mse += mse
    sum_r2 += r2
    sum_sma += sMAPE

print(f'Average mse: {sum_mse/4}  Average r2 score: {sum_r2/4}  Average sMAPE: {sum_sma/4}')

for key, value in model_dict.items():
    y_pred = value[0]
    y_test = value[1]
    x = len(y_pred)
    xx = np.array(range(x))
    fig, ax = plt.subplots(1, 1)
    ax.plot(xx, y_pred, label='predict')
    ax.plot(xx, y_test, label='true')
    ax.legend()
plt.show()

