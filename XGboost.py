import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBClassifier, plot_importance



train_clinical = pd.read_csv(r"C:\Users\Administrator.LAPTOP-IKOB9DL5\PycharmProjects\9417project\train_clinical_data.csv")
train_peptides = pd.read_csv(r"C:\Users\Administrator.LAPTOP-IKOB9DL5\PycharmProjects\9417project\train_peptides.csv.zip")
train_proteins = pd.read_csv(r"C:\Users\Administrator.LAPTOP-IKOB9DL5\PycharmProjects\9417project\train_proteins.csv.zip")


def prepare_dataset(train_proteins, train_peptides):

    df_protein_grouped = train_proteins.groupby(['visit_id', 'UniProt'])['NPX'].mean().reset_index()
    df_peptide_grouped = train_peptides.groupby(['visit_id', 'Peptide'])['PeptideAbundance'].mean().reset_index()

    df_protein = df_protein_grouped.pivot(index='visit_id', columns='UniProt', values='NPX').rename_axis(
        columns=None).reset_index()
    df_peptide = df_peptide_grouped.pivot(index='visit_id', columns='Peptide', values='PeptideAbundance').rename_axis(
        columns=None).reset_index()

    pro_pep_df = df_protein.merge(df_peptide, on=['visit_id'], how='left')

    return pro_pep_df


pro_pep_df = prepare_dataset(train_proteins, train_peptides)


def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


model_dict = {}
mse_dict = {}
smape_dict = {}
r2_score_dict={}

FEATURES = [i for i in pro_pep_df.columns if i not in ["visit_id"]]
FEATURES.append("visit_month")


target = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]

# Loop through each label
for label in target:

    dataset_df = pro_pep_df.merge(train_clinical[['visit_id', 'patient_id', 'visit_month', label]], on=['visit_id'],
                                  how='left')
    dataset_df = dataset_df.dropna(subset=[label])
    feature_list = FEATURES.copy()
    feature_list.append(label)
    feature_1 = [i for i in feature_list if i not in target]
    X_train, X_test, y_train, y_test = train_test_split(dataset_df[feature_1],dataset_df[label], test_size=0.2, random_state=27)

    if label in ["updrs_1"]:
        rf = xgb.XGBRegressor(learning_rate=0.046,
                              n_estimators=685,
                              max_depth=4,
                              min_child_weight=4,
                              subsample=0.84,
                              seed=27,
                              gamma=0.0,
                              )
    if label in ["updrs_2"]:
        rf = xgb.XGBRegressor(learning_rate=0.046,
                              n_estimators=692,
                              max_depth=3,
                              min_child_weight=4,
                              subsample=0.44,
                              seed=27,
                              gamma=0.0,
                              )
    if label in ["updrs_3"]:
        rf = xgb.XGBRegressor(learning_rate=0.056,
                              n_estimators=483,
                              max_depth=4,
                              min_child_weight=6,
                              subsample=0.55,
                              seed=27,
                              gamma=0.0,
                              )
    if label in ["updrs_4"]:
        rf = xgb.XGBRegressor(learning_rate=0.03,
                             n_estimators=406,
                             max_depth=6,
                             min_child_weight=4,
                             subsample=0.5,
                             gamma=0.0,
                             seed=27,
                             )

    rf.fit(X_train,y_train)

    model_dict[label] = rf

    preds = rf.predict(X_test)

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_importance(rf,height=0.5,ax=ax,max_num_features=60)
    plt.show()


    mse_dict[label]=metrics.mean_squared_error(y_test.values.tolist(), preds.flatten())

    r2_score_dict[label]=r2_score(y_test.values.tolist(), preds.flatten())

    smape_dict[label] = smape(y_test.values.tolist(), preds.flatten())


for name, value in mse_dict.items():
  print(f"label {name}: mse {value:.4f}")

print("\nAverage mse", sum(mse_dict.values())/4)

print('\n')


for name, value in r2_score_dict.items():
  print(f"label {name}: r2_score {value:.4f}")

print("\nAverage r2_score", sum(r2_score_dict.values())/4)



print('\n')

for name, value in smape_dict.items():
  print(f"label {name}: sMAPE {value:.4f}")

print("\nAverage sMAPE", sum(smape_dict.values())/4)