import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, partial
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance



train_clinical = pd.read_csv(r"C:\Users\Administrator.LAPTOP-IKOB9DL5\PycharmProjects\9417project\train_clinical_data.csv")
train_peptides = pd.read_csv(r"C:\Users\Administrator.LAPTOP-IKOB9DL5\PycharmProjects\9417project\train_peptides.csv.zip")
train_proteins = pd.read_csv(r"C:\Users\Administrator.LAPTOP-IKOB9DL5\PycharmProjects\9417project\train_proteins.csv.zip")



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


def smape(A, F):

    return  2.0 * np.mean(np.abs(F- A) / (np.abs(F) + np.abs(A))) * 100
    # return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def xgboostproducing(space):


    params = {'nthread': -1,
              'max_depth': space['max_depth'],
              'n_estimators': space['n_estimators'],
              'eta': space['learning_rate'],
              'subsample': space['subsample'],
              'min_child_weight': space['min_child_weight'],
              'objective': 'reg:linear',
              'silent': 0,
              'gamma': 0,
              'alpha': 0,
              'lambda': 0,
              'scale_pos_weight': 0,
              'seed': 27,
              }

    xrf = xgb.train(params, dtrain, params['n_estimators'],evallist, early_stopping_rounds=100)

    return get_score(xrf)


def get_score(tranformer):
    xrf = tranformer
    dpredict = xgb.DMatrix(X_test)
    prediction = xrf.predict(dpredict, ntree_limit=xrf.best_ntree_limit)

    return smape(y_test, prediction)


model_dict = {}
mse_dict = {}
smape_dict = {}

FEATURES = [i for i in pro_pep_df.columns if i not in ["visit_id"]]
FEATURES.append("visit_month")

# List of target labels to loop through and train models
target = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]

# Loop through each label
for label in target:
    dataset_df = pro_pep_df.merge(train_clinical[['visit_id', 'patient_id', 'visit_month', label]], on=['visit_id'],
                                  how='left')
    dataset_df = dataset_df.dropna(subset=[label])
    feature_list = FEATURES.copy()
    feature_list.append(label)

    feature_1 = [i for i in feature_list if i not in target]

    X_train, X_test, y_train, y_test = train_test_split(dataset_df[feature_1], dataset_df[label], test_size=0.2,
                                                        random_state=27)

    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dtest = xgb.DMatrix(data=X_test, label=y_test)
    evallist = [(dtest, 'test'), (dtrain, 'train')]

    space = {
        'max_depth': hp.uniformint('max_depth',3,8),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
        "n_estimators": hp.uniformint("n_estimators", 300, 800),
        'subsample': hp.uniform('subsample',0.001,1),
        'min_child_weight': hp.uniformint('min_child_weight',3,10),
     }

    algo = partial(tpe.suggest, n_startup_jobs=1)
    best_for_model = fmin(xgboostproducing, space, algo=algo, max_evals=20, pass_expr_memo_ctrl=None)
    print(best_for_model)

