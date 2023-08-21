# import packages
import random
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# load the dataset
train_proteins = pd.read_csv("E:/Courses/COMP9417/amp-parkinsons-disease-progression-prediction/train_proteins.csv")
train_peptides = pd.read_csv("E:/Courses/COMP9417/amp-parkinsons-disease-progression-prediction/train_peptides.csv")
train_clinical = pd.read_csv("E:/Courses/COMP9417/amp-parkinsons-disease-progression-prediction/train_clinical_data.csv")

# prepare the data set
def prepare_dataset(train_proteins, train_peptides):
    # Step 1: Grouping 
    df_protein_grouped = train_proteins.groupby(["visit_id", "UniProt"])["NPX"].mean().reset_index()
    df_peptide_grouped = train_peptides.groupby(["visit_id", "Peptide"])["PeptideAbundance"].mean().reset_index()
    
    # Step 2: Pivoting
    df_protein = df_protein_grouped.pivot(index = "visit_id", columns = "UniProt", values = "NPX").rename_axis(columns = None).reset_index()
    df_peptide = df_peptide_grouped.pivot(index = "visit_id",columns = "Peptide", values = "PeptideAbundance").rename_axis(columns = None).reset_index()
    
    # Step 3: Merging
    pro_pep_df = df_protein.merge(df_peptide, on = ["visit_id"], how = "left")
    
    return pro_pep_df

pro_pep_df = prepare_dataset(train_proteins, train_peptides)

data = pro_pep_df
clinical = pd.read_csv("E:/Courses/COMP9417/amp-parkinsons-disease-progression-prediction/train_clinical_data.csv")

# split training data and test data
def split_dataset(dataset, test_ratio = 0.20, random_state = 27):
    random.seed(random_state)
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

# scoring metric of the competition (SMAPE)
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

# we need to predict updrs_1, updrs_2, updrs_3, updrs_4
# one model for each label, 4 models in total
model_dict = {} # store the models
mse_dict = {} # store the mse
smape_dict = {} # store the SMAPE

# features used for training
# do not use visit_id, and add column visit_month from clinic
FEATURES = [i for i in data.columns if i not in ["visit_id"]]
FEATURES.append("visit_month")

# data processing
def data_processing(label, n_components):
    df = data.merge(clinical[["visit_id", "patient_id", "visit_month", label]], on = ["visit_id"], how = "left")

    # drop the null values with respect to the label row
    df = df.dropna(subset = [label])

    feature_list = FEATURES.copy() 
    feature_list.append(label)

    # normalize the features
    X = df[feature_list]
    # interpolation
    X = X.bfill().ffill().interpolate(method = "polynomial", order = 2)
    y = X[label]
    X = X.drop(label, axis = 1)
    cols = X.columns
    scaler = MinMaxScaler()
    normalized_X = scaler.fit_transform(X)
    normalized_X = pd.DataFrame(normalized_X, columns = cols)
    normalized_X[label] = y.reset_index(drop = True)

    # now normalized_X is our data frame, we perform PCA and select 200 features
    X = normalized_X.drop(columns = [label]) 
    y = normalized_X[label]

    n = n_components
    pca = PCA(n_components = n)
    X_pca = pca.fit_transform(X)

    # principal components
    pc = pd.DataFrame(data = X_pca, columns = [f"PC{i+1}" for i in range(n)])
    pc[label] = y.reset_index(drop = True)
    
    # return the principal components
    return pc

# MLP
def MLP(label, X_train, X_val, y_train, y_val, opt, alpha):
    model = keras.Sequential([
    keras.layers.Dense(64, activation = "relu", input_shape = (X_train.shape[1], )),
    keras.layers.Dense(32, activation = "relu"),
    keras.layers.Dense(1)
    ])

    # optimizer
    if opt == "adam":
        optimizer = keras.optimizers.Adam(learning_rate = alpha)
    elif opt == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate = alpha)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate = alpha)
    model.compile(optimizer = optimizer, loss = "mean_absolute_error")
    
    # fit the model
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # early stopping
    early_stopping = EarlyStopping(monitor = "val_loss", patience = 10, verbose = 0, restore_best_weights = True)
    history = model.fit(X_train, y_train, epochs = 1000, batch_size = 32, 
                        validation_data = (X_val, y_val), verbose = 0, callbacks = [early_stopping])

    # make predictions
    pred = model.predict(X_val)

    # compute the validation MSE
    #mse = history.history["val_loss"][-1]
    mse = mean_squared_error(pred, y_val)

    # compute the validation SMAPE
    smape_ = smape(pred.reshape((pred.shape[0], )), y_val)

    # compute the r2 value
    try:
        r2 = r2_score(y_val, pred)
    except:
        r2 = 0

    # save the model
    model_dict[label + " " + opt + " " + str(alpha)] = model

    print(opt + " " + str(alpha) + " MSE: " + str(mse) + " SMAPE: " + str(smape_) + " r2: " + str(r2))

# function to run the model
def run_model(l):
    for optimizer in ["adam", "SGD", "RMSprop"]:
        for alpha in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
            MLP(l, X_train, X_val, y_train, y_val, optimizer, alpha)

# label = updrs_1
train_df, val_df = split_dataset(data_processing("updrs_1", 200)) # choose feature = 200
y_train = train_df["updrs_1"]
y_val = val_df["updrs_1"]
X_train = train_df.drop("updrs_1", axis = 1)
X_val = val_df.drop("updrs_1", axis = 1)
run_model("updrs_1")

# label = updrs_2
train_df, val_df = split_dataset(data_processing("updrs_2", 200)) # choose feature = 200
y_train = train_df["updrs_2"]
y_val = val_df["updrs_2"]
X_train = train_df.drop("updrs_2", axis = 1)
X_val = val_df.drop("updrs_2", axis = 1)
run_model("updrs_2")

# label = updrs_3
train_df, val_df = split_dataset(data_processing("updrs_3", 200)) # choose feature = 200
y_train = train_df["updrs_3"]
y_val = val_df["updrs_3"]
X_train = train_df.drop("updrs_3", axis = 1)
X_val = val_df.drop("updrs_3", axis = 1)
run_model("updrs_3")

# label = updrs_4
train_df, val_df = split_dataset(data_processing("updrs_4", 200)) # choose feature = 200
y_train = train_df["updrs_4"]
y_val = val_df["updrs_4"]
X_train = train_df.drop("updrs_4", axis = 1)
X_val = val_df.drop("updrs_4", axis = 1)
run_model("updrs_4")

# deep learning model (similar to MLP, with more hidden layers)
def deep_learning(label, X_train, X_val, y_train, y_val, opt, alpha):
    model = keras.Sequential([
    keras.layers.Dense(64, activation = "relu", input_shape = (X_train.shape[1], )),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dense(256, activation = "relu"),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dense(64, activation = "relu"),
    keras.layers.Dense(32, activation = "relu"),
    keras.layers.Dense(1)
    ])

    # optimizer
    if opt == "adam":
        optimizer = keras.optimizers.Adam(learning_rate = alpha)
    elif opt == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate = alpha)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate = alpha)
    model.compile(optimizer = optimizer, loss = "mean_absolute_error")
    
    # fit the model
    # early stopping
    early_stopping = EarlyStopping(monitor = "val_loss", patience = 10, verbose = 0, restore_best_weights = True)

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    history = model.fit(X_train, y_train, epochs = 1000, batch_size = 32, 
                        validation_data = (X_val, y_val), verbose = 0, callbacks = [early_stopping])

    # make predictions
    pred = model.predict(X_val)

    # compute the validation MSE
    mse = mean_squared_error(pred, y_val)

    # compute the validation SMAPE
    smape_ = smape(pred.reshape((pred.shape[0], )), y_val)

    # compute the r2 value
    try:
        r2 = r2_score(y_val, pred)
    except:
        r2 = 0

    # save the model
    model_dict[label + " " + opt + " " + str(alpha) + " deep"] = model

    print(opt + " " + str(alpha) + " MSE: " + str(mse) + " SMAPE: " + str(smape_) + " r2: " + str(r2))

# function to run the deep learning model
def run_model_deep(l):
    for optimizer in ["adam", "SGD", "RMSprop"]:
        for alpha in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
            deep_learning(l, X_train, X_val, y_train, y_val, optimizer, alpha)

# label = updrs_1
train_df, val_df = split_dataset(data_processing("updrs_1", 200)) # choose feature = 200
y_train = train_df["updrs_1"]
y_val = val_df["updrs_1"]
X_train = train_df.drop("updrs_1", axis = 1)
X_val = val_df.drop("updrs_1", axis = 1)
run_model_deep("updrs_1")

# label = updrs_2
train_df, val_df = split_dataset(data_processing("updrs_2", 200)) # choose feature = 200
y_train = train_df["updrs_2"]
y_val = val_df["updrs_2"]
X_train = train_df.drop("updrs_2", axis = 1)
X_val = val_df.drop("updrs_2", axis = 1)
run_model_deep("updrs_2")

# label = updrs_3
train_df, val_df = split_dataset(data_processing("updrs_3", 200)) # choose feature = 200
y_train = train_df["updrs_3"]
y_val = val_df["updrs_3"]
X_train = train_df.drop("updrs_3", axis = 1)
X_val = val_df.drop("updrs_3", axis = 1)
run_model_deep("updrs_3")

# label = updrs_4
train_df, val_df = split_dataset(data_processing("updrs_4", 200)) # choose feature = 200
y_train = train_df["updrs_4"]
y_val = val_df["updrs_4"]
X_train = train_df.drop("updrs_4", axis = 1)
X_val = val_df.drop("updrs_4", axis = 1)
run_model_deep("updrs_4")

'''
for MLP, the best model for 4 labels are:
label1: RMSprop 0.01 
label2: RMSprop 0.05
label3: adam 0.005
label4: adam 0.0001
'''
# store the training loss and validation loss
t_loss_mlp = []
v_loss_mlp = []

# function to plot MSE vs epochs and fitted plot
def MLP_plot(label, X_train, X_val, y_train, y_val, opt, alpha):
    model = keras.Sequential([
    keras.layers.Dense(64, activation = "relu", input_shape = (X_train.shape[1], )),
    keras.layers.Dense(32, activation = "relu"),
    keras.layers.Dense(1)
    ])

    # optimizer
    if opt == "adam":
        optimizer = keras.optimizers.Adam(learning_rate = alpha)
    elif opt == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate = alpha)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate = alpha)
    model.compile(optimizer = optimizer, loss = "mean_absolute_error")
    
    # fit the model
    # early stopping
    early_stopping = EarlyStopping(monitor = "val_loss", patience = 10, verbose = 0, restore_best_weights = True)

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    history = model.fit(X_train, y_train, epochs = 1000, batch_size = 32, 
                        validation_data = (X_val, y_val), verbose = 0, callbacks = [early_stopping])

    # make predictions
    pred = model.predict(X_val)

    # loss
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    # store the loss
    t_loss_mlp.append(loss)
    v_loss_mlp.append(val_loss)

label = "updrs_1"
train_df, val_df = split_dataset(data_processing(label, 200)) # choose feature = 200
y_train = train_df[label]
y_val = val_df[label]
X_train = train_df.drop(label, axis = 1)
X_val = val_df.drop(label, axis = 1)
MLP_plot(label, X_train, X_val, y_train, y_val, "RMSprop", 0.01)

label = "updrs_2"
train_df, val_df = split_dataset(data_processing(label, 200)) # choose feature = 200
y_train = train_df[label]
y_val = val_df[label]
X_train = train_df.drop(label, axis = 1)
X_val = val_df.drop(label, axis = 1)
MLP_plot(label, X_train, X_val, y_train, y_val, "RMSprop", 0.05)

label = "updrs_3"
train_df, val_df = split_dataset(data_processing(label, 200)) # choose feature = 200
y_train = train_df[label]
y_val = val_df[label]
X_train = train_df.drop(label, axis = 1)
X_val = val_df.drop(label, axis = 1)
MLP_plot(label, X_train, X_val, y_train, y_val, "adam", 0.005)

label = "updrs_4"
train_df, val_df = split_dataset(data_processing(label, 200)) # choose feature = 200
y_train = train_df[label]
y_val = val_df[label]
X_train = train_df.drop(label, axis = 1)
X_val = val_df.drop(label, axis = 1)
MLP_plot(label, X_train, X_val, y_train, y_val, "adam", 0.0001)

# plot loss vs epochs
plt.figure(figsize = (10, 4))

plt.subplot(1, 2, 1)
for i in range(4):
    plt.plot(range(1, len(t_loss_mlp[i]) + 1), t_loss_mlp[i], label = f"updrs_{i+1}")
plt.xlabel("Number of Epochs")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Training Error vs. Epochs")
plt.legend()
plt.grid(True)

# Plot validation error (MSE) vs. epochs
plt.subplot(1, 2, 2)
for j in range(4):
    plt.plot(range(1, len(v_loss_mlp[j]) + 1), v_loss_mlp[j], label = f"updrs_{j+1}")
plt.xlabel("Number of Epochs")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Validation Error vs. Epochs")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("-------")

'''
for deep learning, the best model for 4 labels are:
label1: SGD 0.05 
label2: SGD 0.05
label3: SGD 0.01
label4: RMSprop 0.005
'''
# store the training loss and validation loss
t_loss_deep = {}
v_loss_deep = {}

# function to plot MSE vs epochs and fitted plot
def deep_learning_plot(label, X_train, X_val, y_train, y_val, opt, alpha):
    model = keras.Sequential([
    keras.layers.Dense(64, activation = "relu", input_shape = (X_train.shape[1], )),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dense(256, activation = "relu"),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dense(64, activation = "relu"),
    keras.layers.Dense(32, activation = "relu"),
    keras.layers.Dense(1)
    ])

    # optimizer
    if opt == "adam":
        optimizer = keras.optimizers.Adam(learning_rate = alpha)
    elif opt == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate = alpha)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate = alpha)
    model.compile(optimizer = optimizer, loss = "mean_absolute_error")
    
    # fit the model
    # early stopping
    early_stopping = EarlyStopping(monitor = "val_loss", patience = 10, verbose = 0, restore_best_weights = True)

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    history = model.fit(X_train, y_train, epochs = 1000, batch_size = 32, 
                        validation_data = (X_val, y_val), verbose = 0, callbacks = [early_stopping])

    # loss
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    # store the loss
    t_loss_deep[label + "_" + opt + "_" + str(alpha)] = loss
    v_loss_deep[label + "_" + opt + "_" + str(alpha)] = val_loss

def run_model_deep2(l):
    for optimizer in ["adam", "SGD", "RMSprop"]:
        for alpha in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
            deep_learning_plot(label, X_train, X_val, y_train, y_val, optimizer, alpha)

label = "updrs_1"
train_df, val_df = split_dataset(data_processing(label, 200)) # choose feature = 200
y_train = train_df[label]
y_val = val_df[label]
X_train = train_df.drop(label, axis = 1)
X_val = val_df.drop(label, axis = 1)
run_model_deep2(label)

label = "updrs_2"
train_df, val_df = split_dataset(data_processing(label, 200)) # choose feature = 200
y_train = train_df[label]
y_val = val_df[label]
X_train = train_df.drop(label, axis = 1)
X_val = val_df.drop(label, axis = 1)
run_model_deep2(label)

label = "updrs_3"
train_df, val_df = split_dataset(data_processing(label, 200)) # choose feature = 200
y_train = train_df[label]
y_val = val_df[label]
X_train = train_df.drop(label, axis = 1)
X_val = val_df.drop(label, axis = 1)
run_model_deep2(label)

label = "updrs_4"
train_df, val_df = split_dataset(data_processing(label, 200)) # choose feature = 200
y_train = train_df[label]
y_val = val_df[label]
X_train = train_df.drop(label, axis = 1)
X_val = val_df.drop(label, axis = 1)
run_model_deep2(label)

# plot loss vs epochs
plt.figure(figsize = (10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(t_loss_deep["updrs_1_SGD_0.05"]) + 1), t_loss_deep["updrs_1_SGD_0.05"], label = "updrs_1")
plt.plot(range(1, len(t_loss_deep["updrs_2_SGD_0.05"]) + 1), t_loss_deep["updrs_2_SGD_0.05"], label = "updrs_2")
plt.plot(range(1, len(t_loss_deep["updrs_3_SGD_0.01"]) + 1), t_loss_deep["updrs_3_SGD_0.01"], label = "updrs_3")
plt.plot(range(1, len(t_loss_deep["updrs_4_RMSprop_0.005"]) + 1), t_loss_deep["updrs_4_RMSprop_0.005"], label = "updrs_4")
plt.xlabel("Number of Epochs")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Training Error vs. Epochs")
plt.legend()
plt.grid(True)

# Plot validation error (MSE) vs. epochs
plt.subplot(1, 2, 2)
plt.plot(range(1, len(v_loss_deep["updrs_1_SGD_0.05"]) + 1), v_loss_deep["updrs_1_SGD_0.05"], label = "updrs_1")
plt.plot(range(1, len(v_loss_deep["updrs_2_SGD_0.05"]) + 1), v_loss_deep["updrs_2_SGD_0.05"], label = "updrs_2")
plt.plot(range(1, len(v_loss_deep["updrs_3_SGD_0.01"]) + 1), v_loss_deep["updrs_3_SGD_0.01"], label = "updrs_3")
plt.plot(range(1, len(v_loss_deep["updrs_4_RMSprop_0.005"]) + 1), v_loss_deep["updrs_4_RMSprop_0.005"], label = "updrs_4")
plt.xlabel("Number of Epochs")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Validation Error vs. Epochs")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()