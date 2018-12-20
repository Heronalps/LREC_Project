import argparse
import numpy as np
from Preprocessing.reducer import *
from Preprocessing.read_file import read
from Preprocessing.preprocess import makefile
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

FILENAME = "data_2018"
PATH = "./dataset/2018/"

directory_2017 = {
    "Pro_41" : "41N",
    "Pro_43" : "92E",
    "Pro_98_80" : "80",
    "Pro_98_81W" : "81W",
    "Pro_112_Clem_12-11-17" : "64E",
    "Pro_112_Daisy_1-4-18" : "64E",
    "Pro_112_Daisy_12-13-17" : "64E"
}

directory_2018 = {
    # "Pro_143_field_81C_8-19-18" : "81C",
    "Pro_98_field_80_9-18-18" : "80"
}

# This script train the NN model on 2017 data and test on 2018 data

def test():
    # makefile(PATH, directory_2018, FILENAME)
    
    df_2017 = pd.read_csv("./data_2017.csv")
    df_2018 = pd.read_csv("./data_2018.csv")

    # Dimensionality Reduction based on correlation analysis
    df_2017 = reduce(df_2017, 2017)
    df_2018 = reduce(df_2018, 2018)

    # Since the columns in 2017 dataset are way more than 2018.
    # Filter them out in the 2017 training dataset.
    df_2017 = df_2017.filter(items = df_2018.columns)
    
    # Cast block num from int64 to str object
    df_2018.loc[:, 'block_Num'] = df_2018.loc[:, 'block_Num'].astype(str)

    kf = KFold(n_splits=10)
    solver = MLPClassifier(activation='relu',
                           solver='adam',
                           alpha=1e-5,
                           hidden_layer_sizes=(100, 2),
                           random_state=1,
                           verbose=True)

    # convert dataframe to ndarray, since kf.split returns nparray as index
    feature_train = df_2017.iloc[:, 0: -1].values
    target_train = df_2017.iloc[:, -1].values
    feature_test = df_2018.iloc[:, 0: -1].values
    target_test = df_2018.iloc[:, -1].values

    for train_indices, test_indices in kf.split(feature_train, target_train):
        solver.fit(feature_train[train_indices], target_train[train_indices])
        print(solver.score(feature_train[test_indices], target_train[test_indices]))

    y_pred = solver.predict(feature_test)
    print("Accuracy Score : " + str(accuracy_score(y_pred, target_test)))
    print(classification_report(target_test, y_pred))

    unique, count = np.unique(target_test, return_counts=True)
    print("Test dataset Distribution")
    print(np.asarray((unique, count)).T)

    unique, count = np.unique(y_pred, return_counts=True)
    print("Prediction dataset Distribution")
    print(np.asarray((unique, count)).T)


def train():
    # makefile(PATH, directory_2018, FILENAME)
    
    df_2017 = pd.read_csv("./data_2017.csv")
    df_2018 = pd.read_csv("./data_2018.csv")

    # Dimensionality Reduction based on correlation analysis
    df = reduce(df_2017, 2017)
    df_2018 = reduce(df_2018, 2018)
    df = df.filter(items = df_2018.columns)

    # Scamble and subset data frame into train + validation(80%) and test(10%)
    df = df.sample(frac=1).reset_index(drop=True)
    split_ratio = 0.8
    print('Train and Test Split Ratio : ', split_ratio)
    df_train = df[ : int(len(df) * split_ratio)]
    df_test = df[int(len(df) * split_ratio) : ]

    kf = KFold(n_splits=10)
    solver = MLPClassifier(activation='relu',
                           solver='adam',
                           alpha=1e-5,
                           hidden_layer_sizes=(100, 2),
                           random_state=1,
                           verbose=True)

    # convert dataframe to ndarray, since kf.split returns nparray as index
    feature_train = df_train.iloc[:, 0: -1].values
    target_train = df_train.iloc[:, -1].values
    feature_test = df_test.iloc[:, 0: -1].values
    target_test = df_test.iloc[:, -1].values

    # Train NN with train dataset
    # The classification is random and could be skewed, because the training set is sampled.
    # Generally, it is down to 0.996 accuracy and ~1.0 f1-score for 5 fields.
    
    for train_indices, test_indices in kf.split(feature_train, target_train):
        solver.fit(feature_train[train_indices], target_train[train_indices])
        print(solver.score(feature_train[test_indices], target_train[test_indices]))

    y_pred = solver.predict(feature_test)
    print("Accuracy Score : " + str(accuracy_score(y_pred, target_test)))
    print(classification_report(target_test, y_pred))

    unique, count = np.unique(target_test, return_counts=True)
    print("Test dataset Distribution")
    print(np.asarray((unique, count)).T)

    unique, count = np.unique(y_pred, return_counts=True)
    print("Prediction dataset Distribution")
    print(np.asarray((unique, count)).T)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    train()
    # test()
    # makefile(PATH, directory_2018, FILENAME)