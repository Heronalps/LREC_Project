import argparse
import numpy as np
import pandas as pd
from Preprocessing import read_file
from Preprocessing import dim_reduce
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def main():
    # directory = ["Pro_41", "Pro_43", "Pro_98_80", "Pro_98_81W", "Pro_112_Clem_12-11-17", "Pro_112_Daisy_1-4-18", "Pro_112_Daisy_12-13-17"]
    # blocks = ["41N", "92E", "80", "81W", "64E", "64E", "64E"]
    # frames = []
    # for i in range(len(directory)) :
    #     print(directory[i])
    #     temp_df = read_file.read("./dataset/" + directory[i], blocks[i])
    #     frames.append(temp_df)
    #
    # # Aggregate dfs into one giant df
    # df = pd.concat(frames, sort=False)

    #df.to_csv("temp2.csv", index = False)
    df = pd.read_csv("./temp2.csv")

    # Dimensionality Reduction based on correlation analysis
    df = dim_reduce.reduce(df)


    # Scamble and subset data frame into train + validation(80%) and test(10%)
    df = df.sample(frac=1).reset_index(drop=True)
    split_ratio = 0.8
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
    main()