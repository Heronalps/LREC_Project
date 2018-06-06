import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Preprocessing.pca_reducer import *
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def train():
    print("===== Loading data ======")
    df = pd.read_csv("./temp2.csv")
    df_target = df[['block_Num']]
    df = pca_reduce(df)
    pca = PCA(n_components=3, svd_solver='auto')
    df_feature = df.iloc[:, 0:-1]

    pca.fit(df_feature.values)
    print("PCA Explained Variance Ratio : ", pca.explained_variance_ratio_)
    print("PCA Singular Values : ", pca.singular_values_)
    first_pc = pca.components_[0]
    second_pc = pca.components_[1]
    third_pc = pca.components_[2]
    print('====First Principal Component==== ')
    print(first_pc)

    transform_df = pd.DataFrame(pca.transform(df_feature), columns=['pc_1', 'pc_2', 'pc_3'])
    transform_df = pd.concat([transform_df, df_target], axis=1)
    print(transform_df.head)
    # print(type(transform_df))
    # print(transform_df.index)

    # Scamble and subset data frame into train + validation(80%) and test(10%)
    transform_df = transform_df.sample(frac=1).reset_index(drop=True)
    split_ratio = 0.8
    df_train = transform_df[: int(len(transform_df) * split_ratio)]
    df_test = transform_df[int(len(transform_df) * split_ratio):]

    print("===== Feeding first 3 Principal Components to Neural Network =====")

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
    train()