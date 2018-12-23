import pandas as pd
import numpy as np
import os, boto3
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor

local_repo = os.path.join(os.path.sep, "tmp", os.path.basename('csv'))
client = boto3.client('s3')

def read_csv_s3(file_name):
    if not os.path.exists(local_repo):
        os.makedirs(local_repo)
    path = local_repo + '/' + file_name
    bucket = 'lrec-datasets'
    client.download_file(bucket, file_name, path)
    df = pd.read_csv(path)
    return df

def lambda_handler(event, context):
    file_name = event['file_name']
    df = read_csv_s3(file_name)

    y = df.block_Num
    X = df.drop(['block_Num'], axis=1).select_dtypes(exclude=['object'])

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

    # This Imputer uses the default strategy as mean
    my_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    train_X = my_imputer.fit_transform(train_X)
    test_X = my_imputer.transform(test_X)

    # y ndarray must be transformed from object to int64
    # otherwise, DMatrix in XGBoost is not able to be created

    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)

    my_model = XGBRegressor()
    my_model.fit(train_X, train_y, verbose=False)

    predictions = my_model.predict(test_X)

    from sklearn.metrics import mean_absolute_error
    print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
    print("====================")
    print(np.unique(test_y, return_counts=True))
    print(np.unique(predictions.round(), return_counts=True))
    print("====================")
    print("Accuracy Score: " + str(accuracy_score(test_y, predictions.round(), normalize=True)))