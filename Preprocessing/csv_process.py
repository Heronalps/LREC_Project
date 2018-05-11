import pandas as pd
import numpy as np
from numpy import NAN
from pathlib import Path

#directory = "./dataset/Pro_41"
def main(directory, block):
    pathlist = Path(directory).glob('**/*.csv')
    frames = []

    for path in pathlist :

        # Skip batch info rows
        dataFrame = pd.read_csv(str(path), skiprows = 5)

        # Drop bottom no Grader rows
        dataFrame = dataFrame.dropna()

        # Drop duplicate columns
        dataFrame = dataFrame.loc[:, ~dataFrame.columns.duplicated()]

        # Append to the list of DataFrame
        frames.append(dataFrame)

    df = pd.concat(frames)
    df.to_csv("total.csv", index = False)
    #print(df.shape)


#df = pd.read_excel("./dataset/Pro_41/batch01526_merged.csv.xlsx", header = None, index_col = False)
#df.head()
#print(df)
