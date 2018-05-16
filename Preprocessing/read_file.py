import pandas as pd
from pathlib import Path

#directory = "./dataset/Pro_41"

def read(directory, block):
    pathlist = Path(directory).glob('*')
    frames = []
    print("=====Start reading files in " + directory + " ========")
    for path in pathlist :
        # print(str(path))
        # Skip batch info rows
        if str(path).endswith('csv') :
            dataFrame = pd.read_csv(str(path), skiprows = 5)
        elif str(path).endswith('xlsx'):
            dataFrame = pd.read_excel(str(path), skiprows = 5)


        # Drop bottom no Grader rows
        dataFrame = dataFrame.dropna()

        # Drop duplicate columns
        dataFrame = dataFrame.loc[:, ~dataFrame.columns.duplicated()]

        # Assign Block # to DataFrame
        dataFrame = dataFrame.assign(block_Num = block)

        # Append to the list of DataFrame
        frames.append(dataFrame)
    print("=====Finish reading files=======")
    return pd.concat(frames, sort=False)





