import pandas as pd
from Preprocessing.read_file import read
# This function constructs spreadsheet of preprocessed dataset.
# Parameter: Hashmap (directory : block)

def makefile(path, directory, filename):
    frames = []
    for dir, block in directory.items():
        print(dir)
        temp_df = read(path + dir, block)
        frames.append(temp_df)

    # Aggregate dfs into one giant df
    df = pd.concat(frames, sort=False)

    df.to_csv(filename + '.csv', index = False)