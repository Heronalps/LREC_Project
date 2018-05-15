import argparse
import pandas as pd
from Preprocessing import read_file
from Preprocessing import dim_reduce



def main():
    directory = ["Pro_41", "Pro_43", "Pro_98_80", "Pro_98_81W", "Pro_112_Clem_12-11-17", "Pro_112_Daisy_1-4-18", "Pro_112_Daisy_12-13-17"]
    blocks = ["41N", "92E", "80", "81W", "64E", "64E", "64E"]
    frames = []
    for i in range(len(directory)) :
        print(directory[i])
        temp_df = read_file.read("./dataset/" + directory[i], blocks[i])
        frames.append(temp_df)
    # Aggregate dfs into one giant df
    df = pd.concat(frames)
    print(df.shape)
    df
    #df.to_csv("temp.csv", index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main()