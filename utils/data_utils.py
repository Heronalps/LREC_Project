import os
import pandas as pd


BLOCK_MAP = {"Pro 41 Navel 1 -26-18": "41", "Pro 43 Fileld 92E Nova 12-4-17": "43_92E", "Pro 98 81W 11-14-17": "98_81W",
             "Pro 98 Field 81 Lemons 1-17-18": "98_81", "Pro 112 64E Daisy 12-13-17": "112_64E", "Pro 112 Clem 12-11-17": "112"}

def read_directory(directory, block):
    """
    Read .csv, .xlsx files in a directory, assign them a block number
    :param directory: folder containing LREC csv files
    :param block: each folder is a different block, so each file in a directory will have the same block number
    :return: Pandas object containing merged fruit information
    """
    panda_frames = []
    for data in os.listdir(directory):
        path = directory + "/" + data
        if data.endswith(".csv"):
            data_frame = pd.read_csv(str(path), skiprows=5)
        if data.endswith(".xlsx"):
            data_frame = pd.read_excel(str(path), skiprows=5)
        else:
            continue

        # Drop bottom no Grader rows
        data_frame = data_frame.dropna()

        # Drop duplicate columns
        data_frame = data_frame.loc[:, ~data_frame.columns.duplicated()]

        # Assign block number to dataframe
        data_frame = data_frame.assign(block_Num=block)

        panda_frames.append(data_frame)

    return pd.concat(panda_frames)


def merge(root_directory):
    """
    Merge all blocks from a directory
    :param root_directory: root folder containing folders with fruit data
    :return: panda object containing merged fruit data from all blocks
    """
    panda_frames = []

    for directory in os.listdir(root_directory):
        path = root_directory + "/" + directory
        if os.path.isdir(path):
            block_frame = read_directory(path, BLOCK_MAP[directory])
            panda_frames.append(block_frame)

    return pd.concat(panda_frames)
