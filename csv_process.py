import pandas as pd
import numpy as np
from numpy import NAN

df = pd.read_excel("./dataset/Pro_41/Merged_Fruit_logs/batch01526_merged.csv.xlsx", header = None, index_col = False)
df.head()
print(df)
