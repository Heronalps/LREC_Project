import pandas as pd

'''
This function truncates the data frame to remain meaningful features
-------------------
Parameters: data frame, year
Returns: truncated data frame
'''

def truncate(data_frame):
    print("===== Start truncating features ======")
    data_frame = data_frame.rename(index=str, columns={"IQS 1 avg": "brix", "IQS 2 avg": "granulation",
                                    "IQS 3 avg": "seed_counts", "IQS 4 avg": "frost_damage"})

    data_frame = data_frame.drop([
        'EventTime', 'Cup', 'Lane', 'RodPulse', 'RodIndex', 'BatchId', 'QualityName'
    ], axis=1)
    data_frame = data_frame.fillna(data_frame.mean()[:])
    
    # Convert Grade str to integer : A -> 1, B -> 2
    data_frame.Grade = [ord(x) - 64 for x in data_frame.Grade]

    return data_frame

'''
This function reduce the positive and negative correlated features
---------
Parameter: data frame, year
Returns: Filtered data frame
'''

def reduce(df_feature, year):
    print("===== Start reducing features ======")
    # df_feature = truncate(data_frame, year)
    df_target = df_feature[['block_Num']]
    df_feature = df_feature.drop('block_Num', axis=1)
    # import pdb; pdb.set_trace();

    correlation_matrix = df_feature.astype('float64').corr()
    for col in correlation_matrix.columns:
        if correlation_matrix[col].between(0.8, 1.0, inclusive=False).any() or \
                correlation_matrix[col].between(-1.0, -0.8).any():
                
            # Drop row and column in the correlation matrix
            correlation_matrix = correlation_matrix.drop(col)
            correlation_matrix = correlation_matrix.drop(col, axis=1)


    df_feature_filtered = df_feature[correlation_matrix.columns]

    # Reuse the index of df_feature_filtered
    df_feature_filtered = pd.concat([df_feature_filtered, df_target], axis=1)

    print("===== Finish reducing features ======")

    return df_feature_filtered
