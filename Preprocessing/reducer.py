import pandas as pd

def truncate(data_frame, year):
    if year == 2017:
        print("===== Start reducing features ======")
        data_frame = data_frame.rename(index=str, columns={"IQS 1 avg": "brix", "IQS 2 avg": "granulation",
                                        "IQS 3 avg": "seed_counts", "IQS 4 avg": "frost_damage"})

        df_feature = data_frame[['2nd Color Variation (Dark Yellow +)', 'Elongation', 'Fruit', 'brix',
                                'granulation', 'Orange', 'Calyx size', 'seed_counts', 'Overall Roundness',
                                'Stem Size',  # Fill with mean

                                'Creases', 'Dark Orange', 'frost_damage', 'Lemon (Dark Yellow +)',
                                'Light Green', 'Orange/Yellow', 'Red', 'Red/Orange', 'Rough skin',
                                'Ridges', 'Stem', 'Stem Area', 'Stem Rot',  # Fill with zero

                                'Stem angle', 'Trapeziod Angle (Deg)',  # Fill with 180

                                'DownWeight', 'Flatness', 'Fruit Center X (mm)', 'Fruit Center Y (mm)', 'Grade',
                                'LR_Diff', 'Major Diameter (mm)', 'SizerSize', 'Smoothness', 'Start of Batch',
                                'Texture', 'UpWeight', 'VisionGrade', 'Volume', 'Weight', 'block_Num'  # No Filling
                                ]]

        # Fill with mean
        df_feature = df_feature.fillna(df_feature.mean()['2nd Color Variation (Dark Yellow +)':'Stem Size'])

        # Fill with zero
        df_feature.loc[:, 'Creases': 'Stem Rot'] = df_feature.loc[:, 'Creases': 'Stem Rot'].fillna(0)

        # Fill with 180
        df_feature.loc[:, 'Stem angle': 'Trapeziod Angle (Deg)'] = df_feature.loc[:, 'Stem angle': 'Trapeziod Angle (Deg)'].fillna(180)
    
    elif year == 2018:
        print("===== Start reducing features ======")
        df_feature = data_frame[['2nd Color Variation (Dark Yellow +)', # Fill with mean

                                'Lemon (Dark Yellow +)',
                                'Light Green', 'Stem Area',  # Fill with zero

                                'DownWeight', 'Flatness', 'Fruit Center X (mm)', 'Fruit Center Y (mm)', 'Grade',
                                'LR_Diff', 'Major Diameter (mm)', 'SizerSize', 'Smoothness', 'Start of Batch',
                                'Texture', 'UpWeight', 'VisionGrade', 'Volume', 'Weight', 'block_Num'  # No Filling
                                ]]

        # Fill with mean
        df_feature = df_feature.fillna(df_feature.mean()['2nd Color Variation (Dark Yellow +)'])

        # Fill with zero
        df_feature.loc[:, 'Lemon (Dark Yellow +)': 'Stem Area'] = df_feature.loc[:, 'Lemon (Dark Yellow +)': 'Stem Area'].fillna(0)
    
    return df_feature

def reduce(data_frame, year):
    df_feature = truncate(data_frame, year)
    df_target = df_feature[['block_Num']]
    df_feature = df_feature.drop('block_Num', axis=1)
    # import pdb; pdb.set_trace();

    # Convert Grade str to integer : A -> 1, B -> 2
    df_feature.Grade = [ord(x) - 64 for x in df_feature.Grade]

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
