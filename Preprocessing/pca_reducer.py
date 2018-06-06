import pandas as pd

def pca_reduce(data_frame):


    # Filter as many relevant features as possible

    # Apply pca to all features

    # Apply neural network based on first a set of principal components

    # Contrast result with normal neural network



    print("===== Start reducing features ======")
    data_frame = data_frame.rename(index=str, columns={"IQS 1 avg": "brix", "IQS 2 avg": "granulation",
                                       "IQS 3 avg": "seed_counts", "IQS 4 avg": "frost_damage"})

    # aggregate overall roundness and roundness

    data_frame['Roundness'].update(data_frame['Overall Roundness'])
    data_frame['Yellow'].update(data_frame['Yellow/Green'])


    df_feature = data_frame[['2nd Color Variation (Dark Yellow +)', 'Blem', 'Calyx size', 'Curvature',
                             'Elongation', 'Fruit', 'brix', 'granulation', 'Orange', 'seed_counts',
                             'Stem Size', 'Vertical Diameter Stability', # Fill with mean

                             'Creases', 'Dark Blem', 'Dark Green', 'Dark Orange', 'Dark Scar', 'Dark Yellow',
                             'Green', 'frost_damage', 'Lemon (Dark Yellow +)', 'Light Green', 'Light Scar', 'Light Yellow',
                             'Orange/Yellow', 'Red', 'Red/Orange', 'Rough skin', 'Ridges', 'Scar',
                             'Stem', 'Stem Area', 'Stem Rot', 'Touching Factor', # Fill with zero

                             'Stem angle', 'Trapeziod Angle (Deg)',  # Fill with 180

                             'DownWeight', 'Flatness', 'Fruit Center X (mm)', 'Fruit Center Y (mm)', 'Grade',
                             'LR_Diff', 'Major Diameter (mm)', 'MajorDiameter', 'Minor Diameter (mm)', 'MinorDiameter',
                             'SizerSize', 'Smoothness', 'Start of Batch',
                             'Texture', 'UpWeight', 'VisionGrade', 'Volume', 'Weight', 'Roundness', 'Yellow'  # No Filling
                             ]]

    df_target = data_frame[['block_Num']]

    # Fill with mean
    df_feature = df_feature.fillna(df_feature.mean()['2nd Color Variation (Dark Yellow +)':'Vertical Diameter Stability'])

    # Fill with zero
    df_feature.loc[:, 'Creases': 'Touching Factor'] = df_feature.loc[:, 'Creases': 'Touching Factor'].fillna(0)

    # Fill with 180
    df_feature.loc[:, 'Stem angle': 'Trapeziod Angle (Deg)'] = df_feature.loc[:, 'Stem angle': 'Trapeziod Angle (Deg)'].fillna(180)

    # Convert Grade str to integer : A -> 1, B -> 2
    df_feature.Grade = [ord(x) - 64 for x in df_feature.Grade]


    # Reuse the index of df_feature_filtered
    df_feature_filtered = pd.concat([df_feature, df_target], axis=1)

    print("===== Finish reducing features ======")

    print('===== Dimensionality of whole dataset =====')
    print(df_feature_filtered.shape)

    return df_feature_filtered
