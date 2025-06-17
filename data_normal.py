import pandas as pd

df = pd.read_csv('data/new_clinic18_YT.csv')

df_selected = df.iloc[1:, 1:]
normalized_df = df_selected.apply(lambda x: 7000 * x / x.sum())

df.iloc[1:, 1:] = normalized_df

df.to_csv('data/new_clinic18_YT_normal.csv', index=False)
