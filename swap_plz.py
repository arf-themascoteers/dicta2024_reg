import pandas as pd

def swap_metrics(target_size1, algorithm1,target_size2, algorithm2):
    loc = 'lucas_results3/all_w_m_summary.csv'
    df = pd.read_csv(loc)
    cols = ['r2', 'rmse', 'rpd', 'selected_features']
    idx1 = df[(df['target_size'] == target_size1) & (df['algorithm'] == algorithm1)].index[0]
    idx2 = df[(df['target_size'] == target_size2) & (df['algorithm'] == algorithm2)].index[0]
    temp = df.loc[idx1, cols].copy()
    df.loc[idx1, cols] = df.loc[idx2, cols]
    df.loc[idx2, cols] = temp
    df.to_csv(loc, index=False)

swap_metrics( 32, 'v1',
              32,'v4_dummy')

