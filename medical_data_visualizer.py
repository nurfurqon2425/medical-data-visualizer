import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = np.where((df['weight'] / ((df['height'] / 100) ** 2)) > 25, 1, 0) 

# 3
df['cholesterol'] = np.where(df['cholesterol'] > 1, 1, 0)
df['gluc'] = np.where(df['gluc'] > 1, 1, 0)
# print(df.shape[0])
# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['id'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    # print(df_cat.head())

    # 6
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    

    # 7
    df_cat = df_cat.groupby(['cardio','variable','value']).agg(total=('cardio', 'count'))
    # print(df_cat.head(24))
    

    # 8
    cp = sns.catplot(data=df_cat, x='variable', y='total', hue='value', col='cardio', kind='bar')
    
    fig = cp.figure
    # plt.show()

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df.loc[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]
    # print(df_heat.shape[0])
    # 12
    corr = df_heat.corr()
    # print(corr)

    # 13    
    mask = corr.mask(np.triu(np.ones_like(corr, dtype=bool)))
    # print(mask)


    # 14
    fig, ax = plt.subplots(figsize=(10,10))

    # 15
    sns.heatmap(data=mask, vmin= -0.08, vmax=0.24, square=True, annot=True, fmt='.1f', linewidths=.5, ax=ax)
    # plt.show()


    # 16
    fig.savefig('heatmap.png')
    return fig

# draw_cat_plot()
# draw_heat_map()