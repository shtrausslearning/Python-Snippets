import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Plot Correlation Matrix
def corrmat(df,id=False,figsize=(10,10)):
    
    corr_mat = df.corr().round(2)
    f, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_mat, dtype=np.bool))
    mask = mask[1:,:-1]
    corr = corr_mat.iloc[1:,:-1].copy()
    sns.heatmap(corr,mask=mask,vmin=-0.5,vmax=0.5,center=0, 
                cmap='YlGnBu',square=False,lw=2,annot=True,cbar=False)

# Plot Correlation to Target Variable only
def corrmat2(df,target='demand',figsize=(10,1),ret_id=False):
    
    corr_mat = df.corr().round(2);shape = corr_mat.shape[0]
    corr_mat = corr_mat.transpose()
    corr = corr_mat.loc[:, df.columns == target].transpose().copy()
    
    if(ret_id is False):
        f, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr,vmin=-0.5,vmax=0.5,center=0, 
                     cmap='YlGnBu',square=False,lw=2,annot=True,cbar=False)
    
    if(ret_id):
        return corr
      
''' EXAMPLE '''
from sklearn import datasets
import pandas as pd

# convert sklearn dataset to pandas DataFrame
def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

df_diab = sklearn_to_df(datasets.load_diabetes())

# collation heatmap
corrmat(df_diab)
corrmat2(df_diab,target='target')
