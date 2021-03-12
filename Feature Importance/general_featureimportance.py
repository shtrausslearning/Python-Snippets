import plotly.express as px
import shap
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.feature_selection import SelectKBest,f_regression
from xgboost import plot_importance,XGBClassifier,XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import numpy as np
import pandas as pd

# Feature importance is one way of determining what affects the target variable.
# We can utilise various methods/libraries to help us explore the relative importance of 
# features and get a sense of how accurate the matrix correlation values are.
# Different approaches are likely to predict different top features, so let's focus on the
# less relevant features only.
# Let's use the function from the Building an Asset Trading Strategy notebook, I found it 
# was much more useful than unsupervised learning dimensionality reduction, when it comes 
# to feature importance evaluation.
  
line_colors = ["#7CEA9C", '#50B2C0', "rgb(114, 78, 145)", 
               "hsv(348, 66%, 90%)", "hsl(45, 93%, 58%)"]
            
# Plot Correlation
def corrMat2(df,feature='target',figsize=(9,0.5),ret_id=False):

    corr_mat = df.corr().round(2);shape = corr_mat.shape[0]
    corr_mat = corr_mat.transpose()
    corr = corr_mat.loc[:, df.columns == feature].transpose().copy()

    if(ret_id is False):
        f, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr,vmin=-0.3,vmax=0.3,center=0, 
                    cmap=cmap,square=False,lw=2,annot=True,cbar=False)
        plt.title(f'Feature Correlation to {feature}')

    if(ret_id):
        return corr

''' Plot Relative Feature Importance '''
def feature_importance(tldf,feature='target',n_est=500,figsize=[800,400]):

    # Select Numerical Features only & drop NaN
    ldf0 = tldf.select_dtypes(include=['float64','int64','uint8'])
    ldf = ldf0.dropna()

    # Input dataframe containing feature & target variable
    X = ldf.copy()
    y = ldf[feature].copy()
    del X[feature]

#   CORRELATION
    imp = corrMat2(ldf,feature,figsize=(15,0.5),ret_id=True)
    del imp[feature]
    s1 = imp.squeeze(axis=0);s1 = abs(s1)
    s1.name = 'Correlation'

#   SHAP
    model = CatBoostRegressor(silent=True,n_estimators=n_est).fit(X,y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_sum = np.abs(shap_values).mean(axis=0)
    s2 = pd.Series(shap_sum,index=X.columns,name='Cat_SHAP').T

#   RANDOMFOREST
    model = RandomForestRegressor(n_est,random_state=0, n_jobs=-1)
    fit = model.fit(X,y)
    rf_fi = pd.DataFrame(model.feature_importances_,index=X.columns,
                        columns=['RandForest']).sort_values('RandForest',ascending=False)
    s3 = rf_fi.T.squeeze(axis=0)

#   XGB 
    model=XGBRegressor(n_estimators=n_est,learning_rate=0.5,verbosity = 0)
    model.fit(X,y)
    data = model.feature_importances_
    s4 = pd.Series(data,index=X.columns,name='XGB').T

#   KBEST
    model = SelectKBest(k=X.shape[1], score_func=f_regression)
    fit = model.fit(X,y)
    data = fit.scores_
    s5 = pd.Series(data,index=X.columns,name='K_best')

    # Combine Scores
    df0 = pd.concat([s1,s2,s3,s4,s5],axis=1)
    df0.rename(columns={'target':'lin corr'})

    x = df0.values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled,index=df0.index,columns=df0.columns)
    df = df.rename_axis('Feature Importance via', axis=1)
    df = df.rename_axis('Feature', axis=0)
    df['total'] = df.sum(axis=1)
    df = df.sort_values(by='total',ascending=True)
    del df['total']
    fig = px.bar(df,orientation='h',barmode='stack',color_discrete_sequence=line_colors)
    fig.update_layout(template='plotly_white',height=figsize[1],width=figsize[0],margin={"r":0,"t":60,"l":0,"b":0})
    for data in fig.data:
        data["width"] = 0.6 #Change this value for bar widths
    fig.show()
                
''' Example Application '''
from sklearn import datasets

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

#df_boston = sklearn_to_df(datasets.load_boston())
#display(df_boston.head())
#df_cali = sklearn_to_df(datasets.fetch_california_housing())
#display(df_cali.head())
df_diab = sklearn_to_df(datasets.load_diabetes())
# print(df_diab.isna().sum()) # check missing data in columns

feature_importance(df_diab,feature='target')