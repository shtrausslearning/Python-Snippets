from sklearn.base import BaseEstimator, TransformerMixin
import shap
import plotly.express as px
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.feature_selection import SelectKBest,f_regression
from xgboost import plot_importance,XGBClassifier,XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

line_colors = ["#7CEA9C", '#50B2C0', "rgb(114, 78, 145)", "hsv(348, 66%, 90%)", "hsl(45, 93%, 58%)"]

class transformer(BaseEstimator,TransformerMixin):
    
    def __init__(self,drop_nan=False,select_dtype=False,show_nan=False,title='Title',show_counts=False,
                 figsize=(None,None), feature_importance = False, target = 'PRICE'):
        self.drop_nan = drop_nan
        self.select_dtype = select_dtype
        self.show_nan = show_nan
        self.title = title
        self.figsize = figsize
        self.feature_importance = feature_importance
        self.target = target  # target variable
        
    # Apply Fit
    def fit(self,X,y=None):
        return self
        
    # Apply Some Transformation to the Feature Matrix
    def transform(self,X):
        
        ''' Drop All NAN values in DataFrame'''
        if(self.drop_nan):
            X = X.dropna();
            return X
            
        ''' Split DataFrame into Numerical/Object features'''
        if(self.select_dtype):
            X1 = X.select_dtypes(include=['float64','int64','uint8'])     # return only numerical features from df
            X2 = X.select_dtypes(exclude=['float64','int64','uint8'])
            return X1,X2
        
        ''' Plot Feature Importance '''
        if(self.feature_importance):
            
             # Plot Correlation to Target Variable only
            def corrMat2(df,target=self.target,figsize=(9,0.5),ret_id=False):

                corr_mat = df.corr().round(2);shape = corr_mat.shape[0]
                corr_mat = corr_mat.transpose()
                corr = corr_mat.loc[:, df.columns == self.target].transpose().copy()

                if(ret_id is False):
                    f, ax = plt.subplots(figsize=figsize)
                    sns.heatmap(corr,vmin=-0.3,vmax=0.3,center=0, 
                                cmap=cmap,square=False,lw=2,annot=True,cbar=False)
                    plt.title(f'Feature Correlation to {self.target}')

                if(ret_id):
                    return corr

            ''' Plot Relative Feature Importance '''
            def feature_importance(tldf,feature=self.target,n_est=500):

                # X : Numerical / Object DataFrame
                ldf0,_ = transformer(select_dtype=True).transform(X=tldf)
                ldf = transformer(drop_nan=True).transform(X=ldf0)  

                # Input dataframe containing feature & target variable
                X = ldf.copy()
                y = ldf[feature].copy()
                del X[feature]

            #   CORRELATION
                imp = corrmat2(ldf,feature,figsize=(15,0.5),ret_id=True)
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
                fig.update_layout(template='plotly_white',height=self.figsize[1],width=self.figsize[0],margin={"r":0,"t":60,"l":0,"b":0});
                for data in fig.data:
                    data["width"] = 0.6 #Change this value for bar widths
                fig.write_html("/kaggle/working/feature_importance.html") # output html format
                fig.show()
                
            feature_importance(X)
           
''' EXAMPLE '''
from sklearn import datasets
import pandas as pd

# convert sklearn dataset to pandas DataFrame
def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

df_diab = sklearn_to_df(datasets.load_diabetes())

# Feature Importance
transformer(feature_importance=True,figsize=(800,400),target='target').transform(X=df_diab)
