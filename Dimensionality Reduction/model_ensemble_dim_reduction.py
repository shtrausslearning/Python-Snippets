# Simple Model Imputation Ensemble (XGB+kNN)

from sklearn.neighbors import KNeighborsRegressor

# function that imputes a dataframe 
def impute_model(df,cols=None):

    # separate dataframe into numerical/categorical
    ldf = df.select_dtypes(include=[np.number])           # select numerical columns in df
    ldf_putaside = df.select_dtypes(exclude=[np.number])  # select categorical columns in df
    # define columns w/ and w/o missing data
    cols_nan = ldf.columns[ldf.isna().any()].tolist()         # list of features w/ missing data 
    cols_no_nan = ldf.columns.difference(cols_nan).values     # get all colun data w/o missing data
    
    if(cols is not None):
        cols_nan = cols
        df1 = ldf[cols_nan].describe()
    
    fill_id = -1
    for col in cols_nan:    
        fill_id+=1
        imp_test = ldf[ldf[col].isna()]   # indicies which have missing data will become our test set
        imp_train = ldf.dropna()          # all indicies which which have no missing data 
        model0 = GBoost(n_estimators=10,tree_id='xgb_bagging')  # XGB Bagging Model 
        model1 = KNeighborsRegressor(n_neighbors=15)            # KNR Unsupervised Approach
        knr = model0.fit(imp_train[cols_no_nan], imp_train[col])
        xgb = model1.fit(imp_train[cols_no_nan], imp_train[col])
        knrP = knr.predict(imp_test[cols_no_nan])
        xgbP = xgb.predict(imp_test[cols_no_nan])
        pred = (knrP + xgbP)*0.5 # Simple Model Ensemble
        ldf.loc[df[col].isna(), col] = pred
        ldf.loc[df[col].isna(),'fill_id'] = fill_id
        
    df2 = ldf[cols_nan].describe()
    pd_html([df1,df2],['before imputation','after imputation'])
        
    return pd.concat([ldf,ldf_putaside],axis=1)
