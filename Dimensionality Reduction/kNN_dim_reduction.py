from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import FastICA
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# function to plot a two PCA Feature Plot using Pandas 
def scatterPlot(xDF, yDF, algoName):
    
    sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    tempDF = pd.DataFrame(data=xDF.loc[:,0:1], index=xDF.index)
    tempDF = pd.concat((tempDF,yDF), axis=1, join="inner")
    tempDF.columns = ["Component 1","Component 2","Label"]
    g = sns.scatterplot(x="Component 1",y="Component 2",data=tempDF,hue="Label",
                        linewidth=0.5,alpha=0.5,s=15,edgecolor='k')
    plt.title(algoName);plt.legend()
    
    for i in ['top', 'right', 'bottom', 'left']:
        ax.spines[i].set_color('black')
    
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False);ax.spines['left'].set_visible(False)
    ax.grid(axis = 'both',ls='--',alpha = 0.9)
    plt.show()

# DataFrame input Dimensionality Reduction Function
def dimRed(ldf,feature='target',n_comp=5,plot_id=False,model_id='pca'):
    
    # Given a dataframe, split feature/target variable
    X = ldf.copy()
    y = ldf[feature].copy()
    del X[feature]
    
    n_jobs = -1; rs = 32
    
    if(model_id is 'pca'):
        whiten = False
        model = PCA(n_components=n_comp,whiten=whiten,random_state=rs)
    if(model_id is 'sparsepca'):
        alpha = 1
        model = SparsePCA(n_components=n_comp,alpha=alpha,random_state=rs,n_jobs=n_jobs)
    elif(model_id is 'kernelpca'):
        kernel = 'rbf'; gamma = None
        model = KernelPCA(n_components=n_comp,kernel=kernel,gamma=gamma,n_jobs=n_jobs,random_state=rs)
    elif(model_id is 'incrementalpca'):
        batch_size = None
        model = IncrementalPCA(n_components=n_comp,batch_size=batch_size)
    elif(model_id is 'truncatedsvd'): 
        algorithm = 'randomized';n_iter = 5
        model = TruncatedSVD(n_components=n_comp,algorithm=algorithm,n_iter=n_iter,random_state=rs)
    elif(model_id is 'gaussianrandomprojection'):
        eps = 0.5
        model = GaussianRandomProjection(n_components=n_comp,eps=eps,random_state=rs)
    elif(model_id is 'sparserandomprojection'):
        density = 'auto'; eps = 0.5; dense_output = True
        model = SparseRandomProjection(n_components=n_comp,density=density, 
                                       eps=eps, dense_output=dense_output,random_state=rs)

    # Manifold Approaches
    if(model_id is 'isomap'):
        n_neigh = 2
        model = Isomap(n_neighbors=n_neigh,n_components=n_comp, n_jobs=n_jobs)    
    elif(model_id is 'mds'):
        n_init = 1; max_iter = 50; metric = False
        model = MDS(n_components=n_comp,n_init=n_init,max_iter=max_iter,metric=True,
                    n_jobs=n_jobs, random_state=rs)
    elif(model_id is 'locallylinearembedding'):
        n_neigh = 10; method = 'modified'
        model = LocallyLinearEmbedding(n_neighbors=n_neigh,n_components=n_comp, method=method, \
                                    random_state=rs, n_jobs=n_jobs)
    elif(model_id is 'tsne'):
        learning_rate = 300; perplexity = 30; early_exaggeration = 12; init = 'random'
        model = TSNE(n_components=n_comp, learning_rate=learning_rate, \
                    perplexity=perplexity, early_exaggeration=early_exaggeration, \
                    init=init, random_state=rs)
    elif(model_id is 'minibatchdictionarylearning'):
        alpha = 1; batch_size = 200; n_iter = 25
        model = MiniBatchDictionaryLearning(n_components=n_comp,alpha=alpha,
                                            batch_size=batch_size,n_iter=n_iter,random_state=rs)
    elif(model_id is 'fastica'):
        algorithm = 'parallel'; whiten = True; max_iter = 100
        model = FastICA(n_components=n_comp, algorithm=algorithm,whiten=whiten, 
                          max_iter=max_iter, random_state=rs)
    
    # Unsupervised Dimension Reduction 
    X_red = model.fit_transform(X)
    X_red = pd.DataFrame(data=X_red, index=X.index)
    if(plot_id):
         scatterPlot(X_red, y,model_id)
    X_red[feature] = y
    
    return X_red # return new feature matrix

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

dimRed(ldf=df_diab,feature='target',n_comp=3,plot_id=True)
