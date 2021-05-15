import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

def corrmat3(df,size=None):
    
    corr = df.corr().round(3)
    xcols = corr.columns.tolist();ycols = xcols
    if(size is None):
        width = 700; height = 500
    else:
        width = size[0]; height = size[1]
    
    layout = dict(
        width = width,height = height,
        yaxis= dict(tickangle=-30,side = 'left'),
        xaxis= dict(tickangle=-30,side = 'top'))
    fig = ff.create_annotated_heatmap(
        z=corr.values,x= xcols,y= ycols,
        colorscale='ylgnbu',showscale=False)
    fig['layout'].update(layout)
    fig.update_layout(margin={"r":0,"t":60,"l":0,"b":0})
    
    return py.iplot(fig)

''' EXAMPLE '''
from sklearn import datasets
import pandas as pd

# convert sklearn dataset to pandas DataFrame
def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

df_diab = sklearn_to_df(datasets.load_diabetes())

# Correlaion Heatmap
corrmat3(df_diab,size=(650,650))
