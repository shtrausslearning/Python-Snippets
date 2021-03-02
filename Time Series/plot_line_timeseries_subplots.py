import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime

# Subplot Plotly Timeseries Line Plots (Plot n verticle)
# Input x axis -> date_time # lst-> column plots, # nplots -> total number of subplots,
#       lw_id -> lineplot thickness list

def plot_vsubplots(ldf,lst,title='',nplots=None,lw_id=None,size=[500,1000]):
        
    assert(nplots is not None) 
    fig = make_subplots(rows=nplots,shared_xaxes=True)
    ii=-1
    for i in lst:
        ii+=1
        fig.add_trace(go.Scatter(x=ldf.index,y=ldf[lst[ii]], mode='lines',
                                 name=lst[ii],line=dict(width=lw_id[ii])), row=ii+1, col=1) 

    # Plot Aesthetics
    fig.update_layout(height=size[0],width=size[1],template='plotly_white',title=title,
                      margin=dict(l=50,r=80,t=50,b=40));fig.show()

# data sample
nperiods = 200
np.random.seed(123)
df = pd.DataFrame(np.random.randint(-10, 12, size=(nperiods, 4)),columns=list('ABCD'))
datelist = pd.date_range(datetime.datetime(2020, 1, 1).strftime('%Y-%m-%d'),periods=nperiods).tolist()
df['dates'] = datelist 
df = df.set_index(['dates'])
df.index = pd.to_datetime(df.index)
df.iloc[0] = 0
df = df.cumsum().reset_index()

# plot example 
plot_vsubplots(df,['A','B'],nplots=2,lw_id=[2,2])
