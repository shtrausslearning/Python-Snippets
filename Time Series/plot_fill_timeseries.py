import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime

# function to set background color for a
# specified variable and a specified level

fig = px.line(df, x='dates', y=df.columns[1:])
fig.update_xaxes(showgrid=True, gridwidth=1)
fig.update_yaxes(showgrid=True, gridwidth=1)

def filler(fig, variable, level, mode, fillcolor, layer,size=[400,1000],title=None):
    
    # Define Logical Functons to indicate above or below specific value for variable
    if mode == 'above':
        fill_id = df[variable].gt(level)
    if mode == 'below':
        fill_id = df[variable].lt(level)

    ldf = df[fill_id].groupby((~fill_id).cumsum())['dates'].agg(['first','last'])

    # Define Shape for fill regions
    for index, row in ldf.iterrows():
        # print(row['first'], row['last'])
        fig.add_shape(type="rect",xref="x", yref="paper",
                      x0=row['first'],x1=row['last'],
                      y0=0,y1=1,
                      line=dict(color="rgba(0,0,0,0)",width=3),
                      fillcolor=fillcolor,opacity=0.2,layer=layer) 
    
    # Update Aesthetics
    fig.update_layout(height=size[0],width=size[1],template='plotly_white',title=title,
                          margin=dict(l=50,r=80,t=50,b=40))
    fig.update_xaxes(showgrid=True, gridwidth=1)
    fig.update_yaxes(showgrid=True, gridwidth=1)
    return(fig)

# Example Usage

# data sample
nperiods = 200
np.random.seed(123)
df = pd.DataFrame(np.random.randint(-10, 12, size=(nperiods, 4)),
                  columns=list('ABCD'))
datelist = pd.date_range(datetime.datetime(2020, 1, 1).strftime('%Y-%m-%d'),periods=nperiods).tolist()
df['dates'] = datelist 
df = df.set_index(['dates'])
df.index = pd.to_datetime(df.index)
df.iloc[0] = 0
df = df.cumsum().reset_index()

# Fill Example
fig = filler(fig = fig, variable = 'A', level = 100, mode = 'above',
               fillcolor = 'blue', layer = 'below')

fig.show()
