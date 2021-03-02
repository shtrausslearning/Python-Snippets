# Subplot Plotly Timeseries Line Plots (Plot n verticle)
# Input x axis -> date_time # lst-> column plots, # nplots -> total number of subplots,
#       lw_id -> lineplot thickness

def plot_vsubplots(ldf,lst,title='',nplots=None,lw_id=None,size=[400,1000]):

    # lw_id list of line widths if added
        
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
