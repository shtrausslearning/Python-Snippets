# Plot Multiple Timeseries plots in one figure

colours = ['tab:blue','tab:red','tab:green']
def plot_line2(ldf,lst,title=''):
    
    ii=-1
    plt.figure(figsize=(14,5))
    for i in lst:
        ii+=1
        ax = ldf[lst[ii]].plot(color=colours[ii],label=lst[ii],lw=1.5)
    plt.title(title)
    plt.legend();plt.show()
