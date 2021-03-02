# Split for TimeSeries (Split Entire Dataframe into Subsets)

def TimeSeries_Split(ldf,split_id=[None,None],test_id=False,cut_id=None):
    
    # Reduce the number of used data (destructive)
    if(cut_id is not None):
        print('data reduction used')
        ldf = ldf.iloc[-cut_id:]
        t1 = ldf.index.max();t0 = ldf.index.min()
        print(f'Dataset Min.Index: {t0} | Max.Index: {t1}')
        
    # Option to split training/test sets
    if(split_id[0] is not None):
        # General Percentage Split (Non Shuffle requied for Time Series)
        train_df,pred_df = train_test_split(ldf,test_size=split_id[0],shuffle=False)
    elif(split_id[1] is not None):
        # specific time split 
        train_df = df.loc[:split_id[1]]; pred_df = df.loc[split_id[1]:] 
    else:
        print('Choose One Splitting Method Only')
        
    return train_df,pred_df # return 
