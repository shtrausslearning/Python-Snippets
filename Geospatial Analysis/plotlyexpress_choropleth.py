''' Interactive Choropeth w/ Plotly Express Example '''

tmp = df_merged2.copy()
tmp['Percent_Unemployed'] = tmp['Percent_Unemployed']*100
zmin = tmp['Percent_Unemployed'].min(); zmax = tmp['Percent_Unemployed'].max()
fig = px.choropleth_mapbox(tmp, geojson=lga_json, locations=tmp.index, 
                                    color=tmp['Percent_Unemployed'],
                                    color_continuous_scale="speed",
                                    range_color=(zmin,zmax),
                                    hover_name = df_merged2.LGA_NAME20,   
                                    mapbox_style="carto-positron",
                                    zoom=7,height=400,
                                    center = dict(lat= -38 , lon=145),  
                                    opacity=0.8,  
                                    title = 'Percentage Unemployed Males [20-24]')
fig.update_layout(margin=dict(l=0, r=0, t=50, b=0));fig.show()