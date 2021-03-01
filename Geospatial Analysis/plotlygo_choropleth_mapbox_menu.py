''' Dropdown Menu Interactive Choropeth w/ Plotly Go Example '''

# Unique Token ID
mapbox_access_token = 'pk.eyJ1Ijoic2h0cmF1c3NhcnQiLCJhIjoiY2tqcDU2dW56MDVkNjJ6angydDF3NXVvbyJ9.nx2c5XzUH9MwIv4KcWVGLA'

trace = []    
# Set the data for the map
for i in features:
    trace.append(go.Choroplethmapbox(geojson = lga_json,
                                    locations = df_merged.index,    
                                    z = df_merged[i].values,                     
                                    text = df_merged.index,
                                    hovertemplate = "<b>%{text}</b><br>" +
                                                            "%{z}<br>" +
                                                            "<extra></extra>",
                                    colorbar=dict(thickness=10, ticklen=3,outlinewidth=0),
                                    marker_line_width=1, marker_opacity=0.8, colorscale="Blues_r",
                                    visible=False)
                            )
trace[0]['visible'] = True # set the visibility of the first entry class's visibility content

# add a dropdown menu in the layout
layout.update(height=500,updatemenus=list([dict(x=0.8,y=1.1,xanchor='left',yanchor='middle',buttons=lst)]))

# The rest is the same
fig=go.Figure(data=trace, layout=layout)
fig.update_layout(title_text='Suburb Mean Values', title_x=0.01)
fig.update_layout(
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=90,
        center=dict(lat=-32, lon=115.9),
        pitch=0,
        zoom=9.5
    )
)
fig.update_layout(margin={"r":0,"t":80,"l":0,"b":0},mapbox_style="light");fig.show()