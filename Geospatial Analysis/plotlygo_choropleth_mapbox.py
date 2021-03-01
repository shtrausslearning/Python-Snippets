''' Interactive Choropeth w/ Plotly Go Example '''

MAPBOX_ACCESSTOKEN = 'pk.eyJ1Ijoic2h0cmF1c3NhcnQiLCJhIjoiY2tqcDU2dW56MDVkNjJ6angydDF3NXVvbyJ9.nx2c5XzUH9MwIv4KcWVGLA'

# define colourbar bounds
zmin = df_merged0['Percent_Unemployed'].min()
zmax = df_merged0['Percent_Unemployed'].max()

# Set the data for the map
data = go.Choroplethmapbox(
        geojson = lga_json,             #this is your GeoJSON
        locations = df_merged0.index,    #the index of this dataframe should align with the 'id' element in your geojson
        z = df_merged0['Percent_Unemployed'], #sets the color value
        text = df_merged0.LGA_NAME20,    #sets text for each shape
        colorbar=dict(thickness=20, ticklen=3, 
                     tickformat='%',outlinewidth=0), #adjusts the format of the colorbar
        marker_line_width=1, marker_opacity=0.8, colorscale="speed", #adjust format of the plot
        zmin=zmin, zmax=zmax,           #sets min and max of the colourbar
        hovertemplate = "<b>%{text}</b><br>" +
                    "%{z:.0%}<br>" +
                    "<extra></extra>")  # sets the format of the text shown when you hover over each shape

# Set the layout for the map
layout = go.Layout(
    title = {'text': f"Percentage Unemployed Females [20-24]",
            'font': {'size':20}},       
    mapbox1 = dict(
        center = dict(lat= -38 , lon=145),zoom = 7,
        accesstoken = MAPBOX_ACCESSTOKEN),                      
    autosize=True,
    height=400)

# Generate the map
fig=go.Figure(data=data, layout=layout)
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0));fig.show()
