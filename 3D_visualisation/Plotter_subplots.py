# Subplot Plotter Example

# Start a Plot
plotter = pv.Plotter(line_smoothing=True,polygon_smoothing=True,point_smoothing=True,
                     shape=(2,1),border=True,notebook=True,window_size=(1000,1000),
                     multi_samples=10)

# Subplot 1
plotter.subplot(0,0)
# plotter.add_text('15 m/s',position='lower_right',font_size=13)
plotter.add_mesh(lst_pv[0][0],style='surface', show_edges=True,opacity=0.05,color='black')
plotter.camera_position = cam_iso1  # define camera position to define view 
plotter.enable_anti_aliasing() 

# Subplot 2
plotter.subplot(1,0)
# plotter.add_text('25 m/s',position='lower_right',font_size=13)
plotter.add_mesh(lst_pv[0][1],style='surface', show_edges=True,opacity=0.05,color='black')
plotter.camera_position = cam_iso2  # define camera position to define view 
# plotter.enable_parallel_projection()  # 2D type of view 
plotter.enable_anti_aliasing()
plotter.camera.Zoom(5) 

# Plot Aesthetics & Camera Settings
plotter.add_axes()
plotter.background_color = 'white' 
plotter.screenshot('./out_readpvcc.png')
plotter.show()
