import pyvista as pv
import pandas as pd
import numpy as np
import time

# Read & Load VTM (Multiblock VTK)
# load all read files into lst_pv list

t0 = time.time()
lst = ['fastpoly_10.vtm','fastpoly_20.vtm'] 
lst_pv = []
for i in lst:
    tpv = pv.read(i)
    lst_pv.append(tpv)  # load all read files into lst_pv
t1 = time.time()
tt = t1-t0
print(f'load time: {tt}')
