import numpy as np
import pandas as pd
import json 

from pyntcloud import PyntCloud
import pickle, h5py

from tqdm import tqdm

import os
os.environ["QT_API"] = "pyqt5"

import pyvista as pv
import binvox_rw
# from pyvistaqt import BackgroundPlotter, MultiPlotter

def show_voxels(voxels,true_threshold=0.5,title='',n_rows=1):
    '''
    Args:
    voxel (numpy.ndarray(l,w,h)): A voxel grid of dimension l x w x h
    '''
    n_voxels = voxels.shape[0]
    n_cols = int(np.ceil(n_voxels/n_rows))
    # voxels = np.transpose(voxels, (0,3,2,1))
    plotter = pv.Plotter(shape=(n_rows,n_cols),title=title)
    r = 0; c = 0
    for i in range(n_voxels):
        voxel = voxels[i]
        grid = pv.UniformGrid()
        grid.dimensions = voxel.shape
        grid.spacing = (1, 1, 1)  
        grid.point_data["values"] = voxel.flatten(order="F")  # Flatten the array!
        grid = grid.threshold(true_threshold, invert=False)
        plotter.subplot(r,c)
        plotter.add_mesh(grid,name=f'{r}_{c}',show_edges=True)
        c+=1
        if c==n_cols:
            r+=1
            c=0
        if r==n_rows:
            break
    plotter.show()
    return 

with open('airbus_64.binvox', 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)

data = model.data
data = np.expand_dims(data,0).astype(float)
show_voxels(data)
print()