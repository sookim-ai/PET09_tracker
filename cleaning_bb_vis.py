from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.colors as clr
import matplotlib.patches as patches
from PIL import Image
import scipy.ndimage
import numpy as np
import sys
import skimage.measure
from testing import *


#execution:
#python *.py image.jpg

f=np.load("./trial_epc2/test_result_4_track_28.npy")

for i in range(len(f)):
    img_in=f[i,:,:,:]
    img_o=np.divide(img_in-np.amin(img_in),np.amax(img_in)-np.amin(img_in))
    min_val=np.amin(img_o);max_val=np.amax(img_o);mid_val=(min_val+max_val)/2.0;
    print("Timestep: ",i,"max: ",max_val,"min: ",min_val)
    cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162','#DCE6F1'], N=256)
    bounds=[min_val,mid_val,max_val]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = pyplot.subplots(1)
    pyplot.subplots_adjust(hspace = .001)
    img = ax.imshow(img_o,interpolation='nearest',cmap = cmap)
    pyplot.show()
    pyplot.close()

    img_inn=cleaning_boundingbox(img_o)
    img_oo=np.divide(img_inn-np.amin(img_inn),np.amax(img_inn)-np.amin(img_inn))
    min_val=np.amin(img_oo);max_val=np.amax(img_oo);mid_val=(min_val+max_val)/2.0;
    print("max: ",max_val,"min: ",min_val)
    cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162','#DCE6F1'], N=256)
    bounds=[min_val,mid_val,max_val]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = pyplot.subplots(1)
    pyplot.subplots_adjust(hspace = .001)
    img = ax.imshow(img_oo,interpolation='nearest',cmap = cmap)
    pyplot.show()
    pyplot.close()


