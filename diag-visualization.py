import matplotlib as mpl
from matplotlib import pyplot
import matplotlib.colors as clr
import numpy as np
import random
image=np.load('./img.npy') #[10,10,h,w]
label=np.load("./label.npy")
data_num,t,h,w=np.shape(image)
for i in range(data_num):
 print(np.shape(image))
 for k in range(t):
    inn=image[i,k,:,:]
    yy=label[i,k,:,:]
    print("Timestep "+str(k))
    min_val=0
    max_val=1
    mid_val=(min_val+max_val)/2.0
    # make a color map of fixed colors
    cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162','#DCE6F1'], N=256)
    bounds=[min_val,mid_val,max_val]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    print(i,k)
    print("image")
    # tell imshow about color map so that only set colors are used
    img = pyplot.imshow(inn,interpolation='nearest',cmap = cmap)
    
    # make a color bar
    pyplot.colorbar(img,cmap=cmap,
                norm=norm,boundaries=bounds,ticks=[min_val,mid_val,max_val])
    pyplot.show()

    print("label")
    # tell imshow about color map so that only set colors are used
    img = pyplot.imshow(yy,interpolation='nearest',cmap = cmap)
    
    # make a color bar
    pyplot.colorbar(img,cmap=cmap,
                norm=norm,boundaries=bounds,ticks=[min_val,mid_val,max_val])
    pyplot.show()

#    print("prediction label")
#    # tell imshow about color map so that only set colors are used
#    img = pyplot.imshow(pre,interpolation='nearest',cmap = cmap)
    
#    # make a color bar
#    pyplot.colorbar(img,cmap=cmap,
#                norm=norm,boundaries=bounds,ticks=[min_val,mid_val,max_val])
#    pyplot.show()



