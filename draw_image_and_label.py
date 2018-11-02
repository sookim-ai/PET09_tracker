from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.colors as clr
import matplotlib.patches as patches
from PIL import Image
import scipy.ndimage
import numpy as np
import sys
import skimage.measure

#execution:
#python *.py image.jpg


track_all=[i for i in range(35)]
for track in range(len(track_all)):
    filename = ["/export/kim79/h2/Crowd_PETS09/test/div_image_"+str(track)+".npy"]
    filename2 = ["/export/kim79/h2/Crowd_PETS09/test/div_label_"+str(track)+".npy"]
    labelname=["./bbox/test_result_0_track_"+str(track)+".npy"]
    #(4649, 1, 576, 768, 3)

    f=np.load(filename[0])
    f2=np.load(filename2[0])

    f=skimage.measure.block_reduce(np.asarray(f), (1,2,2,1), np.max)
    f2=skimage.measure.block_reduce(np.asarray(f2), (1,2,2,1), np.max)

    fl=np.load(labelname[0])
    print(np.shape(f),np.shape(fl))



    for i in range(len(f)):
        img_o=f[i,:,:,:]
        min_val=0;max_val=1;mid_val=(min_val+max_val)/2.0;
        print("max: ",max_val,"min: ",min_val)
        cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162','#DCE6F1'], N=256)
        bounds=[min_val,mid_val,max_val]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = pyplot.subplots(1)
        pyplot.subplots_adjust(hspace = .001)
        img = ax.imshow(img_o,interpolation='nearest',cmap = cmap)
    #    pyplot.show()
        pyplot.savefig("input_track"+str(track)+"_"+str(i)+".png")
        pyplot.close()

        img_in=f2[i,:,:,:]
        img_o=np.divide(img_in-np.amin(img_in),np.amax(img_in)-np.amin(img_in))
        min_val=np.amin(img_o);max_val=np.amax(img_o);mid_val=(min_val+max_val)/2.0;
        print("max: ",max_val,"min: ",min_val)
        cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162','#DCE6F1'], N=256)
        bounds=[min_val,mid_val,max_val]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = pyplot.subplots(1)
        pyplot.subplots_adjust(hspace = .001)
        img = ax.imshow(img_o,interpolation='nearest',cmap = cmap)
    #    pyplot.show()
        pyplot.savefig("groundtruth_track"+str(track)+"_"+str(i)+".png")
        pyplot.close()


        img_in=fl[i,:,:,:]
        img_o=np.divide(img_in-np.amin(img_in),np.amax(img_in)-np.amin(img_in))
        min_val=np.amin(img_o);max_val=np.amax(img_o);mid_val=(min_val+max_val)/2.0;
        print("max: ",max_val,"min: ",min_val)
        cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162','#DCE6F1'], N=256)
        bounds=[min_val,mid_val,max_val]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = pyplot.subplots(1)
        pyplot.subplots_adjust(hspace = .001)
        img = ax.imshow(img_o,cmap = cmap)
    #    pyplot.show()
        pyplot.savefig("output_track"+str(track)+"_"+str(i)+".png")
        pyplot.close()   
