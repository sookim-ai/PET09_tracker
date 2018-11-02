import matplotlib as mpl
from matplotlib import pyplot
import matplotlib.colors as clr
import numpy as np
#import skimage.measure
import random
name=["input"]
image_inn=np.load('./X_test.npy') #(84, 24, 10, 128, 257, 3)
label_inn=np.load("./Y_test.npy") #(84, 24, 10, 128, 257, 1)
prediction_inn=np.load("./test_result_fwbw_1.npy")[:,0,:,:,:,:,:] #(84,2, 24, 20, 128, 257, 1)
d1,d2,d3,d4,d5,d6=np.shape(image_inn)
print(np.shape(image_inn)) #(80, 25, 40, 64, 64, 1)
print(np.shape(label_inn)) #(80, 25, 40, 64, 64, 1)
print(np.shape(prediction_inn))
for ii in range(d1): #84
    t_start=15
    t_end=15
    image_in=image_inn[ii,:,t_start-1:d3-t_end+1,:,:,:]
    label_in=label_inn[ii,:,t_start-1:d3-t_end+1,:,:,0] ##[24,10,128,257]
    prediction_in=prediction_inn[ii,:,t_start-1:d3-t_end+1,:,:,0] #[24,10,128,257]
    for i in range(d2): #24
        image=image_in[i,:,:,:,:] #[10,128,257,3]
        label=label_in[i,:,:,:] #[10,128,257]
        prediction=prediction_in[i,:,:,:] #[10,128,257]
        s1,s2,s3=np.shape(prediction) 
        for k in range(s1): #10 time
            pyplot.figure(1,figsize=(15,3))
            yy=label[k,:,:]
            pre=prediction[k,:,:]
            img_o=image[k,:,:,:] 
            h=64; w=64;
            yy=np.reshape(yy,[h,w])
            pre=np.reshape(pre,[h,w])
            print("Timestep "+str(k))
            min_val=0;max_val=1;mid_val=(min_val+max_val)/2.0;
            # make a color map of fixed colors
            cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162','#DCE6F1'], N=256)
            bounds=[min_val,mid_val,max_val]
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            print("Filenumber",ii,"Batch ",i,"time steps ",k)
            #u850,v850,prect
            for ch in range(d6):
                pyplot.subplot(3,12,s1*ch+k+1)
                pyplot.subplots_adjust(hspace = .001)
                if k==0: pyplot.title(name[ch])
                img = pyplot.imshow(img_o[:,:,ch],interpolation='nearest',cmap = cmap)
                pyplot.axis('off')
            #Ground truth heat map
            print("ground truth density map")
            pyplot.subplot(3,12,s1*d6+k+1)
            pyplot.subplots_adjust(hspace = .001)
            if k==0: pyplot.title("Ground Truth")
            img = pyplot.imshow(yy,interpolation='nearest',cmap = cmap)
            pyplot.axis('off')
            #Prediction heat map
            pyplot.subplot(3,12,s1*(d6+1)+k+1)
            if k==0: pyplot.title('Prediction')
            pyplot.subplots_adjust(hspace = .001)
            img = pyplot.imshow(pre,interpolation='nearest',cmap = cmap)
            pyplot.axis('off')
        pyplot.tight_layout()
        pyplot.show()
        pyplot.savefig("all_"+str(ii)+"_"+str(i)+".png")
        pyplot.close()
