import tensorflow as tf
from tensorflow.contrib import rnn
from function import *
from read_input import *
from inference import *
from load_data import *
from rnn import *
import numpy as np

threshold = 0.3

def get_max(my_list):
    m = None
    for item in my_list:
        if isinstance(item, list):
            item = get_max(item)
        if not m or m < item:
            m = item
    return m

def get_min(my_list):
    m = None
    for item in my_list:
        if isinstance(item, list):
            item = get_min(item)
        if not m or m > item:
            m = item
    return m

def find_minimum_continuous_digts_cluster(h):
    h_index=[]
    h_sub=[h[0]]
    h_cp=h
    for j in xrange(1,len(h)):
        if h_cp[j]==h_cp[j-1]+1:
            h_sub.append(h_cp[j])
        else:
            h_index.append(h_sub)
            h_sub=[h[j]]
    h_index.append(h_sub)
    length=[]
    for j in range(len(h_index)):
        length.append(len(h_index[j]))
    indx=length.index(max(length))
    h_cluster=h_index[indx]
    return h_cluster


def cleaning_boundingbox(image):
    #np.shape(image): [h,w,channels]
    #in image, find tightest bounding box
    image = np.reshape(image,[288,384,3])
    image_1ch = image[:,:,0]+image[:,:,1]+image[:,:,2]
    print(np.shape(image),np.shape(image_1ch)) #((1, 288, 384, 3), (1, 288, 3))
    h_index, w_index = np.where( image_1ch > float(threshold) )
    if len(h_index)==0: return np.zeros(np.shape(image))
    else:
        #align w according to h
        h=[h_index[0]]; w=[]; w_inside=[w_index[0]]
        for i in xrange(1,len(h_index)):
            if h_index[i]!=h_index[i-1]:
                h.append(h_index[i])
                w.append(w_inside)
                w_inside=[w_index[i]]
            else: 
                w_inside.append(w_index[i])
        w.append(w_inside)
        # bounding box citeria(1) h should be countiuous
        print(h_index,h)
        h_cluster=find_minimum_continuous_digts_cluster(h)
        #align w according to h_cluster index
        w_cluster=[]
        for j in range(len(h_cluster)):
            w_cluster.append(find_minimum_continuous_digts_cluster(w[h.index(h_cluster[j])]) )
        w_index_all=[]
        for j in range(len(h_cluster)):
            w_index_all=w_index_all+w_cluster[j]
        w_index=np.unique(w_index_all)
        w_index_count=[]
        length=len(w_cluster)
        for i in w_index:
            count_val=0
            for j in range(len(w_cluster)):
                if i in w_cluster[j]:
                    count_val=count_val+1
            w_index_count.append(count_val)
        w_index_update=[]
        for i in range(len(w_index)):
            if w_index_count[i] > int(length*0.5):
                w_index_update.append(w_index[i])
        w_index_update=find_minimum_continuous_digts_cluster(w_index_update)
        w_min=get_min(w_index_update)
        w_max=get_max(w_index_update)
        h_max=get_max(h_cluster)
        h_min=get_min(h_cluster)
        print(h_max,h_min,w_max,w_min)
        #generate mask on image
        image_mask =  np.zeros(np.shape(image))
        for ch in range(3):
            for i in range(288):
                for j in range(384):
                    if i > h_min and i < h_max and j > w_min and j < w_max:
                        image_mask[i,j,ch] = image[i,j,ch]
                    else:
                        image_mask[i,j,ch] = 0.0
    return image_mask          
          
    






def switch_first_frame(name, test_X, image, timestep, itr):
#    if int(name) > 3 and int(name)%2==0:
    image=cleaning_boundingbox(image)
    image_out=np.zeros([batch_size,itr+timestep,h,w,3])
    #Swap first input
    for k in range(batch_size):
        for j in range(itr+timestep):
            if j<(itr+1):
                image_out[k,j,:,:,:]=image
            else:
                image_out[k,j,:,:,:]=test_X[k,j-itr,:,:,:]
    return image_out 



def test(name,sess,loss_op,train_op,X,Y,prediction,last_state,fout_log):
    fetches = {'final_state': last_state,
              'prediction_image': prediction}
    for ii in range(35): #total 35 tracks in test data
        image=np.load("/export/kim79/h2/Crowd_PETS09/test/div_image_"+str(ii)+".npy") #one track
        init_label=np.load("/export/kim79/h2/Crowd_PETS09/test/div_label_"+str(ii)+".npy")[0,:,:,:]
        image=skimage.measure.block_reduce(np.asarray(image), (1,2,2,1), np.max)
        init_label=skimage.measure.block_reduce(np.asarray(init_label), (2,2,1), np.max)
        image=np.divide(image,255.0)
        init_label=np.divide(init_label,255.0)
        output_image = np.zeros(np.shape(image))
        d1,d2,d3,d4=np.shape(image) #(100,h,w,3)
        timestep=3
        for j in range(int(d1/(timestep-1))+1):
            if j ==  int(d1/(timestep-1)): # append to last block
                end = d1
                start = end - timestep
                print("hit")
            else:
                start = j*(timestep-1)
                end = (j+1)*(timestep-1)+1
                if end > d1:
                    end=d1
                    start =  end - timestep
            #print(ii, int(d1/(timestep-1)),j,start,end,end-start)
            test_x=np.reshape(image[start:end,:,:,:],[1, end-start, h, w, channels])
            test_X=np.concatenate([test_x for i in range(batch_size)],0) # all batch has same vector
            if j == 0:
                X_te=switch_first_frame(name,test_X, init_label, timestep, 10) #X = tf.placeholder("float", [FLAGS.batch_size, None, h,w,channels])
            else: #recursive feed back
                o1,o2,o3,o4,o5=np.shape(output)
                X_te=switch_first_frame(name,test_X, output[0,o2-1:o2,:,:,:], (end-start), 10) #X = tf.placeholder("float", [FLAGS.batch_size, None, h,w,channels]) 
            eval_out=sess.run(fetches, feed_dict={X:X_te})
            output=eval_out['prediction_image']
            len_out=len(output)
            #print(output[0,10:10+(end-start),:,:,:])
            output_image[start:end,:,:,:] = output[0,10:10+(end-start),:,:,:]
        print(ii)
        np.save("test_result_"+str(name)+"_track_"+str(ii)+".npy", output_image)




