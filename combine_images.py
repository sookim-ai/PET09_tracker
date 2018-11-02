import sys
from PIL import Image
import os
for ii in xrange(0,35):
    file_list=os.popen("ls image2/track"+str(ii)+"/input_*").read()
    aa=file_list.split("\n")
    length=len(aa)-1    
    print(length, ii)
    images = map(Image.open, ['track'+str(ii)+'/'+'input_track'+str(ii)+'_'+str(i)+'.png' for i in range(length)])
    gt     = map(Image.open, ['track'+str(ii)+'/'+'groundtruth_track'+str(ii)+'_'+str(i)+'.png' for i in range(length)]) 
    result = map(Image.open, ['track'+str(ii)+'/'+'output_track'+str(ii)+'_'+str(i)+'.png' for i in range(length)]) 
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    print(total_width,max_height)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset=0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
    new_im.save('image_'+'track'+str(ii)+'.png')

    x_offset=0
    for im in gt:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
    new_im.save('gt_'+'track'+str(ii)+'.png')

    x_offset=0
    for im in result:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
    new_im.save('result_'+'track'+str(ii)+'.png')


for ii in range(35):
    image_list =[ 'image_track'+str(ii)+'.png' , 'gt_track'+str(ii)+'.png' , 'result_track'+str(ii)+'.png']
    imgs = [ Image.open(i) for i in image_list ]
    min_img_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    img_merge = np.vstack( (np.asarray( i.resize(min_img_shape,Image.ANTIALIAS) ) for i in imgs ) )
    img_merge = Image.fromarray( img_merge)
    img_merge.save( 'combine_'+str(ii)+'.jpg' )
