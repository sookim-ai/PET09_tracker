import numpy as np
import os,sys

os.system("mkdir trial")
os.system("mv ./*.npy ./trial")
os.system("mkdir image")
os.system("python draw_image_and_label.py")
os.system("mv ./*.png ./image")
os.system("cd ./image") #this may not work.
os.system("pwd")
exit()
for i in range(35):
    os.system("mkdir track"+str(i))
    os.system("mv *_track"+str(i)+"_* track"+str(i))
os.system("cp ../combine_images.py  .")
os.system("python combine_images.py")
os.system("cd ..")








