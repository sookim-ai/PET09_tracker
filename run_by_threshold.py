import os,sys

for i in range(50):
    threshold=0.01+float(i)*0.01
    a=5
    print (str(threshold) + " starting ..")
    os.system("python error_matric_noplot_v1.py "+str(threshold)+" "+str(a) + " >> output.txt")  

