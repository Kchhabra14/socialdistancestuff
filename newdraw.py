import cv2
import numpy as np

def draw(img,result,classes):
    frame = img
    for (index,(classid,confidence,box,centroid)) in enumerate(result):
            (startx,starty,endx,endy)=box
            color=(0,255,0)
            (cx,cy)= centroid
            label=str(classes[classid])
            cv2.rectangle(frame,(startx,starty),(endx,endy),color,2)
            cv2.circle(frame,(cx,cy),5,color,1)
            cv2.putText(frame,label+" "+str(int(confidence*100))+"%",(startx,starty-10),cv2.FONT_HERSHEY_COMPLEX,0.75,color,2)
    return frame
