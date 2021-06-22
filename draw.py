import cv2
import numpy as np
from scipy.spatial import distance as dist

def drawonframe(frame,final,classes):
    img = frame
    for (index,(classid,prob,bbox,centeriod)) in enumerate(final):
        (startx,starty,endx,endy) =bbox
        color=(0,255,0)
        (cx,cy) = centeriod
        label=str(classes[classid])
        cv2.rectangle(frame,(startx,starty),(endx,endy),color,2)
        cv2.circle(frame,(cx,cy),5,color,1)
        cv2.putText(frame,label+"  "+str(int(prob*100))+"%",(startx,starty-10),cv2.FONT_HERSHEY_COMPLEX,0.75,color,2)
    return img

def drawonframepeople(frame,final,classes):
    img = frame
    for (index,(classid,prob,bbox,centeriod)) in enumerate(final):
        if classid == 0:
            (startx,starty,endx,endy) =bbox
            color=(0,255,0)
            (cx,cy) = centeriod
            cv2.rectangle(frame,(startx,starty),(endx,endy),color,2)
            cv2.circle(frame,(cx,cy),5,color,1)
    return img

def drawonframepeopleviolate(frame,final,classes):
    img = frame
    violate = set()
    voilates = 0
    if len(final) >= 2:
        centroids = np.array([r[2] for r in final])
        D = dist.cdist(centroids, centroids, metric="euclidean")
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                if D[i, j] < 75:
                    violate.add(i)
                    violate.add(j)

    for (index,(classid,prob,bbox,centeriod)) in enumerate(final):
        if classid == 0:
            (startx,starty,endx,endy) =bbox
            color=(0,255,0)
            if index in violate:
                color=(0,0,255)
            (cx,cy) = centeriod
            cv2.rectangle(frame,(startx,starty),(endx,endy),color,2)
            cv2.circle(frame,(cx,cy),5,color,1)
            cv2.putText(frame,"number of people = "+str(len(final)),(10,frame.shape[0]-25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    return img