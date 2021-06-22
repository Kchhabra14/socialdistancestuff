from draw import drawonframe
from detection import detectpeople
from config import *
import cv2
import numpy as np

net = cv2.dnn.readNetFromDarknet(Modelconfig,Modelweights)

classes = []
with open(Modelclasses,"r") as file:
    classes=[eachline.strip() for eachline in file.readlines()]

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layernames = net.getLayerNames()
outputlayers = [layernames[index[0]-1] for index in net.getUnconnectedOutLayers()]

video = cv2.VideoCapture("video2.mp4")

originalheight = video.get(4)
originalwidth = video.get(3)

ratio = modwidth / float(originalwidth)
modheight = int(originalheight * ratio)

while True:
    cap,frame = video.read()
    frame = cv2.resize(frame,(modwidth,modheight),interpolation=cv2.INTER_AREA)
    results = detectpeople(frame,net,outputlayers,modheight,modwidth)
    img = drawonframe(frame, results,classes)
    cv2.imshow("frame",img)
    if cv2.waitKey(1) & 0xff==ord('q'):
            break