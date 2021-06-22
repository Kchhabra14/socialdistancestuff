import numpy as np
import cv2
from config import *

def detectpeople(frame, net, outputlayers,height,width):
	(H, W) = height,width
	results = []
    
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(outputlayers)

	boxes = []
	classids = []
	centroids = []
	confidences = []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > confidencethreshold:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
				classids.append(classID)
	finalindexs = cv2.dnn.NMSBoxes(boxes, confidences,confidencethreshold,nmsthreshold)

	if len(finalindexs) > 0:
		for i in finalindexs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			r = (classids[i],confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)
	return results               