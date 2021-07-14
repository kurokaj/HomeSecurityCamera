# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 16:34:41 2021

@author: jonne.kurokallio
"""

# Imports
#import cv2 
import numpy as np 
#import onnx
#import onnxruntime as ort
#from onnx_tf.backend import prepare

# Tracker
from collections import OrderedDict
from scipy.spatial import distance as dist 
import time

# Update first the tracker, then feed the updated persons to identification model and update the labels

class person():
    def __init__(self, label="Unknown"):
        self.label = label          # String : Name of the tracked person
        self.recent_labels = []     # List : 20 most recent detections from frames (to make the verification more accurate)
        self.location = (0,0)       # Tuple : (x,y) coordinates of the tracked object
    
    def addRecentLabel(self,l):
        # Function to set the most recent label to the list
        length = len(self.recent_labels)
        if length < 20:
            self.recent_labels.append(l)
        else:
            self.recent_labels = self.recent_labels[1:]
            self.recent_labels.append(l)
    
    def updateLabel(self,l):
       # Function to update the object label after every scan
        self.addRecentLabel(l)
        self.label = max(set(self.recent_labels), key=self.recent_labels.count)
    

class tracker():
    def __init__(self, maxDisappeared=100):
        # Keys for every object
        self.nextObjectID = 0
        # Objects
        self.objects = OrderedDict()
        # Objects outside frame
        self.disappeared = OrderedDict()
        # Running number of visitors
        self.numberOfVisitors = 0
        # The amount of frames object can be outside of frame before it is considered a new visitor
        self.maxDisappeared = maxDisappeared
        
    
    def addVisitor(self,person):
        # Function to add a new visitor
        self.objects[self.nextObjectID] = person
        self.disappeared[self.nextObjectID] = 0
        self.numberOfVisitors += 1
        self.nextObjectID += 1 
    
    def dropVisitor(self,key):
        del self.objects[key]
        del self.disappeared[key]
        
        
    def updateTracker(self, newPeople):
        # Main function to track people and update labels after recognition frame
        
        # If nothing is detected
        if len(newPeople) == 0:
            # Mark all tracked objects as disappeared
            for ID in list(self.disappeared.keys()):
                self.disappeared[ID] += 1
                
                # Check if enough time has elapsed to delete object
                if self.disappeared[ID] > self.maxDisappeared:
                    self.dropVisitor(ID)
                    
            return self.objects
        
        # An array of centroids for current frames NEW PEOPLE CENTROIDS
        inputCentroids = np.zeros((len(newPeople),2),dtype="int")
        
        # Loop the bounding boxes
        for (idx, person) in enumerate(newPeople):
            inputCentroids[idx] = calculateCentroid(person.location)
        
        # If the register is empty loop and place new values to the register
        if len(self.objects) == 0:
            for p in newPeople:
                self.addVisitor(p)
        # If the register is not empty, compare new found people to the register and update
        else:
            objectIDs = list(self.objects.keys())
            # This will probably throw error!!!!!! one liner needed
            
            
            #objectCentroids = list(self.objects.values().location)
            objectCentroids = [calculateCentroid(value.location) for value in list(self.objects.values())]
            
            # Calculate the euclidean distances of old and new detections 
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # Visualize this to understand better
            # Find the smallest distances
            rows = D.argmin(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            # Associate new object IDs
            usedRows = set()
            usedCols = set()
            
            for (row, col) in zip(rows, cols):
                # The row or column has been examinded -> ignore
                if row in usedRows or col in usedCols:
                    continue
                
                # Otherwise grab the ID of the object for current row and set its new cendroid
                # Reset disappear counter
                objectID = objectIDs[row]
                self.objects[objectID].location = newPeople[col].location # Update location
                self.objects[objectID].updateLabel(newPeople[col].label)
                self.disappeared[objectID] = 0

                # Row and column has been inspected
                usedRows.add(row)
                usedCols.add(col)
            
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            # Check if some of the objects have disappeared
            # ie. there is less new detections than there is registered ones
            if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
                for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
                    objectID = objectIDs[row]  
                    self.disappeared[objectID] += 1
					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.dropVisitor(objectID)
            # The number of new detections is greater than registered -> new detections
            else:
                    for col in unusedCols:
                    # Will throw error because we need to register a person!!!!
                        self.register(inputCentroids[col])
            
        return self.objects
                    

def calculateCentroid(box):
    # Function to calculate the cendroid of given box
    # (x1,y1,x2,y2)
    x = box[0] + (box[2]-box[0])/2
    y = box[1] + (box[3]-box[1])/2
    
    return np.array((x,y))
    
    
# Main function
def main():
    
    p1 = person()
    p1.label = "Jonne"
    p1.location = (5,5,10,10)
    p2 = person()
    p2.location = (25,25,30,30)
    print(p2.label)
    
    p1.updateLabel("Matias")
    p1.updateLabel("Jonne")
    p1.updateLabel("Matias")
    p1.updateLabel("Matias")
    p1.updateLabel("Jonne")
    p1.updateLabel("Matias")
    
    print(p1.recent_labels)
    print(p1.label)
    
    p3 = person()
    p3.label = "Matias"
    p3.location = (8,7,10,10)
    
    p4 = person()
    p4.label = "Hemmo"
    p4.location = (20,19,10,10)
    
    
    objectTracker = tracker()
    objectTracker.updateTracker([p1,p2])
    
    for v in objectTracker.objects.values():
        print(v.label)
        print(v.recent_labels)
        print(v.location)
        
    print(objectTracker.nextObjectID)
    
        
    objectTracker.updateTracker([p3,p4])
    
    for v in objectTracker.objects.values():
        print(v.label)
        print(v.recent_labels)
        print(v.location)
        
    print(objectTracker.nextObjectID)
    
    i = 0
    while i<20:
        start_time = time.time()

        objectTracker.updateTracker([p3])
        for v in objectTracker.objects.values():
            print(v.label)
            print(v.recent_labels)
            print(v.location)
        
        print(objectTracker.nextObjectID)
        i=i+1
        print("--- %s seconds ---" % (time.time() - start_time))
    
    #video_capture = cv2.VideoCapture(0)
    
    #onnx_model = onnx.load('ultra_light/ultra_light_models/Mb_Tiny_RFB_FD_train_input_640.onnx')
    #predictor = prepare(onnx_model)
    #ort_session = ort.InferenceSession(onnx_path)
    #input_name = ort_session.get_inputs()[0].name

    
    # while True:
    #     ret, frame = video_capture.read()
        
    #     # Resize the image for the ultra_light_640.onnx model (face detection)
    #     h, w, _ = frame.shape
        
    #     # preprocess img acquired
    #     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     img = cv2.resize(img, (640, 480)) 
    #     cv2.imshow('Video', img)
        
    #     img_mean = np.array([127, 127, 127])
    #     img = (img - img_mean) / 128
    #     img = np.transpose(img, [2, 0, 1])
    #     img = np.expand_dims(img, axis=0)
    #     img = img.astype(np.float32)
        
        
    #     # 'q' to quit!
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
        

        
    # # Release handle to the webcam
    # video_capture.release()
    # cv2.destroyAllWindows()
    
    
# Execute main function   
if __name__ == "__main__":
    main()