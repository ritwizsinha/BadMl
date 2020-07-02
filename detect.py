import cv2 
import numpy as np
import math
from keras.models import model_from_json
from string import ascii_uppercase
count = 0
map = dict()
for i in ascii_uppercase:
    if( i != 'J' and i !='Z'):
        map[count] = i
        count+=1

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

bgSubtractor = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=30, detectShadows=False)

def bgSubMasking(frame):
    fgmask = bgSubtractor.apply(frame, learningRate=0)    
    # cv2.imshow("fgmask",fgmask)
    kernel = np.ones((4, 4), np.uint8)
    # The effect is to remove the noise in the background
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    # cv2.imshow("enhancedFGMASK",fgmask)
    # To close the holes in the objects
    return cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # cv2.imshow("closedHolesFGMASK",fgmask)
    # Apply the mask on the frame and return
    # aand = cv2.bitwise_and(frame, frame, mask=fgmask)
    # cv2.imshow("and",aand)
    # return aand
def threshold(mask):
    """Thresholding into a binary mask"""
    grayMask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grayMask, 0, 255, 0)
    return thresh
cap = cv2.VideoCapture(0)

def getMaxContours(contours):
    """Find the largest contour"""
    maxIndex = 0
    maxArea = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > maxArea:
            maxArea = area
            maxIndex = i
    return contours[maxIndex]
def countFingers(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        cnt = 0
        if type(defects) != type(None):
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s, 0])
                end = tuple(contour[e, 0])
                far = tuple(contour[f, 0])
                angle = calculateAngle(far, start, end)
                
                # Ignore the defects which are small and wide
                # Probably not fingers
                if d > 10000 and angle <= math.pi/2:
                    cnt += 1
        return True, cnt+1
    return False, 0
    
def calculateAngle(far, start, end):
    """Cosine rule"""
    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
    angle = math.acos((b**2 + c**2 - a**2) / (2*b*c))
    return angle
while True:
    ret, frame = cap.read()
    rows,columns,channels = frame.shape
    frame = frame[0:300,0:300]
    frame = cv2.flip(frame,1)
    frame = bgSubMasking(frame) 
    _, contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frame = cv2.resize(frame,(28,28), interpolation=cv2.INTER_AREA)
    array = np.array(frame)
    array = array.reshape((1,28,28,1))
    pred = loaded_model.predict(array)
    pred = pred.flatten()
    print(map[np.argmax(pred)])
    # if len(contours) > 0:
    #     maxContour = getMaxContours(contours)
    #     print((countFingers(maxContour)))
    # cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cv2.destroyAllWindows()