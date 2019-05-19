# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:08:51 2019

@author: YOGENDER
"""

import cv2 
import numpy as np
import math

cap = cv2.VideoCapture(0)

panel = np.zeros([100,800], np.uint8)
cv2.namedWindow('panel')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Defining a function that will do the detections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x+5 , y+3), (x+w+5, y+h+5), (255, 255, 255), cv2.FILLED)
# =============================================================================
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
#         for (ex, ey, ew, eh) in eyes:
#             cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
# =============================================================================
    return frame

def nothing(x):
    pass

cv2.createTrackbar('Lower_H', 'panel',0, 255, nothing)
cv2.createTrackbar('High_H', 'panel', 43, 255, nothing)

cv2.createTrackbar('Lower_S', 'panel',71,255, nothing)
cv2.createTrackbar('High_S', 'panel', 201, 255, nothing)

cv2.createTrackbar('Lower_V', 'panel', 131, 255, nothing)
cv2.createTrackbar('High_V', 'panel', 255, 255, nothing)

cv2.createTrackbar('S_row', 'panel', 0, 480, nothing)
cv2.createTrackbar('E_row', 'panel', 480, 480, nothing)


cv2.createTrackbar('S_col', 'panel',0, 640, nothing)
cv2.createTrackbar('E_col', 'panel',640, 640, nothing)

cv2.createTrackbar('radius', 'panel',70, 200, nothing)

def distance_transform(image, radius,imp ):
    dist = cv2.distanceTransform(image, cv2.DIST_L2, 3)


    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    cv2.imshow('distance transform', dist)
    _, dist = cv2.threshold(dist, 0.4,1.0 , cv2.THRESH_BINARY, (0, 255, 0))
    cv2.imshow('threshold', dist)

    # Dilate a bit the dist image
    kernel1 = np.ones((3,3), dtype=np.uint8)
    dist = cv2.dilate(dist, kernel1, (0, 255, 0))
    cv2.imshow('Peaks', dist)

    # Create the CV_8U version of the distance image
    # It is needed for findContours()
    dist_8u = dist.astype('uint8')

    # Find total markers
    _, contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center = []
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
 
        # calculate x,y coordinate of center
        if M["m00"] == 0 :
            cX = 0
            cY = 0
        else :
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        cv2.circle(dist, (cX, cY), 5, (0, 255, 0), -1)
        c = (cX, cY)
   
        center.append(c)
   
        cv2.putText(dist, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
 
        # display the image
        cv2.imshow("Image", dist)
    cv2.circle(imp,center[0],radius,[255,0,0], -1 )
    cv2.imshow('circle', imp)    
    
def convexity_defect(image,frame, lower_bound, upper_bound):
    
    
# =============================================================================
#     kernel = np.zeros((5,5), np.uint8)
#     cv2.erode(image, kernel, iterations = 2)
#     dilation = cv2.dilate(image, kernel, iterations = 2)
#     #img_gray = cv2.pyrMeanShiftFiltering(image, 100, 151 )
#     #morphology = cv2.morphologyEx(image,cv2.MORPH_OPEN, kernel)
#     #img_gray = cv2.cvtColor(dilation,cv2.COLOR_BGR2GRAY)
#     #img_gray = cv2.cvtColor(morphology,cv2.COLOR_BGR2GRAY)
#     #img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
# =============================================================================
    kernel = np.zeros((3,3), np.uint8)
    erode = cv2.erode(image, kernel, iterations = 1)
    dilation = cv2.dilate(erode, kernel, iterations = 1)
    
    mask2 = cv2.GaussianBlur(dilation, (35,35), 0)
    
    #thresholding
    
    _, thresh = cv2.threshold(mask2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    
    # findContour
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count_defects = 0
    try :
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
    
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0),0)
        # cropping an image
        roi1 = frame[y:y + h+100, x:x + w+100]
        #apply Gaussian Blur
        blur = cv2.GaussianBlur(roi1, (5,5),0)
    
        #convert bgr -> hsv
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
        mask = cv2.inRange(hsv, lower_bound, upper_bound )
    
        kernel = np.zeros((3,3), np.uint8)
        erode = cv2.erode(mask, kernel, iterations = 1)
        dilation = cv2.dilate(erode, kernel, iterations = 1)
    
        mask2 = cv2.GaussianBlur(dilation, (41,41), 0)
    
        #thresholding
    
        _, thresh = cv2.threshold(mask2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    
        # findContour
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        try :
             cnt = max(contours, key = lambda x: cv2.contourArea(x))
        
        
             x, y, w, h = cv2.boundingRect(cnt)
             cv2.rectangle(roi, (x, y), (x + w, y + h ), (0,255,0), 2)
        
        
             # convex hull
             hull = cv2.convexHull(cnt)
        
             # draw contour and hull
             drawing = np.zeros(roi1.shape, np.uint8)
             cv2.drawContours(drawing, [cnt], -1, (0, 255, 0), 2)
             cv2.drawContours(drawing, [hull], -1, (0,0,255), 2)
        
        
             # convexity defect 
             hull = cv2.convexHull(cnt, returnPoints = False)
             defects = cv2.convexityDefects(cnt, hull)
        
        
             # count defect
             count_defects = 0
        
        
             for i in range(len(defects)):
                s, e, f, d = defects[i,0]
            
                start = tuple(cnt[s][0])
            
                end = tuple(cnt[e][0])
            
                far = tuple(cnt[f][0])
            
            
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            
                angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
            
                if angle <= 90 :
                   count_defects += 1
                   cv2.circle(roi1, far, 1, [0, 255, 0], -1)
            
                cv2.line(roi1, start, end, [0,255,0], 2)
            # Print number of fingersx
             if count_defects == 0:
                  cv2.putText(frame, "ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
             elif count_defects == 1:
                  cv2.putText(frame, "TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
             elif count_defects == 2:
                  cv2.putText(frame, "THREE", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
             elif count_defects == 3:
                  cv2.putText(frame, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
             elif count_defects == 4:
                  cv2.putText(frame, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
             else:
                 pass    
            
        except :
             pass
        
# =============================================================================
#         hull_list = []
#         far_list = []
#         defects_list = []
#         for i in range(len(cnt)):
#             hull = cv2.convexHull(cnt[i], returnPoints = False)
#             hull_list.append(hull)
#             defects = cv2.convexityDefects(cnt[i],hull)
#             defects_list.append(defects)
#             for j in range (len(defects_list)):
#                if type(defects_list[j]) != type(None):
#                    for i in range(defects_list[j].shape[0]):
#                        s,e,f,d = defects_list[j][i,0]
#                        
#                        start = tuple(cnt[s][0])
#                     
#                        end = tuple(cnt[e][0])
#                     
#                        far = tuple(cnt[f][0])
#                        far_list.append(far)
#                        
#                        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
#                        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
#                        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
#                        angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
#                        
#                        if angle <= 90 :
#                            count_defects += 1
#                            cv2.circle(image, far, 1, [0, 255, 0], -1)
#                 
#                        cv2.line(image, start, end, [0,255,0], 2)
# =============================================================================
    except :
          pass                
                       
                       
    
    #cv2.line(image,start,end,[0,255,0],2)
    #cv2.circle(image,far,5,[0,0,255],-1)
    
    #all_image = np.hstack((drawing, roi1))
    cv2.imshow('gesture', frame)
    cv2.imshow('frame', image)
    
    #cv2.imshow('roi', all_image)
    cv2.imshow('mask',mask2)
     
 
        


while(True):
    _, frame = cap.read()
    
     
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    
    l_h = cv2.getTrackbarPos('Lower_H', 'panel')
    u_h = cv2.getTrackbarPos('High_H', 'panel')
    
    l_s = cv2.getTrackbarPos('Lower_S', 'panel')
    u_s = cv2.getTrackbarPos('High_S', 'panel')
    
    l_v = cv2.getTrackbarPos('Lower_V', 'panel')
    u_v = cv2.getTrackbarPos('High_V', 'panel')
    
    s_row = cv2.getTrackbarPos('S_row', 'panel')
    e_row = cv2.getTrackbarPos('E_row', 'panel')
    s_col = cv2.getTrackbarPos('S_col', 'panel')
    e_col = cv2.getTrackbarPos('E_col', 'panel')
    radius = cv2.getTrackbarPos('radius', 'panel')
    #face = frame.copy() 
    roi = frame[s_row:e_row, s_col:e_col]
    #face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    #detect = detect(face_gray, face)
    #cv2.imshow('face', detect)
    blur = cv2.GaussianBlur(roi, (5,5),0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    lower_bound = np.array([l_h,l_s,l_v])
    upper_bound = np.array([u_h,u_s,u_v])
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    #mask[np.all(mask == 255)] = 0
    cv2.imwrite('hand2.jpg', mask)
    imp = cv2.imread('hand2.jpg')
    imp[np.all(imp == 255)] = 0
    
   
    kernel = np.zeros((5,5), np.uint8)
    erode = cv2.erode(mask, kernel, iterations = 1)
    dilation = cv2.dilate(erode, kernel, iterations = 1)
    morphology = cv2.morphologyEx(imp,cv2.MORPH_OPEN, kernel)
    
    blurred = cv2.GaussianBlur(dilation, (11,11), 0)
    
    gray = cv2.cvtColor(morphology, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.pyrMeanShiftFiltering(hsv, 21, 22 )
    #cv2.imshow('pyrMean',blurred)
    #_, bw = cv2.threshold(imgREsult, 150,255, cv2.THRESH_BINARY)

    distance_transform(gray, radius,imp.copy())
    
    convexity_defect(mask, frame, lower_bound, upper_bound)
    
    
    
    
    
    
    
    
    #cv2.imshow('convexity', convexity_def)
    #cv2.imshow('frame', mask)
    cv2.imshow('panel', panel)
    
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
cap.release()
cv2.destroyAllWindows()    
    