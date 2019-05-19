# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 23:05:44 2019

@author: YOGENDER
"""

import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

panel = np.zeros([100,400], np.uint8)
cv2.namedWindow('panel')

def nothing(x):
    pass

cv2.createTrackbar('Lower_H', 'panel',0, 255, nothing)
cv2.createTrackbar('High_H', 'panel', 43, 255, nothing)

cv2.createTrackbar('Lower_S', 'panel',71,255, nothing)
cv2.createTrackbar('High_S', 'panel', 201, 255, nothing)

cv2.createTrackbar('Lower_V', 'panel', 131, 255, nothing)
cv2.createTrackbar('High_V', 'panel', 255, 255, nothing)


while(True):
    _, frame = cap.read()
    
    
    l_h = cv2.getTrackbarPos('Lower_H', 'panel')
    u_h = cv2.getTrackbarPos('High_H', 'panel')
    
    l_s = cv2.getTrackbarPos('Lower_S', 'panel')
    u_s = cv2.getTrackbarPos('High_S', 'panel')
    
    l_v = cv2.getTrackbarPos('Lower_V', 'panel')
    u_v = cv2.getTrackbarPos('High_V', 'panel')
    
    cv2.rectangle(frame, (100,100), (400,400),(0, 255, 0), 2)
    roi = frame[100:400, 100:400]
    
    #apply Gaussian Blur
    blur = cv2.GaussianBlur(roi, (5,5),0)
    
    #convert bgr -> hsv
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, np.array([l_h,l_s,l_v]), np.array([u_h,u_s,u_v]))
    
    kernel = np.zeros((3,3), np.uint8)
    erode = cv2.erode(mask, kernel, iterations = 1)
    dilation = cv2.dilate(erode, kernel, iterations = 1)
    
    mask2 = cv2.GaussianBlur(dilation, (21,21), 0)
    
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
        drawing = np.zeros(roi.shape, np.uint8)
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
                cv2.circle(roi, far, 1, [0, 255, 0], -1)
                
            cv2.line(roi, start, end, [0,255,0], 2) 
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
    except:
        pass    
    
    cv2.imshow('gesture', frame)
   # all_image = np.hstack((drawing, roi))
    cv2.imshow('mask', mask2)
    cv2.imshow('roi', roi)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()    
   
