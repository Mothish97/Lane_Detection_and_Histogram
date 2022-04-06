# -*- coding: utf-8 -*-
"""
@author: mothi
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_frame():
    vidcap = cv2.VideoCapture('whiteline.mp4')
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


def  get_masked_img(img):    
    polygon = np.array([[100,520], [920,520], [520,330], [440,330]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, np.array([polygon], np.int32), (255,255,255))
    img_masked = cv2.bitwise_and(img,mask)
    
    return img_masked

def get_lane_img(img,img_thresh):
    lines=cv2.HoughLinesP(img_thresh,1,threshold=20,theta=np.pi/180,minLineLength=2,maxLineGap=300,lines=np.array([]))
    slope_long=0
    count=0
    img_cpy = np.zeros(img.shape,dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if(count==0):
            slope_long=( y2-y1) /( x2-x1)      
        slope=( y2-y1) / (x2-x1)
        if(slope_long*slope > 0 ):
            cv2.line(img_cpy, (x1, y1), (x2, y2), (0, 255, 0), 3)
        elif(slope_long*slope <0):
            cv2.line(img_cpy, (x1, y1), (x2, y2), (0, 0, 255), 3)
        count+=1
                  
    img_lane = cv2.addWeighted(img,0.9,img_cpy,1,0.0)
    
    return img_lane

if __name__=='__main__':
    #get_frame()
    
    #img = cv2.imread("frame0.jpg")
    
    vidcap = cv2.VideoCapture('whiteline.mp4')
    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))
    size=(frame_width,frame_height)
    result = cv2.VideoWriter('LaneDetectedVideo.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)
    while True:
        success,frame = vidcap.read()
        if not success:
            print("Stream ended..")
            break
        img=frame
        img_bw= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        img_masked= get_masked_img(img_bw)
        
        ret, img_thresh = cv2.threshold(img_masked, 130, 145, cv2.THRESH_BINARY)
        
        img_lane=get_lane_img(img,img_thresh)
        result.write(img_lane)
        #plt.imshow(img_lane, cmap= "gray")
        cv2.imshow('frame', np.uint8(img_lane))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    vidcap.release()
    result.release()
    #plt.imshow(img_lane, cmap= "gray")
    
    
    print("Code")
    