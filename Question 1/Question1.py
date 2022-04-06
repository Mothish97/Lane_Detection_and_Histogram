# -*- coding: utf-8 -*-
"""

@author: mothi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

import os

'''
Get the histogram for flattened image array
'''
def get_histogram(image, hist_length):
    histogram = np.zeros(hist_length)
    
    for pixel in image:
        histogram[pixel] += 1
    return histogram
'''
# create our cumulative sum function
'''
def get_cumulative(histogram):
    
    cumulative_sum = [histogram[1]]
    for i in histogram:
        cumulative_sum.append(cumulative_sum[-1] + i)
    
        
    cumulative_sum= np.array(cumulative_sum)
    # Normalize
    cumulative_sum = ((cumulative_sum - cumulative_sum.min()) * 255) / (cumulative_sum.max() - cumulative_sum.min())        
        
    
    return cumulative_sum



def getHistogramEqualizedImage(img_flat,img):
 
    # execute our histogram function
    histogram = get_histogram(img_flat, 256)
    
    # execute the fn
    cumulative_sum = get_cumulative(histogram)
    
    cumulative_sum = cumulative_sum.astype('uint8')

    img_histogram_equalized = cumulative_sum[img_flat]
    
    img_histogram_equalized = np.reshape(img_histogram_equalized, img.shape)
    
    return img_histogram_equalized
    

def getAdaptiveHistogramImage(img):
    grid_size_x=10
    grid_size_y=8
    
    img_size_x = img.shape[0]
    img_size_y = img.shape[1]

    pixels_in_one_grid_x =  int(img_size_x/grid_size_x)
    pixels_in_one_grid_y =  int(img_size_y/grid_size_y)
    img_adap=img.copy()
    
    
    for i in range(0, grid_size_x, 1):
        for j in range(0, grid_size_y, 1):
            grid = img[i*pixels_in_one_grid_x:(i+1)*pixels_in_one_grid_x, j*pixels_in_one_grid_y:(j+1)*pixels_in_one_grid_y]
            grid_flat= grid.flatten()
            grif_histogram_equalized= getHistogramEqualizedImage(grid_flat,grid)
            img_adap[i*pixels_in_one_grid_x:(i+1)*pixels_in_one_grid_x, j*pixels_in_one_grid_y:(j+1)*pixels_in_one_grid_y]=grif_histogram_equalized
    

    return img_adap
            

if __name__== "__main__":
    path = "adaptive_hist_data/"
    dir_list = os.listdir(path)
    img_input=  cv2.imread("adaptive_hist_data/0000000000.png")
    frame_width = int(img_input.shape[1])
    frame_height = int(img_input.shape[0])
    size=(frame_width,frame_height)
    result_adaptive = cv2.VideoWriter('Adaptive Histogram.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)
    result_histogram = cv2.VideoWriter('Histogram.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)
    for file in dir_list :
        img_file= "adaptive_hist_data/"+ str(file)
        
        img_input=  cv2.imread(img_file)
        
        img_input= cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        
        img_flat = img_input.flatten()
        
        img_histogram_equalized = getHistogramEqualizedImage(img_flat,img_input)
        
        
        
        img_adap=  getAdaptiveHistogramImage(img_input)
        
        result_adaptive.write(img_adap)
        result_histogram.write(img_histogram_equalized)
        print("a")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    result_adaptive.release()
    result_histogram.release()
        #cv2.imshow("a",img_adap)
        

    



    
    
    

    # fig= plt.figure(figsize=(12,12))
    # ax1= fig.add_subplot(2,2,1)
    # ax1.imshow(img_input)
    # ax1.title.set_text('Input Image')
    # ax2= fig.add_subplot(2,2,2)
    # ax2.imshow(img_histogram_equalized)
    # ax2.title.set_text('Histogram Equalized Image') 
    # ax3 = fig.add_subplot(2,2,3)
    # ax3.imshow(img_adap)
    # ax3.title.set_text('Cumulative Histogran')

    


