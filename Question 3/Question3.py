
import cv2
import numpy as np
import matplotlib.pyplot as plt



'''
Getting Threshold Image
'''

def get_threshold_image(img):
    img_binary = cv2.threshold(img, 20, 100, cv2.THRESH_BINARY)
    img_canny = cv2.Canny(img, 20, 100)
    
    #combining binary and canny image
    img_combined_binary_canny = np.zeros_like(img_canny)
    img_combined_binary_canny[((img_binary == 1)) | ((img_canny > 0))] = 1
    
    #getting saturation image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_saturation = hls[:,:,2]

    #thresholding saturation image
    img_saturation_binary = img_binary = cv2.threshold(img_saturation, 150, 255, cv2.THRESH_BINARY)


    # Combine binary and canny and saturation thresholded image
    img_threshold = np.zeros_like(img_combined_binary_canny)
    
    img_threshold[(img_saturation_binary == 1) | (img_combined_binary_canny == 1)] = 1
        
    return img_threshold


'''
Getting Warped and Perspective transformed image
'''
def get_transform_image(img): 
    
    src = np.float32([[568,470], [260,680], [717,470], [1043,680]])
    dst = np.float32([[200,0], [200,680], [1000,0], [1000,680]])
    
    # Given src and dst points, calculate the perspective transform matrix
    img_perspective = cv2.getPerspectiveTransform(src, dst)

    # Warp the image using OpenCV warpPerspective()
    img_warped = cv2.warpPerspective(img, img_perspective, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
    
    return img_warped, img_perspective


'''
Gets the poly fit or the curve equation 
'''
def get_poly_fit(img):
     
    grid = 9   
    #Getting histogram by getting npsum for each column
    histogram = np.sum(img[int(img.shape[0]/2):,:], axis=0)

    #To find start of left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    left_max = np.argmax(histogram[:midpoint])
    right_max = np.argmax(histogram[midpoint:]) + midpoint
    
    
    #Adding threshold of 100 to right and left max intensity to get the start and end of the left and right lines in x
    left_line_start_x = left_max - 100
    left_line_end_x = left_max + 100
    right_line_start_y = right_max - 100
    right_line_end_x = right_max + 100
    
    # Set height of windows
    window_height = np.int(img.shape[0]/grid)
     
    
    # Identify the x and y positions of all nonzero pixels in the image
    img_nonzero = img.nonzero()
    img_nonzero_y = np.array(img_nonzero[0])
    img_nonzero_x = np.array(img_nonzero[1])


    left_lane = []
    right_lane = []
    

    # Step through the windows one by one
    for i in range(grid):
        
        # Identify the low and highest position of grid in y direction 
        low_y = img.shape[0] - (i+1)*window_height
        high_y = img.shape[0] - i*window_height


        # Identify the nonzero pixels in x within the window
        left_index = ((img_nonzero_y >= low_y) & (img_nonzero_y < high_y) & 
        (img_nonzero_x >= left_line_start_x) &  (img_nonzero_x < left_line_end_x)).nonzero()[0]
        
        # Identify the nonzero pixels in y within the window
        right_index = ((img_nonzero_y >= low_y) & (img_nonzero_y < high_y) & 
        (img_nonzero_x >= right_line_start_y) &  (img_nonzero_x < right_line_end_x)).nonzero()[0]
        
        
        # Append these indices to the lists
        left_lane.append(left_index)
        right_lane.append(right_index)
        


    #Combine all the grind into one
    left_lane = np.concatenate(left_lane)
    right_lane = np.concatenate(right_lane)

    # Extract left and right line pixel positions
    leftx = img_nonzero_x[left_lane]
    lefty = img_nonzero_y[left_lane] 
    rightx = img_nonzero_x[right_lane]
    righty = img_nonzero_y[right_lane] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    

    
    
    return left_fit, right_fit,get_warped_lane_image(left_fit,right_fit,left_lane,right_lane,img_nonzero_x,img_nonzero_y,img)

'''
Get the lane image with marking
'''
def get_warped_lane_image(left_fit,right_fit,left_lane,right_lane,img_nonzero_x,img_nonzero_y,img):

    #Get the curve equation
    variable = np.linspace(0, img_warped.shape[0]-1, img_warped.shape[0] )
    left_equation = left_fit[0]*variable**2 + left_fit[1]*variable + left_fit[2]
    right_quation = right_fit[0]*variable**2 + right_fit[1]*variable + right_fit[2]
    
    
    margin=100
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((img, img, img))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[img_nonzero_y[left_lane], img_nonzero_x[left_lane]] = [255, 0, 0]
    out_img[img_nonzero_y[right_lane], img_nonzero_x[right_lane]] = [0, 0, 255]
    
    left_line_window1 = np.array([np.transpose(np.vstack([left_equation-margin, variable]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_equation+margin, 
                                  variable])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_quation-margin, variable]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_quation+margin, 
                                  variable])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    return result
    
    
    
'''
Get the radius of curvature
'''

def get_radius_curvature(img_warped, left_fit, right_fit):
    
    #Get the curve equation
    variable = np.linspace(0, img_warped.shape[0]-1, img_warped.shape[0] )
    left_equation = left_fit[0]*variable**2 + left_fit[1]*variable + left_fit[2]
    right_quation = right_fit[0]*variable**2 + right_fit[1]*variable + right_fit[2]
    
    #Conversion from pixel to meters
    y_meters = 50/720 
    x_meters = 30/700 
    y_eval = np.max(variable)
    
    #Get polynomial equation according to the meter
    left_fit_meter = np.polyfit(variable*y_meters, left_equation*x_meters, 2)
    right_fit_meter = np.polyfit(variable*y_meters, right_quation*x_meters, 2)
    
    # Calculate the new radii of curvature
    left_curvature =  ((1 + (2*left_fit_meter[0] *y_eval*y_meters + left_fit_meter[1])**2) **1.5) / np.absolute(2*left_fit_meter[0])
    right_curvature = ((1 + (2*right_fit_meter[0]*y_eval*y_meters + right_fit_meter[1])**2)**1.5) / np.absolute(2*right_fit_meter[0])
    
    
    # Now our radius of curvature is in meters
    return left_curvature, right_curvature





'''
Put the markin and diplay the radius of curvature
'''
def get_final_image(img, img_warped, left_fit, right_fit, img_perspective, left_curvature, right_curvature):
    
    #Get the Equation 
    variable = np.linspace(0, img_warped.shape[0]-1, img_warped.shape[0] )
    left_equation = left_fit[0]*variable**2 + left_fit[1]*variable + left_fit[2]
    right_quation = right_fit[0]*variable**2 + right_fit[1]*variable + right_fit[2]


    img_cpy = np.zeros(img.shape,dtype=np.uint8)


    pts_left = np.array([np.transpose(np.vstack([left_equation, variable]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_quation, variable])))])
    pts = np.hstack((pts_left, pts_right))

    # Fill the lane with fillpoly
    cv2.fillPoly(img_cpy, np.int_([pts]), (0,0, 255))
    Minv = np.linalg.inv(img_perspective)
    
    # Warp back to the original image
    original_image_warped = cv2.warpPerspective(img_cpy, Minv, (img.shape[1], img.shape[0])) 

    result = cv2.addWeighted(img, 1, original_image_warped, 0.3, 0)
    
    cv2.putText(result, 'Left curvature: {:.0f} m'.format(left_curvature), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Right curvature: {:.0f} m'.format(right_curvature), (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

        
    return result,img_cpy


if __name__=="__main__":
    
    
    vidcap = cv2.VideoCapture('challenge.mp4')
    img = cv2.imread('frame0.jpg')
    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))
    size=(frame_width,frame_height)
    outputVideo = cv2.VideoWriter('RadiusCurvatureResult.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)
    
    while True:
        success,frame = vidcap.read()
        if not success:
            print("Stream ended..")
            break
        img=frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_threshold = get_threshold_image(img)
        img_warped, img_perspective = get_transform_image(img_threshold) 

        left_fit, right_fit, result_warped  = get_poly_fit(img_warped) 
    
        left_curvature, right_curvature = get_radius_curvature(img_warped, left_fit, right_fit)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result,img_warped_marking= get_final_image(img, img_warped, left_fit, right_fit, img_perspective, left_curvature, right_curvature) 

        
        x=int(result.shape[1]/2)
        y=int(result.shape[0]/2)

        # concatanate image Horizontally
        Hori1 = np.concatenate((cv2.resize(result, (x,y)), cv2.resize(result_warped, (x,y))), axis=1)
        Hori2 = np.concatenate((cv2.resize(img, (x,y)), cv2.resize(img_warped_marking, (x,y))), axis=1)
          
        # concatanate image Vertically
        Verti = np.concatenate((Hori1, Hori2), axis=0)
        outputVideo.write(Verti)
        cv2.imshow('Frames', Verti)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    vidcap.release()
    outputVideo.release()


    print ("code")
    
    
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        
    # img_threshold = get_threshold_image(img)
    
    
    # img_warped, img_perspective = get_transform_image(img_threshold) 

    
    
    # left_fit, right_fit, result_warped  = get_poly_fit(img_warped) 
    
    
    # left_curvature, right_curvature = get_radius_curvature(img_warped, left_fit, right_fit)
        
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # result,img_warped_marking= get_final_image(img, img_warped, left_fit, right_fit, img_perspective, left_curvature, right_curvature) 
    
    # plt.imshow(result,cmap='gray')
    
    