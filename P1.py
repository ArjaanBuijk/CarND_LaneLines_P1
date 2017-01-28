##############################################################################
# Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:
# cv2.inRange() for color selection
# cv2.fillPoly() for regions selection
# cv2.line() to draw lines on an image given endpoints
# cv2.addWeighted() to coadd / overlay two images cv2.cvtColor() to grayscale or change color cv2.imwrite() to output images to file
# cv2.bitwise_and() to apply a mask to an image
##############################################################################


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import math

# these sizes are in pixels
hood_size=0
roi_height=0        # roi = region of interest
roi_width_bot=0
roi_width_top=0
roi_shift_bot=0         # sideways shift from center of image
roi_shift_top=0

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
        
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_lanes(img, roi_vertices, lines, color=[255, 0, 0], thickness=None):
    """
    based on draw_lines
    """
    #
    # extrapolate start and end points for each line to boundary of region of interest
    #
    y_top_roi    = roi_vertices[0][1][1]
    y_bot_roi    = roi_vertices[0][0][1]
    
    x_top_roi_min   = roi_vertices[0][1][0]
    x_top_roi_max   = roi_vertices[0][2][0]
    x_bot_roi_min   = roi_vertices[0][0][0]
    x_bot_roi_max   = roi_vertices[0][3][0]
    
    x_tops_left   = []
    x_bots_left   = []
    
    x_tops_right   = []
    x_bots_right   = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
        
            # fit a line (y=Ax+B) through this line
            # np.polyfit() returns the coefficients [A, B] of the fit
            fit_line = np.polyfit((x1, x2), (y1, y2), 1)
            A, B = fit_line
            
            # skip lines with very small slope (A)
            if abs(A) < 0.1:
                continue
            
            # calculate intersection with top & bottom boundary of region of interest
            x_top = (y_top_roi - B) / A
            x_bot = (y_bot_roi - B) / A
            
            # skip this line if either the top or bottom points are outside region of interest
            if x_top < x_top_roi_min or x_top > x_top_roi_max:
                continue
            if x_bot < x_bot_roi_min or x_bot > x_bot_roi_max:
                continue
            
            if A < 0:
                x_tops_left.append(x_top)
                x_bots_left.append(x_bot)
            else:
                x_tops_right.append(x_top)
                x_bots_right.append(x_bot)
            
            # uncomment this if you want to check what lines are maintained
            #thickness = 2
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    # if we didn't find a top or bottom, then skip this image
    if (len(x_tops_left) == 0 or
        len(x_bots_left) == 0 or
        len(x_tops_right) == 0 or
        len(x_bots_right) == 0 ):
        print ('Not all end points found --> skipping this image !!')
        return
    
    x_top_left_average  = int( np.array(x_tops_left).mean()  )
    x_top_right_average = int( np.array(x_tops_right).mean() )
    x_bot_left_average  = int( np.array(x_bots_left).mean()  )
    x_bot_right_average = int( np.array(x_bots_right).mean() )
    
    x_top_left_width  = max(x_tops_left)  - min(x_tops_left)
    x_top_right_width = max(x_tops_right) - min(x_tops_right)
    x_bot_left_width  = max(x_bots_left)  - min(x_bots_left)
    x_bot_right_width = max(x_bots_right) - min(x_bots_right)
    
    if thickness == None:
        thickness = int( 0.5*sum( [x_top_left_width ,
                                   x_top_right_width,
                                   x_bot_left_width ,
                                   x_bot_right_width] ) / 4 )
    
    # plot left lane                 
    cv2.line(img, (x_bot_left_average, y_bot_roi), 
                  (x_top_left_average, y_top_roi), color, thickness)
    
    # plot right lane                 
    cv2.line(img, (x_bot_right_average, y_bot_roi), 
                  (x_top_right_average, y_top_roi), color, thickness)

    
    #for line in lines:
    #    for x1,y1,x2,y2 in line:
    #        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def draw_lanes_on_image(file_in, file_out, show_all=True, show_final=True):
    ####################################################################
    #reading in an image
    image = mpimg.imread(file_in)
    print ("******************************************************************")
    print('Processing image from file: '+file_in)
    combo = process_image(image, show_all, show_final)
    
    plt.savefig(file_out)
    print('Processed image saved as: ' + file_out)

def process_image(image, show_all=False, show_final=False):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    global frame
    frame += 1
    print('Frame : ', str(frame),'-', type(image), 'with dimesions:', image.shape)
    if show_all:
        plt.title('Original Image')
        plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
        plt.show()
    
    ####################################################################
    # Make it gray scale
    gray = grayscale(image)
    if show_all:
        plt.title('Grayscale Image')
        plt.imshow(gray, cmap='gray')
        plt.show()
    
    ####################################################################
    # Next we'll create a masked image with only region of interest
    imshape = image.shape

    # base roi on ratio of image
    x_bot_min_ratio = 0.156
    x_bot_max_ratio = 0.950
    x_top_min_ratio = x_bot_min_ratio
    x_top_max_ratio = x_bot_max_ratio
    
    # height of roi region
    if roi_height == 0:
        height = int( 0.4*imshape[0] )
    else:
        height = roi_height
        
    # bottom width of roi region
    if roi_width_bot == 0:
        width_bot = int( 0.80 * imshape[1] )
    else:
        width_bot = roi_width_bot
        
    # top width of roi region
    if roi_width_top == 0:
        width_top = int( 0.20 * imshape[1] )
    else:
        width_top = roi_width_top
    
    
    x_bot_min = roi_shift_bot + int( 0.5 * (imshape[1] - width_bot) )
    x_bot_max = roi_shift_bot + int( 0.5 * (imshape[1] + width_bot) )
    x_top_min = roi_shift_top + int( 0.5 * (imshape[1] - width_top) )
    x_top_max = roi_shift_top + int( 0.5 * (imshape[1] + width_top) )
    
    # eliminate car hood from picture
    y_bot = imshape[0] - hood_size
    y_top = y_bot - height
    
    roi_vertices = np.array([[(x_bot_min,y_bot),
                              (x_top_min, y_top), 
                              (x_top_max, y_top), 
                              (x_bot_max,y_bot)]], dtype=np.int32)
    
    if show_all:
        plt.title('Region of interest on Image. Wb,Wt,H,Sb,St,Sv='+
                  str(x_bot_max-x_bot_min)+','+
                  str(x_top_max-x_top_min)+','+
                  str(y_top-y_bot)+','+
                  str(roi_shift_bot)+','+
                  str(roi_shift_top)+','+
                  str(hood_size))
        line_img = np.zeros((image.shape[0], image.shape[1], 3), 
                            dtype=np.uint8)
        p=roi_vertices[0]
        roi_lines = np.array([[[ p[0][0], p[0][1], p[1][0], p[1][1] ]],
                              [[ p[1][0], p[1][1], p[2][0], p[2][1] ]],
                              [[ p[2][0], p[2][1], p[3][0], p[3][1] ]],
                              [[ p[3][0], p[3][1], p[0][0], p[0][1] ]]], 
                             dtype=np.int32)
        draw_lines(line_img, roi_lines)
        combo = weighted_img(line_img, image, α=0.8, β=1., λ=0.)
        plt.imshow(combo)
        plt.show()
    
    masked_gray = region_of_interest(gray, roi_vertices)

    # Display the image
    if show_all:
        plt.title('Grayscale with Region of Interest')
        plt.imshow(masked_gray, cmap='Greys_r')
        plt.show()
    
    ####################################################################
    # Set gray level in region of interest to black
    #  if it is darker than the average in the roi.
    # 
    # I did this to get challenge_frame_110 to work properly:
    # (-) The road is concrete (light)
    # (-) There are black tire skid marks that are detected by canny and
    #      mess it all up..
    #
    # find the average gray level in roi
    
    gray_average = int( np.average(masked_gray[np.nonzero(masked_gray)]) )
    
#     for i in range(masked_gray.shape[0]):
#         for j in range(masked_gray.shape[1]):
#             if masked_gray[i][j] < gray_average:
#                 masked_gray[i][j] = gray_average 
    # fast solution: http://stackoverflow.com/questions/19666626/replace-all-elements-of-python-numpy-array-that-are-greater-than-some-value
    masked_gray[ masked_gray < gray_average ] = gray_average
    
    # Display the image
    if show_all:
        plt.title('Equalized gray level')
        plt.imshow(masked_gray, cmap='Greys_r')
        plt.show()
    
    ####################################################################
    # Find the edges, with canny edge detection
    # See: http://homepages.inf.ed.ac.uk/rbf/HIPR2/canny.htm
    #
    # Define a kernel size for Gaussian smoothing / blurring
    # Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
    kernel_size = 5
    gaussian_blur(image, kernel_size)
    
    # Define parameters for Canny and run it
    # thresholds between 0 and 255
    low_threshold = 50
    high_threshold = 150
    #low_threshold = 50
    #high_threshold = 75
    edges = canny(masked_gray, low_threshold, high_threshold)
    
    # Display the image
    if show_all:
        plt.title('Canny Edges')
        plt.imshow(edges, cmap='Greys_r')
        plt.show()
    
    
    
    ####################################################################
    # Find the lanes, using hough transform
    #
    # Define the Hough transform parameters
    rho             = 2          # distance resolution in pixels of the Hough grid
    theta           = np.pi/180  # angular resolution in radians of the Hough grid
    threshold       = 15         # minimum number of votes (intersections in Hough grid cell)
    min_line_len    = 10         # minimum number of pixels making up a line
    max_line_gap    = 5          # maximum gap in pixels between connectable line segments
    
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None :
        print ('ERROR: No hough lines found')
        return image
    
    if show_all:
        plt.title('Hough Lines on Image')
        line_img = np.zeros((edges.shape[0], edges.shape[1], 3), 
                            dtype=np.uint8)
        draw_lines(line_img, lines)
        combo = weighted_img(line_img, image, α=0.8, β=1., λ=0.)
        plt.imshow(combo)
        plt.show()

    # Draw the lanes on the original image    
    line_img = np.zeros((edges.shape[0], edges.shape[1], 3), 
                        dtype=np.uint8)
    draw_lanes(line_img, roi_vertices, lines, thickness=10)
    combo = weighted_img(line_img, image, α=0.8, β=1., λ=0.)
    if show_all:
        plt.title('Final Image with Lanes')
        plt.imshow(combo)
        plt.show()

    return combo

if __name__ == '__main__':
    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip
    from IPython.display import HTML
    
    ################################################################
    # process some test files
    #
    file_dir = "test_images"
    files = ['solidWhiteCurve.jpg',
             'solidWhiteRight.jpg',
             'solidYellowCurve.jpg',
             'solidYellowCurve2.jpg',
             'solidYellowLeft.jpg',
             'whiteCarLaneSwitch.jpg']
 
    frame = 0
    hood_size=0
    roi_height=200
    roi_width_bot=810
    roi_width_top=150
    roi_shift_bot=45
    roi_shift_top=15
    for file in files:
        frame = 0
        draw_lanes_on_image(file_in=file_dir+"/"+file, 
                            file_out=file_dir+"/with_lanes_"+file,
                            show_all=False, show_final=False)
          
    ################################################################
    # process some test videos
    #  
    frame = 0
    hood_size=0
    roi_height=200
    roi_width_bot=810
    roi_width_top=150
    roi_shift_bot=45
    roi_shift_top=15
    white_output = 'white.mp4'
    clip1 = VideoFileClip("solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
      
    frame = 0
    hood_size=0
    roi_height=200
    roi_width_bot=810
    roi_width_top=150
    roi_shift_bot=45
    roi_shift_top=15
    yellow_output = 'yellow.mp4'
    clip2 = VideoFileClip('solidYellowLeft.mp4')
    yellow_clip = clip2.fl_image(process_image)
    yellow_clip.write_videofile(yellow_output, audio=False)
 
     
    

    frame = 0
    hood_size=55
    roi_height=200
    roi_width_bot=840
    roi_width_top=180
    roi_shift_bot=35
    roi_shift_top=25
    
    challenge_output = 'extra.mp4'
    clip2 = VideoFileClip('challenge.mp4')
    challenge_clip = clip2.fl_image(process_image)
    challenge_clip.write_videofile(challenge_output, audio=False)
    
#     ####
#     # test individual frames for this video, extracted with ffmpeg
#     file_dir = "movie_frames"
#     files = ['challenge-110.jpg']
#     for file in files:
#         frame = 0
#         draw_lanes_on_image(file_in=file_dir+"/"+file, 
#                             file_out=file_dir+"/with_lanes_"+file,
#                             show_all=True, show_final=True)
    
    
    
