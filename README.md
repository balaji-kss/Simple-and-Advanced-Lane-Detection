
# **Simple and Advanced Lane Detection**

### Objective
To create a software pipeline to identify lane boundaries in a video from a front facing
camera on a car. ​ This assignment was done as a part of our college Self Driving Car
project.

#### Simple Lane Detection pipeline

1. Convert RGB image to grayscale image.
2. Convert RGB space to HSV space and identify yellow colour.
3. Apply Gaussian blur to remove noise.
4. Use Canny Edge Detection to get the edges in the image.
5. A polygon is chose as the region of interest
	lower_left = [imshape[1]/9,imshape[0]]
	lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
	top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
	top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
6. Used hough lines in the region of interest and merged the resultant lines with the input image.

#### Advanced Lane Detection pipeline

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms and gradients to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit a polynomial expression to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Overlay the detected lane boundaries back onto the original image.

#### Dependencies & my environment


* Python3.5
* Jupyter Notebook, Numpy, OpenCV 3.0 
* Matplotlib, glob
* IPython, pickle
* scikit-image

#### How to compile and run the code

(1) Open the **advanced_lane.ipynb** and run !!

(2) To see ​ simple lane detection​ results, run the script
```sh
python simple_lane_detection.py -v input_videos/white.mp4
```

(3) The input video is taken from ​ input_videos​ folder and the processed video is saved in the ​ output_videos​ folder in the name ​ simple_white.mp4​ .
