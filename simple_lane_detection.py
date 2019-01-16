#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import argparse as ap


def grayscale(img):

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):

    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):

    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):

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


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img):
    
    return cv2.addWeighted(initial_img, 0.8, img, 1., 0.)

def process_frame(image):
    global first_frame

    gray_image = grayscale(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

    kernel_size = 5
    gauss_gray = gaussian_blur(mask_yw_image,kernel_size)

    low_threshold = 50
    high_threshold = 150
    canny_edges = canny(gauss_gray,low_threshold,high_threshold)
    
    imshape = image.shape
    lower_left = [imshape[1]/9,imshape[0]]
    lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
    # top_left = [imshape[1]/2-imshape[1]/9,imshape[0]/2+11*imshape[0]/60]
    # top_right = [imshape[1]/2+imshape[1]/9,imshape[0]/2+11*imshape[0]/60]
    top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
    top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    roi_image = region_of_interest(canny_edges, vertices)
    #cv2.imwrite("out/roi_image_ht"+source_img,roi_image)

    rho = 2
    theta = np.pi/180

    threshold = 20
    min_line_len = 50
    max_line_gap = 200

    line_image = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)
    #cv2.imwrite("line_image_"+source_img,line_image)
    result = weighted_img(line_image, image)
    return result

parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-v', "--videoFile", help="Path to Video File")
args = vars(parser.parse_args())
if args["videoFile"]:
        source = args["videoFile"]

cap = cv2.VideoCapture(source)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
ret,frame = cap.read()
height, width = frame.shape[:2]
out = cv2.VideoWriter('./output_videos/simple_white.mp4',fourcc, 20.0, (width,height))
while(cap.isOpened()):
    ret,frame = cap.read()
    if not ret:
            print "Cannot capture frame device | CODE TERMINATION :( "
            exit()
    #cv2.imshow('in',frame)
    processed = process_frame(frame)
    #res = cv2.resize(processed,(640, 480), interpolation = cv2.INTER_CUBIC)
    cv2.imshow('out1',processed)
    out.write(processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


# for source_img in os.listdir("test_images/"):
#     image = cv2.imread("test_images/"+source_img)
#     processed = process_frame(image)
#     cv2.imwrite("out/res_ht"+source_img,processed)
