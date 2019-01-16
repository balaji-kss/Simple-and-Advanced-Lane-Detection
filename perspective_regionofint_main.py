import numpy as np
import cv2

def perspective_transform(img):
    imshape = img.shape
    #print (imshape)
    vertices = np.array([[(.55*imshape[1], 0.63*imshape[0]), (imshape[1],imshape[0]),
                       (0,imshape[0]),(.45*imshape[1], 0.63*imshape[0])]], dtype=np.float32)
    #print (vertices)
    src= np.float32(vertices)
    dst = np.float32([[0.75*img.shape[1],0],[0.75*img.shape[1],img.shape[0]],
                      [0.25*img.shape[1],img.shape[0]],[0.25*img.shape[1],0]])
    #print (dst)
    M = cv2.getPerspectiveTransform(src, dst)

    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size = (imshape[1], imshape[0]) 
    perspective_img = cv2.warpPerspective(img, M, img_size, flags = cv2.INTER_LINEAR)    
    return perspective_img, Minv

def region_of_interest(img, vertices):

    mask = np.zeros_like(img, dtype=np.uint8)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
