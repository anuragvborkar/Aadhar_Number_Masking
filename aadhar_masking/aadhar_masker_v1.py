import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output
import argparse
import re

output_path = ''

def parse_input():
    
    parser = argparse.ArgumentParser(description='Code to mask Aadhar number in Image')
    parser.add_argument("-i", "--input", required=True, help='Path to input image')
    parser.add_argument("-o", "--output", help='Path to output image', default='output.jpeg')
    
    args = parser.parse_args()

    src = cv.imread(cv.samples.findFile(args.input))
    global output_path
    output_path = args.output
    
    if src is None:
        print('Could not open or find the image: ', args.input)
        exit(0)
    
    return src


def preprocess(img):
    
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 4)
    return img


def display(img, desc):
    
    if len(img.shape)==3:
        height, width, channels = img.shape
    else:
        height, width = img.shape
    cv.namedWindow(desc, cv.WINDOW_NORMAL)
    cv.resizeWindow(desc, 2*width//3, 2*height//3)
    cv.imshow(desc, img)
    cv.waitKey(0)
    cv.destroyWindow(desc)
    return


def is_aadhar_number(text, start_index, confidence):
    
    if ( start_index+1 == len(text) ) or (start_index+2 == len(text)):
        return 0
    
    b1, b2, b3 = text[start_index], text[start_index+1], text[start_index+2]
    
    l1 = b1.isnumeric() and len(b1)==4
    l2 = b2.isnumeric() and len(b2)==4
    l3 = b3.isnumeric() and len(b3)==4
    l4 = True
    l5 = True
    
    #Head
    if start_index-1 >= 0:
        t = text[start_index-1]
        if not (t.isnumeric() and len(t)==4):
            l4 = True
        else:
            l4 = False
    #Tail
    if start_index+3 < len(text):
        t = text[start_index+3]
        if not (t.isnumeric() and len(t)==4):
            l5 = True
        else:
            l5 = False

    if  l1 and l2 and l3 and l4 and l5:
        conf = ( confidence[start_index] + confidence[start_index+1] + confidence[start_index+2] ) / 3
        return conf
    else:
        return 0


def parse_string(text, confidence):
    
    detections = []
    for index, data in enumerate(text):
        c = is_aadhar_number(text, index, confidence)
        if c > 0:
            detections.append((index, c))
    
    return detections


def mask(img, scan, index):
    
    x, y, w, h = scan['left'][index], scan['top'][index], scan['width'][index], scan['height'][index]
    img = cv.rectangle(img, (x, y), (x+w, y+h), (255,255,255), cv.FILLED)
    x, y, w, h = scan['left'][index+1], scan['top'][index+1], scan['width'][index+1], scan['height'][index+1]
    img = cv.rectangle(img, (x, y), (x+w, y+h), (255,255,255), cv.FILLED)
    return img


src = parse_input()
img = src.copy()

img = preprocess(img)

#img = find_corners(img)
#img = perspectivecorrection(img)

scan = pytesseract.image_to_data(img, output_type=Output.DICT)
text = scan['text']
confidence = scan['conf']

detection = parse_string(text, confidence)

if len(detection) != 0:
    for d in detection:
        img = mask(src, scan, d[0])
else:
    print('No detection found')
    exit(0)

display(img, 'Final')
cv.imwrite(output_path, img)
