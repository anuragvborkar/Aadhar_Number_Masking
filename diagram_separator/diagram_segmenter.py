import cv2
import numpy as np
import sys
import statistics

original = None


def load_and_preprocess(path):

    global original
    
    original = cv2.imread(path)
    image = original.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    return thresh


def filter_diagrams(img):

    global original

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    dilate = cv2.dilate(img, kernel, iterations=1) # Dilate needed to smudge different parts of same diagram together
    
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:] #Discrepancy in documentation, hence the [-2:]
    
    h_list = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        h_list.append(h)
    
    h_mode = statistics.mode(h_list)
    
    base_path = sys.argv[1].split('.')[0] + '_diag_'
    index = 1
    pad = 18
    for c in contours:
        
        x, y, w, h = cv2.boundingRect(c)
        height, width, channels = original.shape

        condition_1 = h > h_mode + 10
        # condition_2 = w < 2 * h
        condition_3 = h < height/10 or w < width/10

        if condition_1 and not condition_3:
            roi = original[y-pad:y+h+pad, x-pad:x+w+pad]
            display(roi, 'Diagram_'+str(index))

            path = base_path + str(index) + '.jpeg'
            index += 1
            cv2.imwrite(path, roi)

            original = cv2.rectangle(original, (x-pad, y-pad), (x+w+pad, y+h+pad), (255, 255, 255), cv2.FILLED)
    
    return original


def display(img, desc):
    
    cv2.imshow(desc, img)
    cv2.waitKey(0)
    cv2.destroyWindow(desc)

    return

img = load_and_preprocess(sys.argv[1])
display(original, 'Original')
img = filter_diagrams(img)
qpath = sys.argv[1].split('.')[0] + '_question.jpeg'
cv2.imwrite(qpath, img)
display(img, 'Question')
