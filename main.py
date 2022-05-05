import cv2
import numpy as np
import matplotlib.pyplot as plt
import dm3_lib as dm3
#from PIL import Image
from Aufgabe_Catalyst import read_dm3_file, dm3_to_cv

PATH = r'Bilder/2020_11_04-Pd-aAl__5.dm3'#'Bilder/2020_10_29-Pd-gAl__11.dm3'
SCALE = 0.5
ALPHA = 5  # Contrast
BETA = -250   # Brightness
#import dm3 bild
dm3f = read_dm3_file(PATH)
img = dm3_to_cv(dm3f, SCALE, ALPHA, BETA)

#filtering

#get bg + remove noise
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
th, bw = cv2.threshold(gray, 0.45*np.amax(gray), 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
output_bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
#adaptive threshhold
gray = cv2.GaussianBlur(gray,(21,21),0)
ad_th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 2)

masked= cv2.bitwise_and(ad_th, ad_th, mask = output_bw)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
output_ad_th = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel)


#lap = cv2.Laplacian(img,cv2.CV_32F)
#output = cv2.bitwise_or(img, img, mask=morph)
#print(img)



#contours
contours, hier = cv2.findContours(output_ad_th, 1, 3)
print(len(contours))
if len(contours) != 0:
    for i in range(len(contours)):
        if 5 <= len(contours[i]) <= 400:
                rect = cv2.minAreaRect(contours[i])
                #cv2.ellipse(img, center, axis, angle, 0, 360, (255, 0, 0), 2)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                #print(box)
                #img = cv2.drawContours(img, [box], 0, (255, 255, 255), 1)


# change between images with arrow keys
i=0
frame = [img, gray, bw, output_bw, ad_th, output_ad_th]
names = ['img', 'gray', 'bw','bw out', 'adaptive thresh', 'output_ad_th', '', '', '']
while 1:
    cv2.imshow('image', frame[i])
    print(names[i])
    k = cv2.waitKeyEx(0)
    if k in (65362,2490368):  # Press upkey
        i = i + 1
        if i > len(frame) - 1:
            i = 0
    elif k in (65364, 2621440):  # Press downkey
        i = i - 1
        if i < 0:
            i = len(frame) - 1
    elif k == 27:  # Press Esc
        break
#cv2.imwrite(r'Results/image.jpg', img)
cv2.destroyAllWindows()

if __name__ == '__main__':
    pass
