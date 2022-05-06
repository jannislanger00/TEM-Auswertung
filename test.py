import cv2
import numpy as np
import matplotlib.pyplot as plt
import dm3_lib as dm3
#from PIL import Image
from Aufgabe_Catalyst import read_dm3_file, dm3_to_cv

PATH = r'Bilder/2020_11_04-Pd-aAl__5.dm3'  # 'Bilder/2020_10_29-Pd-gAl__11.dm3'
SCALE = 0.5
ALPHA = 5  # Contrast
BETA = -250   # Brightness

#import dm3 bild
dm3f = read_dm3_file(PATH)
img_original = dm3_to_cv(dm3f, alpha=ALPHA, beta=BETA)
img = dm3_to_cv(dm3f, alpha=ALPHA, beta=BETA)

width = img.shape[0]
#filtering
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray, (13, 13), 0)
th, bw = cv2.threshold(gray, 0.42*np.amax(gray), 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
output_bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
#adaptive threshhold
gray = cv2.GaussianBlur(gray, (31, 31), 0)
ad_th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 2)
masked= cv2.bitwise_and(ad_th, ad_th, mask = output_bw)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
output_ad_th = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel)


dist = cv2.distanceTransform(output_ad_th, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
borderSize = 24
distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize,
                                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
gap = 14
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(borderSize-gap)+1, 2*(borderSize-gap)+1))
kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap,
                             cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
mn, mx, _, _ = cv2.minMaxLoc(nxcor)
th, peaks = cv2.threshold(nxcor, mx*0.45, 255, cv2.THRESH_BINARY)
peaks8u = cv2.convertScaleAbs(peaks)
contours, hierarchy = cv2.findContours(peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


counter = 0
size = np.zeros(len(contours))
for i in range(len(contours)):
    cv2.drawContours(img, contours, i, (0, 0, 255), 1)
    x, y, w, h = cv2.boundingRect(contours[i])
    _, mx, _, mxloc = cv2.minMaxLoc(dist[y:y + h, x:x + w])
    if 7.5 <= mx <= 400:
        center = (int(mxloc[0]+x), int(mxloc[1]+y))
        radius = int(np.round(mx))
        cv2.circle(img, center, radius, (255, 0, 0), 1)
        cv2.line(img_original, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (255,255,255), 1)
        #cv2.rectangle(img_original, (x, y), (x+w, y+h), (0, 255, 255), 1)

        #cv2.putText(img, "{}".format(i), (x-w, y-h), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
        size[counter] = mx*2
        counter += 1

#put text
size = np.trim_zeros(size)
mean = np.sum(size)/counter
cv2.putText(img_original, "counter: " + str(int(counter)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(img_original, "av. size: : " + str(round(mean, 2)) + 'px ~ ' + str(round(mean*dm3f.pxsize[0], 2)) + "nm",
            (75, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.line(img_original, (width - 50, 75), (width - 50 - int(30/dm3f.pxsize[0]), 75), (255,255,255), 3)
cv2.putText(img_original, "30 nm", (width - 80 - int(30/(2*dm3f.pxsize[0])), 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)



# change between images with arrow keys
i=0
frame = [img, ad_th, dist, nxcor, peaks, peaks8u, output_bw, img_original]
for i in range(len(frame)):
    frame[i] = cv2.resize(frame[i], (0, 0), fx=SCALE, fy=SCALE)

names = ['img','bw', 'dist','nxcor', 'peaks', '', 'ad.thresh','', 'template']
while 1:
    cv2.imshow('image', frame[i])
    #print(names[i])
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

cv2.destroyAllWindows()

#saving Data
#cv2.imwrite(r'Results/particles3.jpg', img_original)
size_in_nm = size*dm3f.pxsize[0]
size_in_nm.tofile('Files/sample.csv', sep='\n')

if __name__ == '__main__':
    pass
