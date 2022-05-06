import cv2
import numpy as np
import matplotlib.pyplot as plt
import dm3_lib as dm3
from Aufgabe_Catalyst import read_dm3_file, dm3_to_cv

# Parameters-------------------------------------
#Image
PATH = r'Bilder/2020_11_04-Pd-aAl__5.dm3'#'Bilder/2020_10_29-Pd-gAl__11.dm3'
SCALE = 0.5
ALPHA = 5  # Contrast
BETA = -250   # Brightness
#Particles
PARTICLE_SIZE = 24 # Expected particle size (px)
GAP = 14
MIN_SIZE = None
#Filter ADAPTIVE THRESH
BG_THRESH = 0.42
BLUR_IMG = 31
AD_THRESH_SIZE = 41
AD_THRESH_C = 2
#Filter PEAKS
PEAKS_THRESH = 0.45
#--------------------------------------------------
def nothing(x):
    pass

#Create Trackbars
cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
cv2.createTrackbar("Contrast", "Trackbars", 0, 10, nothing)
cv2.createTrackbar("Brightness", "Trackbars", -50, 400, nothing)

#cv2.createTrackbar("Contrast", "Trackbars", 87, 179, nothing())
#cv2.createTrackbar("Contrast", "Trackbars", 87, 179, nothing())
#cv2.setTrackbarMin("Contrast", "Trackbars", 1)
#cv2.setTrackbarPos("Contrast", "Trackbars", 5)
#import dm3 bild
dm3f = read_dm3_file(PATH)
img_raw = dm3_to_cv(dm3f)

while 1:
    cv2.setTrackbarMin("Brightness", "Trackbars", -400)
    #Get Trackbar Values
    alpha = cv2.getTrackbarPos("Contrast", "Trackbars")
    beta = cv2.getTrackbarPos("Brightness", "Trackbars")

    img = cv2.convertScaleAbs(img_raw, alpha=alpha, beta=beta)

    cv2.imshow("image", img)
    if cv2.waitKey(22) & 0xFF == 27:
        break

#cv2.imwrite(r'Results/image.jpg', img)
cv2.destroyAllWindows()

if __name__ == '__main__':
    pass
