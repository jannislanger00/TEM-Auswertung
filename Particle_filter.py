import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QLabel, QPushButton
from PyQt5.QtCore import Qt
import cv2
from Aufgabe_Catalyst import read_dm3_file, dm3_to_cv
from gui import Export_img
PATH = r'Bilder/2020_11_04-Pd-aAl__5.dm3' #Path dm3 File
DPI = 100

class Filter_img(QMainWindow):

    def __init__(self, path):
        super().__init__()
        self.setGeometry(50,50,650,600)
        self.setWindowTitle("Particle Detection")
        self.Gui_objects()
        self.show()

        self.img = self.import_img(path)
        self.update_img(self.img)
    def Gui_objects(self):
        #Contrast
        self.bg_thresh = 0.35
        bg_slider = QSlider(Qt.Horizontal, self)
        bg_slider.setRange(0, 1*DPI)
        bg_slider.setGeometry(300, 100, 250, 60)
        bg_slider.setSliderPosition(int(self.bg_thresh*DPI))
        self.label_bg_thresh = QLabel('Background Thresholt', self)
        self.label_bg_thresh.setGeometry(30, 100, 270, 60)
        self.label_bg_thresh.setText('Background Thresholt: ' + str(self.bg_thresh))
        bg_slider.valueChanged[int].connect(self.changeValueBg)
        #Brightness
        '''self.beta = 0
        beta_slider = QSlider(Qt.Horizontal, self)
        beta_slider.setRange(-400,400)
        beta_slider.setGeometry(200, 200, 300, 60)
        self.l_beta = QLabel('Brightness', self)
        self.l_beta.setGeometry(30, 200, 150, 60)
        self.l_beta.setText('Brightness: ' + str(self.beta))
        beta_slider.valueChanged[int].connect(self.changeValueB)

        Button = QPushButton('Prozess Image', self)
        Button.resize(200, 32)
        Button.move(250, 300)
        Button.clicked.connect(self.ButtonClick)'''

    def changeValueBg(self, value):
        self.bg_thresh = value / DPI
        self.update_img(self.img)
        self.label_bg_thresh.setText('Background Thresholt: ' + str(self.bg_thresh))


    def changeValueB(self, value):
        self.beta = value
        self.update_img(self.img)
        self.l_beta.setText('Brightness: ' + str(self.beta))

    def ButtonClick(self):
        cv2.imwrite(r'Results/gui_img.jpg', self.res)
        sys.exit(app.exec_())
        cv2.destroyAllWindows()

    def import_img(self, path):
        img_raw = cv2.imread(path)
        return img_raw


    def update_img(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (13, 13), 0)
        th, bw = cv2.threshold(gray, self.bg_thresh * np.amax(gray), 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        output_bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        masked = cv2.bitwise_and(img, img, mask=output_bw)
        #adthresh
        gray = cv2.GaussianBlur(gray, (31, 31), 0)
        ad_th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 2)
        masked = cv2.bitwise_and(ad_th, ad_th, mask=output_bw)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        output_ad_th = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel)

        imgStack = self.stackImages(0.4, ([img, bw], [masked, output_ad_th]))
        cv2.imshow('img', imgStack)

    def stackImages(self, scale, imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                    None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank] * rows
            hor_con = [imageBlank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale,
                                             scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(imgArray)
            ver = hor
        return ver


if __name__ == '__main__':
    app = QApplication(sys.argv)
    #ex = Export_img(PATH)
    #path = ex.getImgPath()
    #print(path)
    fil = Filter_img(r'Results/gui_img.jpg')

    sys.exit(app.exec_())
    cv2.destroyAllWindows()