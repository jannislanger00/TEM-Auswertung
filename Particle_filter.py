import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QLabel, QPushButton, QLineEdit
from PyQt5.QtCore import Qt
import cv2
from Aufgabe_Catalyst import read_dm3_file, dm3_to_cv
from gui import Export_img

PATH = r'Bilder/2020_11_12-PdPt19_100_C-black_2.dm3' #Path dm3 File
DPI = 100
SCALE = 0.8

def import_img(path):
    img_raw = cv2.imread(path)
    return img_raw

def resizeImg(img, scale):
    res = cv2.resize(img, (0,0), fx=scale, fy=scale)
    return res

def stackImages(scale, imgArray):
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



class Filter_img(QMainWindow):

    def __init__(self, path):
        super().__init__()
        #start Values
        self.bg_thresh = 0.35
        self.peaksTresh = 0.45
        #Template
        self.particle_size = 20 # pixel
        self.gap = 14
        #start up
        self.img = import_img(path)
        self.text_input = False
        self.update_all()
        # Toolbar
        self.setGeometry(50,50,650,600)
        self.setWindowTitle("Particle Detection")
        self.createUI()
        self.show()

    def createUI(self):
        # Create textbox
        self.textbox = QLineEdit(self)
        self.textbox.move(300, 20)
        self.textbox.resize(80,40)
        self.textbox.setText(str(self.particle_size))
        if self.textbox.text().isnumeric() == True:
            self.text_input = True
        #Background Thresholt
        bg_slider = QSlider(Qt.Horizontal, self)
        bg_slider.setRange(0, 1*DPI)
        bg_slider.setGeometry(300, 100, 300, 60)
        bg_slider.setSliderPosition(int(self.bg_thresh*DPI))
        self.label_bg_thresh = QLabel('Background Thresholt', self)
        self.label_bg_thresh.setGeometry(30, 100, 270, 60)
        self.label_bg_thresh.setText('Background Thresholt: ' + str(self.bg_thresh))
        bg_slider.valueChanged[int].connect(self.changeBgThresh)

        #Peaks
        peaks_slider = QSlider(Qt.Horizontal, self)
        peaks_slider.setRange(0, 1*DPI)
        peaks_slider.setGeometry(300, 200, 300, 60)
        peaks_slider.setSliderPosition(int(self.peaksTresh*DPI))
        self.label_peaks = QLabel('Peaks Thresh', self)
        self.label_peaks.setGeometry(30, 200, 270, 60)
        self.label_peaks.setText('Peaks Thresh: ' + str(self.peaksTresh))
        print(str(self.peaksTresh))
        peaks_slider.valueChanged[int].connect(self.changePeaksThresh)

        update_button = QPushButton('Update', self)
        update_button.resize(200, 32)
        update_button.move(150, 300)
        update_button.clicked.connect(self.update_all)

        res_button = QPushButton('Aprox. Particles', self)
        res_button.resize(200, 32)
        res_button.move(350, 300)
        res_button.clicked.connect(self.calcParticles)

        self.comp_button = QPushButton('Compare', self)
        self.comp_button.resize(200, 32)
        self.comp_button.move(150, 400)
        self.comp_button.clicked.connect(self.compareImg)

        self.original_button = QPushButton('show Image', self)
        self.original_button.resize(200, 32)
        self.original_button.move(350, 400)
        self.original_button.clicked.connect(self.showOriginal)

    def changeBgThresh(self, value):
        self.bg_thresh = value / DPI
        self.updateBgTresh()
        self.label_bg_thresh.setText('Background Thresholt: ' + str(self.bg_thresh))

    def changePeaksThresh(self, value):
        self.peaksTresh = value / DPI
        self.label_peaks.setText('Peaks Thresh: ' + str(self.peaksTresh))
        self.updatePeaks()

    def updateBgTresh(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (13, 13), 0)
        th, self.bw = cv2.threshold(gray, self.bg_thresh * np.amax(gray), 255, cv2.THRESH_BINARY_INV)
        res = resizeImg(self.bw, SCALE)
        self.k = 0
        cv2.imshow('img', res)

    def updatePeaks(self):
        mn, mx, _, _ = cv2.minMaxLoc(self.nxcor)
        th, peaks = cv2.threshold(self.nxcor, mx * self.peaksTresh, 255, cv2.THRESH_BINARY)
        self.peaks8u = cv2.convertScaleAbs(peaks)
        res = resizeImg(self.peaks8u, SCALE)
        self.k = 3
        cv2.imshow('img', res)

    def showOriginal(self):
        res = resizeImg(self.img, SCALE)
        cv2.imshow('img', res)

    def compareImg(self):
        img_ref = self.imgList[self.k]
        res = self.img.copy()
        if img_ref is not None:
            contours, hierarchy = cv2.findContours(img_ref, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(contours)):
                cv2.drawContours(res, contours, i, (0, 0, 255), 1)

        res = resizeImg(res, SCALE)
        cv2.imshow('img', res)

    def update_all(self):
        if self.text_input == True:
            self.particle_size = int(self.textbox.text())
        #bw
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (13, 13), 0)
        th, self.bw = cv2.threshold(gray, self.bg_thresh * np.amax(gray), 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        output_bw = cv2.morphologyEx(self.bw, cv2.MORPH_OPEN, kernel)
        #Adthresh
        gray = cv2.GaussianBlur(gray, (31, 31), 0)
        ad_th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 2)
        masked = cv2.bitwise_and(ad_th, ad_th, mask=output_bw)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        output_ad_th = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel)
        self.dist = cv2.distanceTransform(output_ad_th, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        borderSize = int((self.particle_size/2) + self.gap)
        print(borderSize)
        distborder = cv2.copyMakeBorder(self.dist, borderSize, borderSize, borderSize, borderSize,
                                        cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)

        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (borderSize - self.gap) + 1, 2 * (borderSize - self.gap) + 1))
        kernel2 = cv2.copyMakeBorder(kernel2, self.gap, self.gap, self.gap, self.gap,
                                     cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
        distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        self.nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
        #Peaks
        mn, mx, _, _ = cv2.minMaxLoc(self.nxcor)
        th, peaks = cv2.threshold(self.nxcor, mx * self.peaksTresh, 255, cv2.THRESH_BINARY)
        peaks8u = cv2.convertScaleAbs(peaks)
        contours, hierarchy = cv2.findContours(peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        self.peaks8u = cv2.convertScaleAbs(peaks)
        self.imgList = list((self.bw, self.dist, self.nxcor, self.peaks8u))
        self.imgArray = list(([self.bw, self.dist], [self.nxcor, self.peaks8u]))
        imgStack = stackImages(SCALE/2, self.imgArray)
        cv2.imshow('img', imgStack)

    def calcParticles(self):
        self.img_draw = self.img.copy()
        contours, hierarchy = cv2.findContours(self.peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        counter = 0
        size = np.zeros(len(contours))
        for i in range(len(contours)):
            #cv2.drawContours(self.img_draw, contours, i, (0, 0, 255), 1)
            x, y, w, h = cv2.boundingRect(contours[i])
            _, mx, _, mxloc = cv2.minMaxLoc(self.dist[y:y + h, x:x + w])
            if 7.5 <= mx <= 400:
                center = (int(mxloc[0] + x), int(mxloc[1] + y))
                radius = int(np.round(mx))
                cv2.circle(self.img_draw, center, radius, (255, 0, 0), 1)
                size[counter] = mx * 2
                counter += 1

        size = np.trim_zeros(size)
        mean = np.sum(size) / counter
        cv2.putText(self.img_draw, "counter: " + str(int(counter)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2)
        cv2.putText(self.img_draw,
                    "av. size: : " + str(round(mean, 2)),
                    (75, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        res = resizeImg(self.img_draw, SCALE)
        cv2.imshow('img', res)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    #ex = Export_img(PATH)
    #cv2.waitKey(0)
    #path = ex.getImgPath()
    #print(path)
    fil = Filter_img('Results/gui_img.jpg')

    sys.exit(app.exec_())
    cv2.destroyAllWindows()