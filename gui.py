import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QLabel, QPushButton
from PyQt5.QtCore import Qt
import cv2
from Aufgabe_Catalyst import read_dm3_file, dm3_to_cv
PATH = r'Bilder/2020_10_06-Pd-Zr__7.dm3'
DPI = 10

class Export_img(QMainWindow):

    def __init__(self, path):
        super().__init__()
        self.setGeometry(50,50,650,600)
        self.setWindowTitle("Adjust Image Contrast")
        self.Gui_objects()
        self.show()

        self.img = self.import_img(path)
        self.update_img(self.img)
    def Gui_objects(self):
        #Contrast
        self.alpha = 1.0
        alpha_slider = QSlider(Qt.Horizontal, self)
        alpha_slider.setRange(10, 10*DPI)
        alpha_slider.setGeometry(200, 100, 300, 60)
        self.l_alpha = QLabel('Contrast', self)
        self.l_alpha.setGeometry(30, 100, 150, 60)
        self.l_alpha.setText('Constrast: ' + str(self.alpha))
        alpha_slider.valueChanged[int].connect(self.changeValueA)
        #Brightness
        self.beta = 0
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
        Button.clicked.connect(self.ButtonClick)

    def changeValueA(self, value):
        self.alpha = value/DPI
        self.update_img(self.img)
        self.l_alpha.setText('Contrast: ' + str(self.alpha))


    def changeValueB(self, value):
        self.beta = value
        self.update_img(self.img)
        self.l_beta.setText('Brightness: ' + str(self.beta))

    def ButtonClick(self):
        self.img_path = r'Results/gui_img.jpg'
        cv2.imwrite(self.img_path, self.res)
        sys.exit(app.exec_())
        cv2.destroyAllWindows()

    def import_img(self, path):
        dm3f = read_dm3_file(path)
        img_raw = dm3_to_cv(dm3f)
        return img_raw


    def update_img(self, img):
        self.res = cv2.convertScaleAbs(img, alpha=self.alpha, beta=self.beta)
        cv2.imshow('img', self.res)

    def getImgPath(self):
        return self.img_path

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Export_img(PATH)
    sys.exit(app.exec_())
    cv2.destroyAllWindows()