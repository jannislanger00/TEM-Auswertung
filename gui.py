import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QLabel
from PyQt5.QtCore import Qt
import cv2
from Aufgabe_Catalyst import read_dm3_file, dm3_to_cv
PATH = r'Bilder/2020_11_04-Pd-aAl__5.dm3'
DPI = 10

class Example(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setGeometry(50,50,650,900)
        self.setWindowTitle("Checkbox Example")
        self.Gui_objects()
        self.show()

        self.img = self.import_img(PATH)
        self.update_img(self.img)

    def Gui_objects(self):

        self.alpha = 1
        alpha_slider = QSlider(Qt.Horizontal, self)
        alpha_slider.setRange(10, 10*DPI)
        alpha_slider.setGeometry(200, 150, 300, 60)
        self.l_alpha = QLabel('Contrast', self)
        self.l_alpha.setGeometry(30, 150, 150, 60)
        #self.l_alpha.setText('Constrast: ' + str(self.alpha))
        alpha_slider.valueChanged[int].connect(self.changeValueA)

        self.beta = 0
        beta_slider = QSlider(Qt.Horizontal, self)
        beta_slider.setRange(-400,400)
        beta_slider.setGeometry(200, 250, 300, 60)
        self.l_beta = QLabel('Brightness', self)
        self.l_beta.setGeometry(30, 250, 150, 60)
        self.l_beta.setText('Brightness: ' + str(self.beta))
        beta_slider.valueChanged[int].connect(self.changeValueB)


    def changeValueA(self, value):
        self.alpha = value/DPI
        print(self.alpha)
        self.update_img(self.img)
        self.l_alpha.setText('Contrast: ' + str(self.alpha))


    def changeValueB(self, value):
        self.beta = value
        self.update_img(self.img)
        self.l_beta.setText('Brightness: ' + str(self.beta))

    def import_img(self, path):
        dm3f = read_dm3_file(path)
        img_raw = dm3_to_cv(dm3f)
        return img_raw


    def update_img(self, img):
        img = cv2.convertScaleAbs(img, alpha=self.alpha, beta=self.beta)
        cv2.imshow('img', img)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
    cv2.destroyAllWindows()