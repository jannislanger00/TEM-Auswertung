import cv2
import numpy as np
import dm3_lib as dm3

# Parameters
PATH = r'Bilder/2020_11_04-Pd-aAl__5.dm3'
SCALE = 0.9
ALPHA = 1.7  # Contrast
BETA = -50   # Brightness


def read_dm3_file(path):
    # import dm3 file
    dm3f = dm3.DM3(path)
    return dm3f


def dm3_to_cv(dm3f, alpha=1.0, beta=0):
    # convert to float32 and divide by the biggest value
    img = dm3f.imagedata
    img = np.float32(img)
    img = img / (np.amax(img))
    # convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # multiply by 255 and transform to uint8 array
    img = np.uint8(img * 255)
    # scale image, default by 1
    '''img = cv2.resize(img, (0, 0), fx=scale, fy=scale)'''
    # Adjust image
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img


def get_px_size(dm3f):
    size_px = dm3f.pxsize
    return size_px


def main():
    # Einlesen dm3 Datei
    dm3_file = read_dm3_file(PATH)
    # Umwandenl in Opencv Format (RGB, uint8)
    img = dm3_to_cv(dm3_file, ALPHA, BETA)
    # Pixelgröße bestimmen
    px_size = get_px_size(dm3_file)

    # Anzeigen
    print(px_size[0], 'Einheit:', px_size[1])
    cv2.imshow('TEM_image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
