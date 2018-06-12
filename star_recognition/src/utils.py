import os
import cv2
import numpy as np

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def load_image(addr):
    # TODO read image
    return cv2.imread(addr)

def resize_image(img, image_size=40):
    # TODO resize image quadratically to image_size
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    return img

def grayscale_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
    
def preprocess_image(img, image_size=40):
    # TODO resize image quadratically to image_size
    #img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    img = resize_image(img, image_size)
    # TODO convert image to grayscale with COLOR_BGR2GRAY
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # TODO convert img to np.float32 type
    img = img.astype(np.float32)
    # TODO scale pixel from range 0 - 255 to -1 - +1
    img = 2*(img/255)-1
    return img

def extract_faces(img, haar_face_cascade):
    if img is None:
        #print(path)
        return []
    #convert the test image to gray image as opencv face detector expects gray images 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #let's detect multiscale (some images may be closer to camera than others) images 
    faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    
    face_imgs = []
    for (x, y, w, h) in faces:
        face_imgs.append(img[y:y+h, x:x+w])
    return face_imgs

def plot_image(img):
    cv2.imshow("Plot", img) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()