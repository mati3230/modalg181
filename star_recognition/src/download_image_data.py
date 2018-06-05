'''
Created on 13.04.2018

@author: mati3230
'''
import urllib.request
import os
import http
import cv2
import ssl
from utils import create_dir

def download_images(url_txt_path, path, label):
    try:
        urls = [line.rstrip('\n') for line in open(url_txt_path)]
    except FileNotFoundError:
        print("could not find {0}".format(url_txt_path))
        return
    for i in range(len(urls)):
        filename="{0}/{1}.{2}.jpg".format(path, label,i)
        if os.path.isfile(filename):
            continue
        url = urls[i]
        try:
            urllib.request.urlretrieve(url=url, filename=filename)
        except urllib.error.HTTPError:
            continue
        except ssl.SSLError:
            continue
        except urllib.error.URLError:
            continue
        except http.client.RemoteDisconnected:
            continue
        
def plot_image(path):
    #load test image
    img_to_plot = cv2.imread(path)
    #convert the test image to gray image as opencv face detector expects gray images 
    gray_img = cv2.cvtColor(img_to_plot, cv2.COLOR_BGR2GRAY)
    cv2.imshow(path, gray_img) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

def plot_face(path):
    #load test image
    img_to_plot = cv2.imread(path)
    #convert the test image to gray image as opencv face detector expects gray images 
    gray_img = cv2.cvtColor(img_to_plot, cv2.COLOR_BGR2GRAY)
    #load cascade classifier training file for haarcascade 
    haar_face_cascade = cv2.CascadeClassifier()
    result = haar_face_cascade.load("D:/Projects/ml-algorithms/star_recognition/haarcascade_frontalface_alt.xml")
    if not result:
        print("Error loading haarcascade")
        return
    #let's detect multiscale (some images may be closer to camera than others) images 
    faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
     
    #print the number of faces found 
    print('Faces found: ', len(faces))
    
    #go over list of faces and draw them as rectangles on original colored 
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(img_to_plot, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    if len(faces) == 0: 
        return
    (x,y,w,h) = faces[0]
    roi = img_to_plot[y:y+h, x:x+w]
    #cv2.imwrite("test.jpg",roi)
    cv2.imshow(path, roi) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

def extract_faces(path, haar_face_cascade):
    #load image
    img = cv2.imread(path)
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

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

dataset_path = "../../datasets/stars_from_google_images"
create_dir(path=dataset_path)

haar_face_cascade = cv2.CascadeClassifier()
result = haar_face_cascade.load("{0}/../haarcascade_frontalface_alt.xml".format(dir_path))
if not result:
    print("Error loading haarcascade")
    raise SystemExit(-1)

stars = ["nicolas_cage", "brad_pitt", "angelina_jolie", "leonardo_dicaprio", "robert_downey_jr"]
#stars = ["robert_downey_jr"]

http.client._MAXHEADERS = 1000

for star in stars:
    print("download images of star: {0}".format(star))
    star_path = "{0}/{1}".format(dataset_path, star)
    create_dir(path=star_path)
    
    download_images(url_txt_path="{0}/{1}_urls.txt".format(dataset_path, star), 
                path=star_path, 
                label=star)
    #plot_image(path="{0}/{1}/{1}.0.jpg".format(dataset_path, star))
    #plot_face(path="{0}/{1}/{1}.0.jpg".format(dataset_path, star))
    
    print("extract faces")
    faces_path = "{0}/faces".format(star_path, star)
    create_dir(path=faces_path)
    
    filenames = os.listdir(star_path)
    for i in range(len(filenames)):
        filename = filenames[i]
        if filename == "faces":
            continue
        face_imgs = extract_faces(path="{0}/{1}".format(star_path, filename), 
                                  haar_face_cascade=haar_face_cascade)
        
        for j in range(len(face_imgs)):
            face_img = face_imgs[j]
            cv2.imwrite("{0}/{1}.{2}.jpg".format(faces_path, star, i+j),face_img)
print("done")
    


