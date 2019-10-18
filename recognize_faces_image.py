# import the packages that are required

from imutils import face_utils
from imutils.face_utils import FaceAligner
import face_recognition
import numpy as np
import argparse
import imutils
import dlib
import pickle
import cv2
import uuid
import rotateImage




# if you want to pass arguments at the time of running code
# follow below code and format for running code


#在命令行输入  python recognize_faces_image.py --encodings encodings.pickle --image examples/1.jpg --detection-method hog

 #运行
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())



# load the known faces and embeddings 加载处理好的名字和面部
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# load the input image and convert it from BGR to RGB
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x, y) coordinates of the bounding box corresponding to
# each face inthe input image and compute facial embeddings for each face
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb, model = args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# initialize the list of names of detected faces
names = []

#加载完成

# loop over facial embeddings  遍历面部编码
for encoding in encodings:
    # compares each face in the input image to our known encodings
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"

#遍历结束

    # check if  match is found or not 匹配人脸
    if True in matches:
        #find the indexes of all matches and initialize a dictionary
        # to count number of times a match occur
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        # loop over matched indexes and maintain a count for each face
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        # Select the recognized face with maximum number of matches and
        # if there is a tie Python selects first entry from the dictionary
        name = max(counts, key=counts.get)

    # update the list of names
    names.append(name)

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
    # draw predicted face name on image
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)

# Output Image

cv2.imshow("Detected face", image)
cv2.waitKey(0)


