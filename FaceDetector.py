import cv2
from random import randrange


#Loading pre trained data from opencv
trainedFaceData = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Choose an image to detect faces in
# img = cv2.imread("ME.png")

webcam = cv2.VideoCapture(0)
key = cv2.waitKey(1)

while True:
    # Read the current frame
    successfulFrameRead,frame=webcam.read()
    flipped = cv2.flip(frame,1)
    #MUST: Convert to grayscale, BGR is RGB, so converts color to grayscale
    grayImg = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)

    face_Coordinates = trainedFaceData.detectMultiScale(grayImg)
    for (x,y,w,h) in face_Coordinates:
        cv2.rectangle(flipped,(x,y),(x+w,y+h),(randrange(128,256),randrange(128,256),randrange(128,256)),5)
#                                       B  G  R
    
    #Display the image
    cv2.imshow('FDApp',flipped)
    #stops the image from immediately closing
    key = cv2.waitKey(1)
    #quit if Q is pressed
    if key==81 or key==113:
        break
    
webcam.release()
"""
#Detect Faces, detectMultiScale means detecting just faces in all sizes
face_Coordinates = trainedFaceData.detectMultiScale(grayImg)


# print(face_Coordinates)

#Draw rectangles
for (x,y,w,h) in face_Coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(128,256),randrange(128,256),randrange(128,256)),5)
#                                       B  G  R
#Display the image
cv2.imshow('FDApp',img)
#stops the image from immediately closing
cv2.waitKey()
"""
print("Code Run Done")

"""
    How does it actually work:
    WTF is HAAR-CASCADE? 
    Its a dude, its his algorithm chain of ml things
    that the image is being passed through and cascades
    all the squares down until a face like figure is 
    detected
    face in this algroithm is made of haar features
    some of which include edge o|x  o/x
    line o|x|o o/x/o
    Four rectangle o|x
                   x|0
    combination of these features can be used 
    to form a face.
    Eg:on grayscale, usually are around the eyes is 
    darker than cheeks, and forehead so o/x/o
    Similarly eye nose eye, dark light dark x|o|x
    Over 1000s of these matches would confirm the 
    presence of a face
    
    HOW IS TRAINING DONE?
    STEP 1: FACES AND NON-FACES(walls, plants, cups, 
    etc.) are shown (Supervised learning, data is 
    labeled by humans at the start)
    STEP 2: Find Haar Features, sum them up, and 
    determine if face exists
    -> So, we have to try every haar feature in evry
    size on every location on every 
    image... yeah it takes a while
    -> calculates the sum of every black pixels on
    haar features and every white pixel, finds the
    difference, stores it, if it matches set 
    threshold, counts as a "good" rating, eliminates
    the "bad" performing haar features, continues,
    for each location,size,image... finally learns
    what a face is
"""