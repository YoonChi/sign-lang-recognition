import cv2
import os
import time
import uuid

IMAGES_PATH = '/Users/anjalia/Desktop/AI_PROJECT-ASL/collectedImages'

labels = ['hello', 'yes', 'no']
number_imgs = 15

for label in labels:
    !mkdir {'/Users/anjalia/Desktop/AI_PROJECT-ASL/collectedImages//'+label}
    cap = cv2.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for imgnum in range(number_imgs):
        ret,frame = cap.read()
        imgname = os.path.join(IMAGES_PATH, label, label+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
