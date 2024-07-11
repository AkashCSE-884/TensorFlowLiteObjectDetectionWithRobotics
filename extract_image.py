# Importing all necessary libraries
import cv2
import os

cam = cv2.VideoCapture("F:/advance/au/Pic&Place images/obj_3.mp4")

try:


    if not os.path.exists('F:/advance/au/3_obj_images'):
        os.makedirs('F:/advance/au/3_obj_images')

except OSError:
    print ('Error: Creating directory of data')


currentframe = 0

while(True):

    ret,frame = cam.read()

    if ret:

        name = 'F:/advance/au/3_obj_images/' + str(currentframe) + '.jpg'
        print ('Creating...' + name)


        cv2.imwrite(name, frame)

        currentframe += 1
    else:
        break


cam.release()
cv2.destroyAllWindows()