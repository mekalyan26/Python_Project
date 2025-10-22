from sre_constants import SUCCESS
import cv2
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.SerialModule import SerialObject
#from serial import Serial

cap = cv2.VideoCapture(0)
detector = FaceDetector()
#arduino = SerialObject('/dev/ttyACM0')

arduino2 = SerialObject(portNo='/dev/ttyACM0', baudRate=115200)

while True:
    SUCCESS, img = cap.read()

    img, bBoxes = detector.findFaces(img)

    if bBoxes:
        arduino2.sendData([1,0])
    else:
        arduino2.sendData([0,1])
    
    cv2.imshow("Video", img)
    cv2.waitKey(1)

