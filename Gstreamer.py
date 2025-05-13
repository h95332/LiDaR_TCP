import os

os.add_dll_directory(r"C:\Program Files\gstreamer\1.0\msvc_x86_64\bin")
os.add_dll_directory(r"C:\Program Files\gstreamer\1.0\msvc_x86_64\lib\\gstreamer-1.0")
import cv2 as cv
import numpy
 
sourceUrl = "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg"
#command = "appsrc ! videoconvert ! x264enc ! rtph264pay ! udpsink port=5004 host=127.0.0.1 "
command = "appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw, format=BGRx ! videoconvert ! x264enc speed-preset=veryfast tune=zerolatency bitrate=800 insert-vui=1 ! h264parse ! rtph264pay name=pay0 pt=96 config-interval=1 ! udpsink port=5004 host=127.0.0.1 auto-multicast=0"
 
cap = cv.VideoCapture(sourceUrl)
if not cap.isOpened():
    print("Capture Not Opened")
 
writer = cv.VideoWriter(command, cv.CAP_GSTREAMER, 0, 30, (1280, 720), True)
if not writer.isOpened():
    print("WRITER NOT OPENED!!!!!")
 
if cap.isOpened() and writer.isOpened():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Not Fram received")
            break
        cv.imshow("frame", frame)
        if cv.waitKey(1) > 0:
            break
   
    cap.release()
    writer.release()