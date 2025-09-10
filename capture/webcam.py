# capture/webcam.py
import cv2


def get_camera_stream(index=0):
    return cv2.VideoCapture(index)
