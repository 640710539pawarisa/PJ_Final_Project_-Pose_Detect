print("เริ่มโปรแกรม...")

import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np

print("นำเข้าไลบรารีสำเร็จแล้ว")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ เปิดกล้องไม่ได้")
    exit()

print("✅ เปิดกล้องได้สำเร็จ")

print("✅ เปิดกล้องได้สำเร็จ")
