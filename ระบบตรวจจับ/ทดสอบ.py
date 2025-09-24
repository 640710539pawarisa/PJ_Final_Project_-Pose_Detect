import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
print("เช็ค")


import cv2
import cvzone
from cvzone.PoseModule import PoseDetector

print("ทุกอย่างพร้อมใช้งานแล้วค่ะ")


##พิมพ์คำสั่งนี้เพื่อเปลี่ยน code page เป็น UTF-8

# chcp 65001
