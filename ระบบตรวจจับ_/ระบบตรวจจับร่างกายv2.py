# ======= ส่วนที่ 1: ติดตั้งและเรียกใช้ไลบรารีที่จำเป็น =======
import cv2  # ไลบรารี OpenCV สำหรับจัดการกล้องและภาพ
import mediapipe as mp  # ไลบรารีสำหรับวิเคราะห์โครงสร้างร่างกาย
from ultralytics import YOLO  # ใช้ YOLOv8 สำหรับตรวจจับคนในภาพ
import numpy as np  # ใช้สำหรับจัดการ array หรือรูปภาพ

# ======= ส่วนที่ 2: โหลดโมเดลที่ใช้ =======
yolo_model = YOLO("yolov8n.pt")  
# โหลดโมเดล YOLOv8n (n = nano) ไฟล์ .pt ใช้ตรวจจับวัตถุในภาพ เช่น คน รถ ฯลฯ

mp_pose = mp.solutions.pose  # เรียกคลาสสำหรับ pose detection
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)  
# สร้างตัวตรวจจับท่าทาง (ตั้งค่าให้วิเคราะห์แบบต่อเนื่อง เรียลไทม์)
mp_draw = mp.solutions.drawing_utils  # ใช้สำหรับวาดเส้นเชื่อมจุดต่างๆ บนร่างกาย

# ======= ส่วนที่ 3: เปิดกล้อง =======
cap = cv2.VideoCapture(1)  # เปิดกล้อง webcam (ถ้าไม่ขึ้นภาพให้เปลี่ยนเป็น 1)

# ======= ส่วนที่ 4: ลูปหลักของโปรแกรม =======
while True:
    success, img = cap.read()  # อ่านภาพจากกล้อง 1 เฟรม (1 รูปต่อรอบ)
    if not success:
        print("ไม่สามารถอ่านภาพจากกล้องได้")
        break  # ถ้าอ่านภาพไม่ได้ ให้หยุดทำงานทันที

    results = yolo_model(img, verbose=False)[0]  
    # ให้ YOLO ตรวจจับวัตถุในภาพ และเลือกผลลัพธ์แรก [0] (เพราะอาจมีหลายภาพส่งพร้อมกัน)
    
    bboxes = []  # ลิสต์เก็บกรอบของแต่ละคนที่เจอ

    for r in results.boxes:
        cls = int(r.cls[0])  # ตรวจว่า object ที่เจอคือคลาสอะไร (0 = คน)
        if cls == 0:
            x1, y1, x2, y2 = map(int, r.xyxy[0])  # ดึงพิกัดกล่องสี่เหลี่ยม (bounding box)
            bboxes.append([x1, y1, x2, y2])  # เก็บกรอบของคนที่ตรวจเจอไว้

    # ======= ส่วนที่ 5: ตรวจ pose แต่ละคนแยกกัน =======
    for box in bboxes:
        x1, y1, x2, y2 = box
        person_img = img[y1:y2, x1:x2]  # ตัดเฉพาะส่วนของภาพที่เป็นคน

        if person_img.size == 0:
            continue  # ถ้าภาพกล่องนั้นไม่มีข้อมูล (พื้นที่เป็น 0) ให้ข้าม

        person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)  # แปลงภาพจาก BGR เป็น RGB
        pose_result = pose.process(person_rgb)  # วิเคราะห์ pose จากภาพของคนคนนั้น

        if pose_result.pose_landmarks:
            mp_draw.draw_landmarks(person_img, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # ถ้ามีผลลัพธ์จากการตรวจ pose → วาดโครงร่างลงในภาพย่อยของคนนั้น

        img[y1:y2, x1:x2] = person_img  # นำภาพย่อยที่วาดโครงร่างแล้ว แปะกลับไปที่ภาพหลัก

    # ======= ส่วนที่ 6: แสดงผลลัพธ์ =======
    cv2.imshow("Multi-Person Pose Detection", img)  # แสดงภาพที่ประมวลผลเสร็จแล้ว

    if cv2.waitKey(1) & 0xFF == ord('q'):  # ถ้าผู้ใช้กด 'q' → หยุดโปรแกรม
        break

# ======= ส่วนที่ 7: ปิดโปรแกรม =======
cap.release()  # ปิดกล้อง
cv2.destroyAllWindows()  # ปิดหน้าต่างทั้งหมด




#////////////////////////
#  1. ติดตั้งไลบรารีที่จำเป็น
# เปิด Terminal หรือ CMD แล้วพิมพ์:

# bash
# Copy
# Edit
# pip install ultralytics mediapipe opencv-python
# ถ้า pip ไม่ทำงาน ลองใช้ python -m pip แทน เช่น:

# bash
# Copy
# Edit
# python -m pip install ultralytics