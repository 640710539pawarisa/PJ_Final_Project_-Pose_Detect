#เช็กก่อนว่า version ไหน
#python --version

#จำเป็นต้องติดตั้้ง
# python -m pip install opencv-python cvzone mediapipe
#/////////////////////////////////////////
#ต้องเปลี่ยน library เพราะ
# -ไลบรารี                                            - ตรวจจับหลายคนได้ไหม - คำอธิบาย                                         
# | ---------------------------------------------------------------------------------------------- 
# - `cvzone.PoseModule.PoseDetector`                 - ❌ ไม่ได้    -ใช้ `mediapipe.pose` แบบรุ่นเดิม รองรับคนเดียว 
# - `mediapipe` แบบตั้งค่าปกติ (`Pose()`)               - ❌ ไม่ได้     - ตรวจได้แค่ 1 คนต่อเฟรม                         
# - **MediaPipe BlazePose GHUM รุ่นหลายคน** (ใหม่ล่าสุด) - ✅ ได้       - ต้องเรียกใช้แบบ advanced                       
# - **วิธีทางอ้อม: YOLO ตรวจจับคน + Pose แยกร่าง**        - ✅ ได้       - ตรวจคนด้วย YOLO แล้วใช้ Pose แยกแต่ละคน        


import cv2  # ใช้ OpenCV สำหรับอ่านกล้อง, จัดการภาพ และแสดงผล
from cvzone.PoseModule import PoseDetector  # คลาสจาก cvzone ใช้ตรวจจับท่าทางร่างกาย

detector = PoseDetector()  # สร้างตัวตรวจจับร่างกาย
cap = cv2.VideoCapture(1)  # เปิดกล้อง (ลองเปลี่ยนเป็น 0 หรือ 1 ถ้ากล้องไม่ติด)

while True:  # ลูปทำงานต่อเนื่อง
    success, img = cap.read()
    # อ่านภาพจากกล้อง (เฟรมต่อเฟรม)
    # success = True ถ้าอ่านได้, False ถ้าอ่านไม่ได้
    # img = ภาพ (numpy array)

    if not success:
        # ถ้าอ่านภาพไม่ได้
        print("ไม่สามารถอ่านภาพจากกล้องได้")
        break  # หยุดลูป

    img = detector.findPose(img)
    # ส่งภาพไปให้ detector วิเคราะห์และวาดจุด + เส้นโครงร่างบนร่างกาย

    lmList, bbox = detector.findPosition(img, bboxWithHands=True)
    # หาตำแหน่งจุด (landmarks) และกล่องครอบร่างกาย
    # lmList = list ของจุด เช่น [[id, x, y, z], …]
    # bbox = [x, y, w, h] กล่องสี่เหลี่ยมครอบร่างกาย
    # bboxWithHands=True = นับมือเข้าไปด้วย

    cv2.imshow("MyResult", img)
    # แสดงภาพที่มีโครงร่าง + จุดในหน้าต่างชื่อ "MyResult"

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # รอเช็คว่ามีคนกดคีย์บอร์ดไหม
        # ถ้ากด q จะหยุดลูป
        break

cap.release()  # ปิดกล้อง
cv2.destroyAllWindows()  # ปิดหน้าต่างที่เปิดไว้
print("ตรวจจับเสร็จสิ้น")  # แจ้งว่าเสร็จแล้ว
