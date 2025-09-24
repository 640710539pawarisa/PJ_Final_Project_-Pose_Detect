#วนเช็คกล้อง 3 ตัว

import cv2

for index in range(3):  # ลองตั้งแต่กล้อง index 0 ถึง 2
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"เปิดกล้อง index {index} ได้")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {index}", frame)
            cv2.waitKey(8000)  # แสดงผล 1 วินาที
            cv2.destroyAllWindows()
        else:
            print(f"กล้อง index {index} เปิดได้ แต่ไม่สามารถอ่านภาพได้")
        cap.release()
    else:
        print(f"ไม่สามารถเปิดกล้อง index {index} ได้")

#recap วนเช็คกล้อง 3 ตัว ได้ 0 คือกล้อง snapcam ไม่ควรใช้, 1 คือกล้อง  notebook  2 คือกล้อง webcam