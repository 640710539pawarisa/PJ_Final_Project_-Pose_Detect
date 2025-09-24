#ตรวจจับใบหน้าและดวงตา จากกล้อง 

import cv2
import numpy as np  
import matplotlib.pyplot as plt
import os

cap = cv2.VideoCapture(2)

#เพื่มส่วนนี้****
#อ่านไฟล์ xml

# path แบบ absolute
face_cascade = cv2.CascadeClassifier(r"C:\Users\aepaw\my_project\libary_detect\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"C:\Users\aepaw\my_project\libary_detect\haarcascade_eye_tree_eyeglasses.xml")

while True:#ใช้ while เพื่อให้หน้าต่างไม่ปิดอัตโนมัติ อ่านทีละเฟรมไ
    check, frame = cap.read() #รับภาพจากวีดีโอ frame ต่อ frame , 1 ภาพก็เป็น 1 frame
    if check == True:
        #เพื่มส่วนนี้****
        #เปลี่ยนสีเป็น GrayScale
        gray_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)#แปลงvideo จาก BGR เป็น GrayScale เพื่อใช้ในการตรวจจับใบหน้า ,frame คือภาพสี
        #ตัวแปร frame คือ ภาพสี เราจะเอาไปแสดงผล, gray_frame คือ ภาพเปลี่ยนสีเป็น GrayScale คือเอาไว้ประมวลผลในการตรวจจับใบหน้า
        
        #ตรวจจับใบหน้า
        face_detect = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=3)#ตรวจจับใบหน้า ,face_detect คือตัวแปรที่เก็บค่าของการตรวจจับใบหน้า
        #ตรวจจับดวงตา
        eye_detect = eye_cascade.detectMultiScale(gray_frame,scaleFactor=1.2,minNeighbors =7)#ตรวจจับดวงตา, eye_detect คือตัวแปรที่เก็บค่าของการตรวจจับดวงตา,
        
        #แสดงตำแหน่งใบหน้า
        
#*************************
# ทำloopซ้อนloop เพื่อแสดงตำแหน่งใบหน้าและดวงตาที่ตรวจพบ 
# วนลูปในตัวแปร face_detectและ eye_detect โดยตัวแปร x,y,w,h และ a,b,c,d คือตัวแปรที่เก็บค่าของตำแหน่งใบหน้าและดวงตาไม่ให้ซ้ำกัน***
        for (x,y,w,h)  in face_detect: #วนลูปเพื่อแสดงตำแหน่งใบหน้าที่ตรวจพบ
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),thickness= 5) #แสดงตำแหน่งใบหน้าที่ตรวจพบ
            for (a,b,c,d) in eye_detect: #วนลูปเพื่อแสดงตำแหน่งดวงตาที่ตรวจพบ
                cv2.rectangle(frame, (a, b), (a + c, b + d),(255,0,0),thickness= 5) #แสดงตำแหน่งดวงตาที่ตรวจพบ
            
        cv2.imshow("Output", frame) #แสดงผล
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break
    else:
        break

cap.release() #ปิดวีดีโอ
cv2.destroyAllWindows()#ปิดหน้าต่าง