#Motion Tracker basic48
#ตรวจจับการเคลื่อนไหว Motion Detection ของvideo basic47
#ใช้ความรู้จาก basic46
import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(1) #เปิดวีดีโอ

#แยกส่วนเพื่อcheck ทีละเฟรม
#หาผลต่างกับเฟรมก่อนหน้า และเฟรมหลัง ว่ามีการคาดเคลื่อนไหม มันจะถือว่าเป็นการเคลื่อนไหว****
check, frame1 = cap.read()
check, frame2 = cap.read()#check คือเช็คว่าอ่านภาพได้ไหม,frame คือภาพ

while (cap.isOpened()):#ใช้ while เพื่อให้หน้าต่างไม่ปิดอัตโนมัติ
    check, frame = cap.read() #รับภาพจากวีดีโอ frame ต่อ frame , 1 ภาพก็เป็น 1 frame, check คือเช็คว่าอ่านภาพได้ไหม,frame คือภาพ
    if check == True:#ถ้าอ่านได้จะเป็น True
        motiondiff = cv2.absdiff(frame1,frame2) #ใช้ในการหาการเคลื่อนไหว ใช้ในการหาการเคลื่อนไหว เก็บลงตัวแปร ชื่อ motiondiff
        #ทำเป็น GrayScale
        gray = cv2.cvtColor(motiondiff,cv2.COLOR_BGR2GRAY)
        #ทำเป็น threshold คือการแปลงภาพเทาเป็นภาพbinary
        
        #แยกสิ่งที่อยู่ในภาพก่อน
        #ใช้ในการหาเส้น Contours
        blur = cv2.GaussianBlur(gray,(5,5),0)#cv2.GaussianBlur(ภาพ,ขนาดตัวกรอง หรือ kernel size ,ค่าsigma) 
        #ค่า sigma ปรับเปลี่ยนค่าได้ เช่น 1
        
        #ทำเป็น threshold คือการแปลงภาพเทาเป็นภาพbinary
        thresh , result = cv2.threshold(blur,15,255,cv2.THRESH_BINARY)#ใช้ในการหาเส้น Contours
        
        #เติมเส้นที่ขาดหาย ภาพคนที่เรา หรือ คือการขยายพื้นที่ของเส้น Contours
        dilation = cv2.dilate(result, None, iterations=3) #ใช้ในการหาเส้น Contours ,dilation คือการขยายของเส้น Contours
        
        #ใช้ในการหาเส้น Contours
        contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#ใช้ในการหาเส้น Contours
        #contours ใส่sคือ contoursทั้งหมด**
        
        #เพิ่มส่วนนี้****
        #วาดสี่เหลี่ยมในวัตถุที่กำลังเคลื่อนไหวหรือเคลื่อนที่
        for contour in contours:#ใช้ในการวาดเส้น Contours  ,contour แต่ละจุดที่เราไปดึงมา ,contours คือcontoursทั้งหมด
            
            #กำหนด ขนาดของสี่เหลี่ยม ที่จะวาด
            if cv2.contourArea(contour) < 2000:#contourArea คือการหาพื้นที่ของ contour ,ถ้าต้องการขอบเขตดูน้อยกว่านั้นให้เพิ่มค่ามากกว่านั้น
                continue
            x,y,w,h = cv2.boundingRect(contour) #.boundingRect คือขอบเขตของพื้นที่วาด contourขึ้นมาแล้วก็โยน contour ไปเข้า
            cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),thickness=2)#วาดสี่เหลี่ยม ,frame1 คือภาพที่จะวาด ,ตําแหน่ง ,ขนาด ,สี ,ขนาดของเส้นสี่เหลี่ยม
        
        cv2.imshow("Output", frame1) #แสดงภาพ
        #ไปยังเฟรมถัดไป
        frame1 = frame2
        check, frame2 = cap.read()
        if cv2.waitKey(80) & 0xFF == ord("q"):
                break
    else:    
        break
    
    
cap.release()#ปิดวีดีโอ
cv2.destroyAllWindows()#ปิดหน้าต่าง,คืนค่าเครื่อง