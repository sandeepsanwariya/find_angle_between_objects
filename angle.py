import cv2
import imutils
import math
import numpy as np
cap=cv2.VideoCapture(0)


def angle_np(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang

while True:
    points = []
    point1 = []
    point2 = []

    success,img=cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #red
    l_r = np.array([0, 102,  72])
    u_r = np.array([7, 255 ,255])

    #yellow

    l_y = np.array([23 , 94 ,168])
    u_y = np.array([78 ,255 ,255])

    #green
    l_g = np.array([46 ,108 , 21])
    u_g = np.array([69, 255 ,255])

    #blue
    l_b = np.array([94, 74 ,86])
    u_b = np.array([150, 255, 188])

    mask1 = cv2.inRange(hsv,l_b,u_b)
    mask2 = cv2.inRange(hsv, l_g, u_g)


    kernel=np.zeros((5,5),np.uint8)

    dil=cv2.dilate(mask1,kernel,iterations=1)
    ero=cv2.erode(dil,kernel,iterations=1)

    fil=cv2.GaussianBlur(mask1,(3,3),0)
    ret,thresh=cv2.threshold(fil,127,255,0)



    dil2=cv2.dilate(mask2,kernel,iterations=1)
    ero2=cv2.erode(dil,kernel,iterations=1)

    fil2=cv2.GaussianBlur(mask2,(3,3),0)
    ret2,thresh2=cv2.threshold(fil2,127,255,0)




    cnts1=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts1 =imutils.grab_contours(cnts1)

    cnts2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)


    for c in cnts1:
        area1=cv2.contourArea(c)
        if area1>5000:
            cv2.drawContours(img,[c],-1,(0,255,0),3)

            M=cv2.moments(c)

            cx=int(M['m10']/M['m00'])
            cy=int(M['m01']/M['m00'])
            point1 = (cx, cy)

            cv2.circle(img,(cx,cy),7,(255,255,255),-1)
            cv2.putText(img,"1",(cx-20,cy-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

    for c in cnts2:
        area2=cv2.contourArea(c)
        if area2>5000:
            cv2.drawContours(img,[c],-1,(0,255,0),3)

            M=cv2.moments(c)

            cx=int(M['m10']/M['m00'])
            cy=int(M['m01']/M['m00'])
            point2 = (cx, cy)
            cv2.circle(img,(cx,cy),7,(255,255,255),-1)
            cv2.putText(img,"2",(cx-20,cy-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

    if len(point2) > 1 and len(point1) > 1:
        # point3=point1[1],point2[0]
        point3 = point2[0] + 200, point2[1]
        cv2.putText(img, "3", (point2[0] + 200 - 20, point2[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        points = (point1, point3, point2)
        #cu = getangle(points)
        #cu = angle_between(point1, point2)
        cu=angle_np(point1,point2,point3)
        cv2.circle(img, point3, 7, (255, 255, 255), -1)
        cv2.line(img, point3, point2, (0, 255, 0), 2)
        cv2.line(img, point2, point1, (0, 255, 0), 2)
        cv2.putText(img, str(cu), (point2[0] - 20, point2[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),
                    3)


    cv2.imshow("tracking", img)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
