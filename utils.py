import cv2 
import numpy as np 

def thresholding(img):
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([80,0,0])
    upperWhite = np.array([172,255,255])
    maskWhite = cv2.inRange(imgHSV,lowerWhite,upperWhite)
    # maskWhite = cv2.inRange(img, (130, 130, 130), (255, 255, 255))
    # kernel = np.ones((3, 3), np.uint8)
    # maskWhite = cv2.erode(maskWhite, kernel, iterations=5)
    # maskWhite = cv2.dilate(maskWhite, kernel, iterations=9)
    return maskWhite

def warpImg(img,points,w,h,inv = False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    if inv: 
        matrix = cv2.getPerspectiveTransform(pts2,pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp =cv2.warpPerspective(img,matrix,(w,h))
    return imgWarp

def nothing(a):
    pass

def initializeTrackbars(initialTrackbarVals,wT = 480, hT = 852):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars",360,240)
    cv2.createTrackbar("Width Top","Trackbars",initialTrackbarVals[0],wT//2, nothing)
    cv2.createTrackbar("Height Top","Trackbars",initialTrackbarVals[1],hT,nothing)
    cv2.createTrackbar("Width Bottom","Trackbars",initialTrackbarVals[2],wT//2,nothing)
    cv2.createTrackbar("Height Bottom","Trackbars",initialTrackbarVals[3],hT,nothing)

def valTrackBars(wT = 480, hT = 240):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    HeightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    HeightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop,HeightTop),(wT - widthTop,HeightTop),(widthBottom,HeightBottom),(wT-widthBottom,HeightBottom)])
    return points

def drawPoints(img,points):
    for x in range(4):
        cv2.circle(img,(int(points[x][0]),int(points[x][1])),8,(0,0,0),cv2.FILLED)
    return img

def getHistogram(img,minPer = 0.1,display = False,region = 1):

    if region == 1 :
        histValue = np.sum(img, axis = 0)
    else:
        histValue = np.sum(img[img.shape[0]//region:,:],axis = 0)

    maxValue = np.max(histValue) #TÌM GIÁ TRỊ LỚN NHẤT
    minValue = minPer * maxValue #GIÁ TRỊ NGƯỠNG

    indexArray  = np.where(histValue >= minValue) #TẤT CẢ GIÁ TRỊ LỚN HƠN HOẶC BẰNG GIÁ TRỊ TỐI THIỂU
    basePoint = int(np.average(indexArray)) #TRUNG BÌNH TẤT CẢ CÁC GIÁ TRỊ CHỈ SỐ TỐI ĐA

    if display: 
        imgHist = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
        for x,intensity in enumerate(histValue):
            cv2.line(imgHist,(x,img.shape[0]),(x,img.shape[0] - intensity//255//region),(255,0,255),1)
            cv2.circle(imgHist,(basePoint,img.shape[0]),20,(0,255,255),cv2.FILLED)
        return basePoint,imgHist,maxValue
    return basePoint

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

