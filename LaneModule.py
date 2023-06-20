import cv2
import numpy as np 
import utils

curveList = []
avgVal = 10 
def getLaneCurve(img,display = 2):
    imgCopy = img.copy()
    imgResult = img.copy()
    #Buoc1
    imgThres = utils.thresholding(img)
    #Buoc2
    hT, wT, c = img.shape
    points = utils.valTrackBars()
    imgWarp = utils.warpImg(imgThres,points,wT,hT)
    imgWarpPoints = utils.drawPoints(img,points)

    #Buoc3
    midPoint, imgHist,_ = utils.getHistogram(imgWarp,minPer = 0.5,display = True,region =4)
    basePoint, imgHist, maxValue = utils.getHistogram(imgWarp,minPer = 0.9,display = True)
    curveRaw = basePoint - midPoint

    #Buoc 4
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))
    #Buoc 5
    if display != 0:
       imgInvWarp = utils.warpImg(imgWarp, points, wT, hT,inv = True)
       imgInvWarp = cv2.cvtColor(imgInvWarp,cv2.COLOR_GRAY2BGR)
       imgInvWarp[0:hT//3,0:wT] = 0,0,0
       imgLaneColor = np.zeros_like(img)
       imgLaneColor[:] = 0, 255, 0
       imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
       imgResult = cv2.addWeighted(imgResult,1,imgLaneColor,1,0)
       midY = 450
       cv2.putText(imgResult,str(curve),(wT//2-80,85),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)
       cv2.line(imgResult,(wT//2,midY),(wT//2+(curve*3),midY),(255,0,255),5)
       cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY-25), (wT // 2 + (curve * 3), midY+25), (0, 255, 0), 5)
       for x in range(-30, 30):
           w = wT // 20
           cv2.line(imgResult, (w * x + int(curve//50 ), midY-10),
                    (w * x + int(curve//50 ), midY+10), (0, 0, 255), 2)
    if display == 2:
       imgStacked = utils.stackImages(0.7,([img,imgWarpPoints,imgWarp],
                                         [imgHist,imgLaneColor,imgResult]))
       cv2.imshow('ImageStack',imgStacked)
    elif display == 1:
       cv2.imshow('Resutlt',imgResult)
    if curve > 100 : 
        curve == 100
    if curve < -100 :
        curve == -100

    return curve,maxValue


if __name__ == '__main__':
    # cap = cv2.VideoCapture('4107994901747421288.mp4')
    # initialTrackbarVals = [124,80,43,215]
    cap = cv2.VideoCapture('vid1.mp4')
    initialTrackbarVals = [102, 80, 20, 214 ]
    utils.initializeTrackbars(initialTrackbarVals)
    frameCouter = 0 
    while True:
        frameCouter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCouter:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            frameCouter = 0 
        success, img = cap.read()

        img = cv2.resize(img,(480,240))
        curve,maxV = getLaneCurve(img,display = 1)
        #cv2.imshow('Video',img)
        cv2.waitKey(1)