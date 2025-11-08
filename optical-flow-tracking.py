# https://www.pyimagesearch.com/wp-content/uploads/2014/11/opencv_crash_course_camshift.pdf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def lucasKanade():
    root = os.getcwd()
    video_path = "/media/t2.mp4"
    
    video = cv2.VideoCapture(video_path)

    shiTomasiCornerParams = dict(maxCorners=5,
                                qualityLevel=0.8,
                                minDistance=100,
                                blockSize=7)

    lucasKanadeParams = dict(   winSize = (15, 15),
                                maxLevel = 2,
                                criteria = (cv2.TERM_CRITERIA_EPS|
                                        cv2.TERM_CRITERIA_COUNT,
                                        10,
                                        0.03))

    randomColors = np.random.randint(0,255,(100, 3))

    _, frameFirst = video.read()
    frameGrayPrev = cv2.cvtColor(frameFirst, cv2.COLOR_BGR2GRAY)
    cornersPrev = cv2.goodFeaturesToTrack(  frameGrayPrev,
                                            mask=None,
                                            **shiTomasiCornerParams)

    print(f'cornersPrev = \n{cornersPrev}\n')
    print(f'cornersPrev = \n{cornersPrev.dtype}\n')
# cornersPrev =
# [[[1740.  194.]]
#    
# [[1848.  196.]]
# 
# [[ 657.  173.]]]
# 
# 


    #newcorner = cv2.selectROI(frameFirst)
    #newcorner = np.array([newcorner[0], newcorner[1] ], dtype=np.float32)
    #cornersPrev.append([[1048, 855]])
    #cornersPrev = np.append(cornersPrev, newcorner, axis=None)

    #cornersPrev = np.array([[[newcorner[0], newcorner[1] ]]], dtype=np.float32)
    print(f'cornersPrev = \n{cornersPrev}\n')
    
    mask = np.zeros_like(frameFirst)

    
    while True:
        
        _, frame = video.read()
        frame_bn_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        key = cv2.waitKey(1) & 0xFF

        current_corners, found_status, _ = cv2.calcOpticalFlowPyrLK(
                frameGrayPrev, frame_bn_current, cornersPrev, None,
                **lucasKanadeParams)

        if current_corners is not None:
            cornersMatchedCur = current_corners[found_status==1]
            cornersMatchedPrev = cornersPrev[found_status==1]

        for i,(curCorner, prevCorner) in enumerate(zip(
                cornersMatchedCur, cornersMatchedPrev)):
            xCur, yCur = curCorner.ravel()
            xPrev, yPrev = prevCorner.ravel()
            mask = cv2.line(mask, ( int(xCur), int(yCur) ), 
                ( int(xPrev), int(yPrev) ), randomColors[1].tolist(), 2)
            print(f'i is {i}')
            frame = cv2.circle(frame, (int(xCur), int(yCur)), 5,
                randomColors[i].tolist(), -1)
            img = cv2.add(frame, mask)
        
        cv2.imshow('Video', frame)
        cv2.waitKey(15)
        frameGrayPrev = frame_bn_current.copy()
        cornersPrev = cornersMatchedCur.reshape(-1, 1, 2)

    video.release()
    cv2.closeAllWindows()





lucasKanade()
