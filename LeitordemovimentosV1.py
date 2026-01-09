# PROGRAMA DE LEITURA DE MOVIMENTOS
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def lucasKanade():
    
    #ler a imagem
    root = os.getcwd()
    videoPath = os.path.join(root,r'C:\Users\ANA\Desktop\Primeiro-passo-c\Movimentos\teste.mp4')
    videoCapObj = cv2.VideoCapture(videoPath)
    
    #parametros
    shiTomasiCornerParams = dict(maxCorners=20,
                                 qualityLevel=0.3,
                                 minDistance=50,
                                 blockSize=7 )
    
    lucasKanadeParams = dict(winSize =(15,15),
                             maxlevel=2,
                             criteria = (cv2.TERM_CRITERIA_EPS|cv2.
                                         TERM_CRITERIA_COUNT,10,0.03))
    
    RandomColors = np.random.randint(0,255,(100, 3))
    
    # Procurando o que trackear
    
    _, frameFirst = videoCapObj.read()
    frameGrayPrev = cv2.cvtColor(frameFirst, cv2.COLOR_BGR2GRAY)
    cornersPrev = cv2.goodFeaturesToTrack(frameGrayPrev,
    mask=None, **shiTomasiCornerParams)
    mask = np.zeros_like(frameFirst)
    
    
    #LOOP para cada frame do vídeo
    
    while True:
        _, frame = videoCapObj.read()
        frameGrayCur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cornersCur, foundStatus,_  = cv2.calcOpticalFlowPyrLK (frameGrayPrev, frameGrayCur, cornersPrev, None, 
        **lucasKanadeParams)
         
        if  cornersCur is not None:
            cornersMatchedCur = cornersCur[foundStatus==1]
            cornersMatchedPrev = cornersPrev[foundStatus==1]
            
        for i,(curCorner,prevCorner) in enumerate(zip
        (cornersMatchedCur, cornersMatchedPrev)):
            xCur, yCur = curCorner.ravel()
            xPrev, yPrev = prevCorner.ravel()
            mask = cv2.line(mask,(int(xCur),int(yCur)),(int(xPrev),int(yPrev)),RandomColors[i].tolist(),2)
            frame = cv2.circle(frame,(int(xCur), int(yCur)),5, RandomColors[i].tolist(),-1)
            img = cv2.add(frame,mask)
            
        cv2.imshow('video', img)
        cv2.waitKey(15)
        frameGrayPrev = frameGrayCur.copy()
        cornersPrev = cornersMatchedCur.reshape(-1,1,2) 
        
def denseOpticalFlow(): 
    #ler a imagem
    root = os.getcwd()
    videoPath = os.path.join(root, r'C:\Users\ANA\Desktop\Primeiro-passo-c\Movimentos\teste.mp4')
    videoCapObj = cv2.VideoCapture(videoPath)
            
               
    
   

# def camera():
#     # catura do vídeo
    # cap = cv2.VideoCapture(0)

#     # Primeiro frame do vídeo
#     ret,frame = cap.read()

#     # Localização inicial da janela

#     r,h,c,w = 250,90,400,125 # simply hardcoded the values
#     track_window = (c,r,w,h)

#     # Tracking
#     roi = frame[r:r+h, c:c+w]
#     hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
#     roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
#     cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

#     # iterações
#     term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
#     while(1):
        
#         ret ,frame = cap.read()
        
        
#         if ret == True:
#             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#             dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            
#             # Meanshift para nova localização
#             ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            
#             # Retangulo
#             x,y,w,h = track_window
#             img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
#             cv2.imshow('img2',img2)

#         if cv2.waitKey(1) & 0xff == ord('q'): break
        
    
#     cv2.destroyAllWindows()
#     cap.release()

