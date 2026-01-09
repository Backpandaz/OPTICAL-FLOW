 # PROGRAMA DE LEITURA DE MOVIMENTOS – versão funcional
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt



    
    # Caminho ABSOLUTO direto do vídeo (correto para Windows)
    # videoPath = r'C:\Users\ANA\Desktop\Primeiro-passo-c\Movimentos\teste.mp4'
    

    



    # Parâmetros do Shi-Tomasi (pontos de interesse)
shiTomasiCornerParams = dict(
maxCorners=20,
qualityLevel=0.3,
minDistance=30,
blockSize=30
    )
    

 
    # Abrindo o vídeo
videoCapObj = cv2.VideoCapture(0)       

    # Parâmetros do Lucas-Kanade Optical Flow
lucasKanadeParams = dict(
        winSize=(15, 15),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            0.03
        )
    )
     

    # Cores aleatórias para desenhar trajetórias
color = np.random.randint(0, 255, (100, 3))

    # Lendo o primeiro frame do vídeo
ret, frameFirst = videoCapObj.read()


    # Convertendo para escala de cinza
frameGrayPrev = cv2.cvtColor(frameFirst, cv2.COLOR_BGR2GRAY)
cornersPrev = cv2.cvtColor(frameFirst, cv2.COLOR_BGR2GRAY)   
    

    # Detectando cantos para trackear
cornersPrev = cv2.goodFeaturesToTrack( frameGrayPrev,mask=None,**shiTomasiCornerParams)
    

    # Máscara para desenhar o histórico de movimento
mask = np.zeros_like(frameFirst)

print("Iniciando rastreamento de movimentos... Pressione ESC para sair.")

    # Loop principal de processamento
while True:
        
        ret, frame = videoCapObj.read()


        # Frame atual em cinza
        frameGrayCur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculando optical flow entre frames
        cornersCur, foundStatus, err = cv2.calcOpticalFlowPyrLK(
            frameGrayPrev,
            frameGrayCur,
            cornersPrev,
            None,
            **lucasKanadeParams
        )

       

        # Selecionando apenas os pontos válidos
        cornersMatchedCur = cornersCur[foundStatus == 1]
        cornersMatchedPrev = cornersPrev[foundStatus == 1]
      

            
            

        # Desenhando trajetórias dos pontos
        for i, (curCorner, prevCorner) in enumerate(
                zip(cornersMatchedCur, cornersMatchedPrev)):

            xCur, yCur = curCorner.ravel()
            xPrev, yPrev = prevCorner.ravel()
            mask = cv2.line(mask,
                (int(xCur), int(yCur)),
                (int(xPrev), int(yPrev)),
                color[i].tolist(), 2)

            frame = cv2.circle(frame,
                            (int(xCur), int(yCur)),
                            5,
                            color[i].tolist(), -1)
        img = cv2.add(frame,mask)
        cv2.imshow('frame',img)

                
        
        # Controle do loop e saída
        key = cv2.waitKey(15)

        if key == 27:  # tecla ESC
            print("Encerrado pelo usuário.")
            break

        # Atualizando para próxima iteração
        frameGrayPrev = frameGrayCur.copy()
        cornersPrev = cornersMatchedCur.reshape(-1, 1, 2)
    
    # Liberando recursos
cv2.destroyAllWindows()
videoCapObj.release()


# def denseOpticalFlow():
#     """
#     Exemplo simples de Dense Optical Flow para visualização geral de movimento
#     (opcional para você usar depois)
#     """

#     videoCapObj = cv2.VideoCapture(0)
  

#     if not  videoCapObj.isOpened():
#         print("Erro ao abrir vídeo para Dense Optical Flow.")
#         return

#     ret, framePrev = videoCapObj.read()

#     if not ret:
#         return

#     grayPrev = cv2.cvtColor(framePrev, cv2.COLOR_BGR2GRAY)

#     hsvMask = np.zeros_like(framePrev)
#     hsvMask[..., 1] = 255

#     while True:
#         ret, frame =  videoCapObj.read()

#         if not ret or frame is None:
#             break

#         grayCur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         flow = cv2.calcOpticalFlowFarneback(
#             grayPrev, grayCur,
#             None,
#             0.5, 3, 15, 3, 5, 1.2, 0
#         )

#         magnitude, angle = cv2.cartToPolar(
#             flow[..., 0], flow[..., 1])

#         hsvMask[..., 0] = angle * 180 / np.pi / 2
#         hsvMask[..., 2] = cv2.normalize(
#             magnitude, None, 0, 255, cv2.NORM_MINMAX)

#         rgb = cv2.cvtColor(hsvMask, cv2.COLOR_HSV2BGR)     

#         cv2.imshow("Dense Optical Flow", rgb)
      
#         key = cv2.waitKey(20)

#         if key == 27:
#             break

#         grayPrev = grayCur.copy()

#     videoCapObj.release()
#     cv2.destroyAllWindows()


# Chamada principal – necessária para rodar no VS Code
# if __name__ == "__main__":
#     lucasKanade()
    
# if __name__ == "__main__":
#     denseOpticalFlow()



    # denseOpticalFlow()
    
    

