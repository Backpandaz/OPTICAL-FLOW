# IMPORTAÇÕES
import cv2
import numpy as np

    
    
    



def lucasKanade():
    
    
    # Abrindo o vídeo
    videoCapObj = cv2.VideoCapture(0)
    



    # Parâmetros do Shi-Tomasi (pontos de interesse)
    shiTomasiCornerParams = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    # Parâmetros do Lucas-Kanade Optical Flow
    lucasKanadeParams = dict(
        winSize=(20, 20),
        maxLevel=10,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            0.03
        )
    )
    

     

    # Cores aleatórias para desenhar trajetórias
    RandomColors = np.random.randint(0, 255, (100, 3))


    # Lendo o primeiro frame do vídeo
    ret, frameFirst = videoCapObj.read()

    if not ret or frameFirst is None:
        print("Não foi possível ler o primeiro frame do vídeo.")
        return

    # Convertendo para escala de cinza
    frameGrayPrev = cv2.cvtColor(frameFirst, cv2.COLOR_BGR2GRAY)
    
    

    # Detectando cantos para trackear
    cornersPrev = cv2.goodFeaturesToTrack(
        frameGrayPrev,
        mask=None,
        **shiTomasiCornerParams
    )
    
    
    r,h,c,w = 250,90,400,125 # Declarando os valores para as variáveis
    track_window = (c,r,w,h)
    
    
    # Setando o ROI para o tracking
    roi = frameFirst[r:r+h, c:c+w]
    hsv_roi = cv2.cvtColor(frameFirst, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((120.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    
 
    # 10 iterações ou 1 ponto
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )


    # Máscara para desenhar o histórico de movimento
    mask = np.zeros_like(frameFirst)
    print("Iniciando rastreamento de movimentos... Pressione ESC para sair.")


    # Loop principal de processamento
    while True:
        ret, frame = videoCapObj.read()
        

        # Se não houver mais frames, encerra
        if not ret or frame is None:
            print("Fim do vídeo.")
            break
        

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



        # Se algo falhar no cálculo
        if cornersCur is None or foundStatus is None:
            continue
        
        

        # Selecionando apenas os pontos válidos
        cornersMatchedCur = cornersCur[foundStatus == 1]
        cornersMatchedPrev = cornersPrev[foundStatus == 1]
        
        

        # Se nenhum ponto foi rastreado com sucesso
        if len(cornersMatchedCur) == 0:
            frameGrayPrev = frameGrayCur.copy()
            cornersPrev = cv2.goodFeaturesToTrack(
                frameGrayPrev, mask=None, **shiTomasiCornerParams)
            mask = np.zeros_like(frame)
            continue
            
          
             
        #Criando o retangulo azul para seguir o objeto               
        if ret == True:
            hsv = cv2.cvtColor(frameFirst, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            
            
            # Meanshift para a nova localização 
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            
            
            # Desenhando o retângulo
            pts = cv2.boxPoints(ret)
            pts = np.int32(pts)
            img2 = cv2.polylines(frameFirst,[pts],True, 255,2)
            cv2.imshow('Pontos + retangulo',img2) 
            
            
        # Desenhando trajetórias dos pontos
        for i, (curCorner, prevCorner) in enumerate(
                zip(cornersMatchedCur, cornersMatchedPrev)):

            xCur, yCur = curCorner.ravel()
            xPrev, yPrev = prevCorner.ravel()

            mask = cv2.line(
                mask,
                (int(xCur), int(yCur)),
                (int(xPrev), int(yPrev)),
                RandomColors[i].tolist(),
                3        
         )
            

            frame = cv2.circle(
                frame,
                (int(xCur), int(yCur)),
                5,
                RandomColors[i].tolist(),
                -1
                
            )


        # Exibindo no VS Code
        cv2.imshow('Mascara', mask)
        
        
        # Imagem final combinando frame + histórico
        imgFinal = cv2.add(frame, mask,img2)
        
      
        # Controle do loop e saída
        key = cv2.waitKey(15)


        if key == 27:  # tecla ESC
            print("Encerrado pelo usuário.")
            break
        
        
        # Reiníciar o loop
        if key == ord('r'):
            print("Reiniciando a tela...")

                        # Zera o histórico
            mask = np.zeros_like(frame)

                    # Recalcula pontos de interesse
            frameGrayPrev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cornersPrev = cv2.goodFeaturesToTrack(
            frameGrayPrev,
            mask=None,
            **shiTomasiCornerParams) 
                
            
      
    # Atualizando para próxima iteração
    frameGrayPrev = frameGrayCur.copy()
    cornersPrev = cornersMatchedCur.reshape(-1, 1, 2)
        
        
    # Liberando recursos
    videoCapObj.release()
    cv2.destroyAllWindows()


  
    # DENSE FLOW
    videoCapObj = cv2.VideoCapture(0)
  

    if not  videoCapObj.isOpened():
        print("Erro ao abrir vídeo para Dense Optical Flow.")
        return

    ret, framePrev = videoCapObj.read()

    if not ret:
        return

    grayPrev = cv2.cvtColor(framePrev, cv2.COLOR_BGR2GRAY)

    hsvMask = np.zeros_like(framePrev)
    hsvMask[..., 1] = 255

    while True:
        ret, frame =  videoCapObj.read()

        if not ret or frame is None:
            break

        grayCur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            grayPrev, grayCur,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude, angle = cv2.cartToPolar(
            flow[..., 0], flow[..., 1])

        hsvMask[..., 0] = angle * 180 / np.pi / 2
        hsvMask[..., 2] = cv2.normalize(
            magnitude, None, 0, 255, cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(hsvMask, cv2.COLOR_HSV2BGR)     
        cv2.imshow("Dense Optical Flow", rgb)
        
      
        key = cv2.waitKey(20)

        if key == 27:
            break

        grayPrev = grayCur.copy()

    videoCapObj.release()
   
   

#FUNÇÃO FINAL
def executarleitor():
    lucasKanade()





#CHAMAR FUNÇÃO     
if __name__ == "__main__":
    executarleitor()
