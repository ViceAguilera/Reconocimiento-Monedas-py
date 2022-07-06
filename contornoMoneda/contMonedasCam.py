from logging import captureWarnings
from tkinter import N
from cv2 import cv2 
import numpy as np

def ordenarPuntos(puntos):
    n_puntos = np.concatenate([puntos[0],puntos[1],puntos[2],puntos[3]]).tolist()
    y_order = sorted(n_puntos, key = lambda n_puntos: n_puntos[1]) 
    x1_order = y_order[0:2]
    x1_order = sorted(x1_order,key = lambda x1_order: x1_order[0])
    x2_order = y_order[2:4]
    x2_order = sorted(x2_order,key = lambda x2_order: x2_order[0])
    return [x1_order[0], x1_order[1],x2_order[0], x2_order[1]]

def alineamiento(imagen, ancho, alto):
    imagen_alineada = None
    grises= cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY) #pasa la imagen a escala de grises
    tipoUmbral,umbral =cv2.threshold(grises,150,255,cv2.THRESH_BINARY) #umbral de imagen (asilar la imagen de su entorno)q
    contorno = cv2.findContours(umbral, cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)[0]
    contorno = sorted(contorno, key = cv2.contourArea, reverse = True)[:1]
    for c in contorno:
        epsilon = 0.01*cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c,epsilon,True)
        if len(aprox)==4:
            puntos = ordenarPuntos(aprox)
            puntos1 = np.float32(puntos)
            puntos2 = np.float32([[0,0], [ancho,0], [0,alto], [ancho,alto]])
            M = cv2.getPerspectiveTransform(puntos1,puntos2)
            imagen_alineada = cv2.warpPerspective(imagen, M, (ancho,alto))
    return imagen_alineada

capturarVideo=cv2.VideoCapture(0)

while True:
    tipocamara,Camara = capturarVideo.read()
    if tipocamara == False:
        break
    imagen_A6 = alineamiento(Camara, ancho = 480, alto = 640)
    if imagen_A6 is not None:
        puntos = []
        imagen_Grises = cv2.cvtColor(imagen_A6,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imagen_Grises,(5,5),1)
        _,umbral2 =cv2.threshold(blur,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
        contorno2= cv2.findContours(umbral2, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(imagen_A6,contorno2,-1,(255,0,0),2)
        suma1 = 0.0
        suma2 = 0.0

        for c_2 in contorno2:
            area = cv2.contourArea(c_2)
            momentos = cv2.moments(c_2)
            if(momentos["m00"]==0):
                momentos["m00"]=1.0
            x = int(momentos["m10"]/momentos["m00"])
            y = int(momentos["m01"]/momentos["m00"])

            if area<7800 and area>6500:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagen_A6,"10",(x,y),font,0.75,(0,0,0),2)
                suma1 = suma1+10
            if area<12524 and area>10000:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagen_A6,"100",(x,y),font,0.75,(0,0,0),2)
                suma1 = suma1+100
        
        cv2.imshow("imagen_A6",imagen_A6)
        cv2.imshow("Camara",Camara)
    if cv2.waitKey(1)==ord("q"):
        break
capturarVideo.release()
cv2.destroyAllWindows()