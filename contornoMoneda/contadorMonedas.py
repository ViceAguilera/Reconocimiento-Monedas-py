from random import gauss
from cv2 import GaussianBlur, cv2
import numpy as np

valorGauss = 3
valorKernel = 3

original = cv2.imread('monedas2.png')
grises = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY) 
gaus = cv2.GaussianBlur(grises,(valorGauss,valorGauss),0) #Se realiza un suavisado de imagen 
canny = cv2.Canny(gaus,60,100) #Se reduce el ruido de imagen

kernel = np.ones((valorKernel,valorKernel),np.uint8) #Se trabaja con entero de 8 bytes para preparar el el contorno mayor 
cierre = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel) #Se trasnorma la forma de la imagen, Se elimina ruido  

contornos, jerarquia = cv2.findContours(cierre.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print("Monedas Encontradas: {}".format(len(contornos)))
cv2.drawContours(original,contornos,-1,(255,255,0))

#mostrar resultado 
cv2.imshow('Modenas en escala de grises',grises)
cv2.imshow('Modenas con suavisado gausiano',gaus)
cv2.imshow('Monedas con reduccion de ruido', canny)
cv2.imshow('Cierre', cierre)
cv2.imshow('Resultado', original)
cv2.waitKey(0)