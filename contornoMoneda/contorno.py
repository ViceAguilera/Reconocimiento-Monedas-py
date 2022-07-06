import cv2
from cv2 import threshold
from cv2 import findContours

imagen = cv2.imread('contorno.jpg')
grises= cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY) #pasa la imagen a escala de grises
_,umbral =cv2.threshold(grises,100,255,cv2.THRESH_BINARY) #umbral de imagen (asilar la imagen de su entorno)
contorno,jerarquia = cv2.findContours(umbral, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imagen,contorno,-1,(251, 63, 52),3)

#mostrar
cv2.imshow('original',imagen)
#cv2.imshow('Imagen en gris',grises)
#cv2.imshow('Imagen Umbral',umbral)
cv2.waitKey(0)
cv2.destroyAllWindows()



