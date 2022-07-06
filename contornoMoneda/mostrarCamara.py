import cv2 as cv

capturarVideo=cv.VideoCapture(0)
if not capturarVideo.isOpened():
    print("No se encontro camara")
    exit()
while True:
    tipocamara,Camara=capturarVideo.read()

    cv.imshow("Camara",Camara)
    if cv.waitKey(1)==ord("q"):
        break

capturarVideo.release()
cv.destroyAllWindows()