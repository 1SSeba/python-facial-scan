import cv2
import os

dataPath = '../data'  # Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

# Crear el reconocedor de rostros
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Leyendo el modelo
face_recognizer.read('modeloLBPHFace.xml')

# Inicializar la cámara web o archivo de video
# Descomentar la línea correspondiente según tu uso
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Para cámara web
# cap = cv2.VideoCapture('Video.mp4')  # Para archivo de video

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def draw_text_with_border(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2, border_thickness=3, border_color=(0, 0, 0)):
    """
    Dibuja texto con un borde alrededor para mejor visibilidad.
    """
    # Dibuja el texto en el borde
    for i in range(border_thickness):
        cv2.putText(img, text, (position[0] - i, position[1] - i), font, font_scale, border_color, thickness, cv2.LINE_AA)
        cv2.putText(img, text, (position[0] + i, position[1] - i), font, font_scale, border_color, thickness, cv2.LINE_AA)
        cv2.putText(img, text, (position[0] - i, position[1] + i), font, font_scale, border_color, thickness, cv2.LINE_AA)
        cv2.putText(img, text, (position[0] + i, position[1] + i), font, font_scale, border_color, thickness, cv2.LINE_AA)

    # Dibuja el texto en el centro
    cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        # LBPHFace
        if result[1] < 70:
            name = imagePaths[result[0]]
            draw_text_with_border(frame, name, (x, y-25), font_scale=1.1, color=(255, 255, 255), border_thickness=2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            draw_text_with_border(frame, 'Desconocido', (x, y-20), font_scale=0.8, color=(0, 0, 255), border_thickness=2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:  # Tecla 'Esc' para salir
        break

cap.release()
cv2.destroyAllWindows()
