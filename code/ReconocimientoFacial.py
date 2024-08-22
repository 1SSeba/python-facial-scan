import cv2
import os

dataPath = '../data' 
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Configurar la resolución de captura a 1080p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def draw_text_with_border(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2, border_thickness=3, border_color=(0, 0, 0)):
    """
    Dibuja texto con un borde alrededor para mejor visibilidad.
    """
    for i in range(border_thickness):
        cv2.putText(img, text, (position[0] - i, position[1] - i), font, font_scale, border_color, thickness, cv2.LINE_AA)
        cv2.putText(img, text, (position[0] + i, position[1] - i), font, font_scale, border_color, thickness, cv2.LINE_AA)
        cv2.putText(img, text, (position[0] - i, position[1] + i), font, font_scale, border_color, thickness, cv2.LINE_AA)
        cv2.putText(img, text, (position[0] + i, position[1] + i), font, font_scale, border_color, thickness, cv2.LINE_AA)

    cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

# Crear una ventana que se puede redimensionar
cv2.namedWindow('Reconocimiento Facial', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener el tamaño actual de la ventana
    height, width = frame.shape[:2]

    # Redimensionar el marco del video para que se ajuste a la ventana
    frame = cv2.resize(frame, (width, height))

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
            cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 1)  # Borde extra para mejor visibilidad
        else:
            draw_text_with_border(frame, 'Desconocido', (x, y-20), font_scale=0.8, color=(0, 0, 255), border_thickness=2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 0, 255), 1)  # Borde extra para mejor visibilidad
    
    # Añadir barra negra al final del video con un borde superior
    frame_height, frame_width, _ = frame.shape
    cv2.rectangle(frame, (0, frame_height - 50), (frame_width, frame_height), (0, 0, 0), -1)  # Barra negra
    cv2.rectangle(frame, (0, frame_height - 52), (frame_width, frame_height - 50), (255, 255, 255), -1)  # Borde blanco superior

    # Añadir el texto sobre la barra negra
    draw_text_with_border(frame, 'Presiona ESC para cerrar', (10, frame_height - 20), font_scale=0.8, color=(255, 255, 255), border_thickness=2)

    cv2.imshow('Reconocimiento Facial', frame)
    k = cv2.waitKey(1)
    if k == 27:  # Esc key to stop
        break

cap.release()
cv2.destroyAllWindows()