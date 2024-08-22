import cv2
import os
import imutils
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Reconocimiento Facial")

# Función para iniciar el proceso de captura y entrenamiento
def iniciar_proceso():
    personName = simpledialog.askstring("Nombre de la Persona", "Por favor, ingresa el nombre de la persona:")

    if not personName:
        messagebox.showwarning("Advertencia", "No ingresaste ningún nombre.")
        return

    # Crear la carpeta para almacenar las imágenes de la persona
    dataPath = '../data'
    personPath = os.path.join(dataPath, personName)

    if not os.path.exists(personPath):
        print('Carpeta creada: ', personPath)
        os.makedirs(personPath)

    # Captura de video
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Clasificador de rostros
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(personPath, f'rostro_{count}.jpg'), rostro)
            count += 1

        cv2.imshow('frame', frame)

        k = cv2.waitKey(1)
        if k == 27 or count >= 300:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Almacenar el modelo después de capturar las imágenes
    print("Iniciando entrenamiento del modelo...")

    # Segunda parte: Entrenamiento del modelo
    peopleList = os.listdir(dataPath)
    print('Lista de personas: ', peopleList)

    labels = []
    facesData = []
    label = 0

    for nameDir in peopleList:
        personPath = os.path.join(dataPath, nameDir)
        print('Leyendo las imágenes de:', nameDir)

        for fileName in os.listdir(personPath):
            print('Rostros:', nameDir + '/' + fileName)
            labels.append(label)
            facesData.append(cv2.imread(os.path.join(personPath, fileName), 0))

        label += 1

    # Crear el reconocedor de rostros y entrenar
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    print("Entrenando...")
    face_recognizer.train(facesData, np.array(labels))

    # Guardar el modelo entrenado
    face_recognizer.write('modeloLBPHFace.xml')
    print("Modelo almacenado...")
    messagebox.showinfo("Éxito", "El proceso de captura y entrenamiento ha finalizado.")

# Crear el botón para iniciar el proceso
btn_iniciar = tk.Button(root, text="Iniciar Proceso", command=iniciar_proceso)
btn_iniciar.pack(pady=20)

# Correr la interfaz gráfica
root.mainloop()
