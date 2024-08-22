import tkinter as tk
from tkinter import messagebox
import subprocess
import os
from dotenv import load_dotenv, set_key

# Cargar variables del archivo .env
load_dotenv()

def run_script(script_name, callback=None):
    try:
        # Ejecutar el script
        result = subprocess.run(['python', script_name], check=True)
        # Llamar al callback si se proporciona
        if callback:
            callback()
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error al ejecutar {script_name}: {e}")

def run_face_scanner_and_trainer():
    def on_scanner_complete():
        # Ejecutar el script del entrenador después de que el escáner termine
        run_script('entrenandoRF.py')
    
    # Ejecutar el script del escáner de rostros
    run_script('capturandoRostros.py', callback=on_scanner_complete)

def update_person_name():
    new_name = person_name_entry.get()
    if new_name:
        set_key('.env', 'PERSON_NAME', new_name)
        messagebox.showinfo("Éxito", f"Nombre de persona actualizado a: {new_name}")
    else:
        messagebox.showwarning("Advertencia", "El nombre de la persona no puede estar vacío.")

def create_main_window():
    # Crear la ventana principal
    window = tk.Tk()
    window.title("Menú Principal")

    # Crear un marco para el campo de entrada del nombre de la persona
    frame_name = tk.Frame(window)
    frame_name.pack(pady=10)

    # Etiqueta y campo de entrada para el nombre de la persona
    lbl_person_name = tk.Label(frame_name, text="Nombre de la persona:")
    lbl_person_name.pack(side=tk.LEFT, padx=5)
    
    global person_name_entry
    person_name_entry = tk.Entry(frame_name)
    person_name_entry.pack(side=tk.LEFT, padx=5)
    
    # Botón para actualizar el nombre de la persona
    btn_update_name = tk.Button(frame_name, text="Actualizar Nombre", command=update_person_name)
    btn_update_name.pack(side=tk.LEFT, padx=5)

    # Crear los botones para ejecutar los scripts
    btn_scan_and_train = tk.Button(window, text="Escanear y Entrenar", command=run_face_scanner_and_trainer)
    btn_reconocimiento = tk.Button(window, text="Reconocimiento Facial", command=lambda: run_script('reconocimientofacial.py'))


    # Agregar los botones a la ventana

    btn_reconocimiento.pack(pady=10)
    btn_scan_and_train.pack(pady=10)  # Agregar el nuevo botón a la ventana

    # Cargar el nombre actual de la persona en el campo de entrada
    current_name = os.getenv('PERSON_NAME', '')
    person_name_entry.insert(0, current_name)

    # Iniciar el bucle principal de la interfaz gráfica
    window.mainloop()

if __name__ == "__main__":
    create_main_window()
