import cv2
import threading
import requests
import numpy as np
import base64
import io
from PIL import Image
import time

# Dirección de la API
url = "http://192.168.40.66:8000/detect_frame"

# Variables globales para comunicación entre hilos
last_frame = None
result_frame = None
lock = threading.Lock()

# Hilo de procesamiento (envío a la API)
def process_frames():
    global last_frame, result_frame
    while True:
        with lock:
            frame = last_frame.copy() if last_frame is not None else None

        if frame is not None:
            _, img_encoded = cv2.imencode('.jpg', frame)
            files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}

            try:
                response = requests.post(url, files=files, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    fuego = data["fuego_detectado"]
                    porcentaje = data["porcentaje_confianza"]

                    print(f"🔥 Fuego: {fuego} | Confianza: {porcentaje:.2f}%")

                    img_bytes = base64.b64decode(data["imagen"])
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    processed = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    with lock:
                        result_frame = processed
                else:
                    print(f"⚠️ Error en respuesta: {response.status_code}")
            except Exception as e:
                print("❌ Error al conectar con la API:", e)

        time.sleep(1)  # puedes ajustarlo a 0.5 o más para menor carga

# Iniciar cámara
cap = cv2.VideoCapture(0)

# Iniciar hilo
threading.Thread(target=process_frames, daemon=True).start()

print("🎥 Presiona ESC para salir...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Actualizar el frame más reciente para procesar
    with lock:
        last_frame = frame.copy()

    # Mostrar imagen procesada si existe
    with lock:
        if result_frame is not None:
            cv2.imshow("Detección remota", result_frame)
        else:
            cv2.imshow("Detección remota", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()