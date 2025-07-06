import requests
import cv2
import numpy as np
import base64
import io
from PIL import Image

# --- 1. Dirección de tu API ---
url = "http://127.0.0.1:8000/detect_frame"

# --- 2. Iniciar cámara ---
cap = cv2.VideoCapture(0)

print("🎥 Presiona ESC para detener la transmisión...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo capturar frame.")
        break

    # --- 3. Codificar imagen y enviarla como archivo a la API ---
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}

    try:
        response = requests.post(url, files=files, timeout=5)
    except requests.exceptions.RequestException as e:
        print("❌ Error al conectar con la API:", e)
        break

    if response.status_code == 200:
        data = response.json()
        fuego_detectado = data["fuego_detectado"]
        porcentaje = data["porcentaje_confianza"]

        # Decodificar imagen en base64
        img_bytes = base64.b64decode(data["imagen"])
        nparr = np.frombuffer(img_bytes, np.uint8)
        annotated_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Mostrar detección en consola y ventana
        print(f"🔥 Fuego detectado: {fuego_detectado} | Confianza: {porcentaje:.2f}%")
        cv2.imshow("Detección remota", annotated_frame)
    else:
        print("❌ Error al procesar imagen:", response.status_code)
        break

    # --- 4. Salir si se presiona ESC ---
    if cv2.waitKey(1) == 27:
        print("⏹️ Transmisión detenida por el usuario.")
        break

# --- 5. Liberar recursos ---
cap.release()
cv2.destroyAllWindows()
