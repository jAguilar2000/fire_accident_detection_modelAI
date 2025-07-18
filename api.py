from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import detection
import numpy as np
import cv2
import io
import base64
import threading
import queue
import time

app = FastAPI()

# Cola para enviar frames al hilo de procesamiento
frame_queue = queue.Queue(maxsize=1)  # Solo un frame activo a la vez
result_dict = {}

# --- Worker que procesa imágenes con YOLO ---
def worker():
    while True:
        try:
            frame_id, frame = frame_queue.get()
            porcentaje, detectado, result_img = detection.detect_frame_and_return(frame)

            _, buffer = cv2.imencode('.jpg', result_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            result_dict[frame_id] = {
                "fuego_detectado": detectado,
                "porcentaje_confianza": round(porcentaje, 2),
                "imagen": img_base64
            }

        except Exception as e:
            print("❌ Error en hilo de detección:", e)

# Iniciar hilo worker
threading.Thread(target=worker, daemon=True).start()

# --- Endpoint de detección ---
@app.post("/detect_frame")
async def detect_frame_api(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    frame_id = str(time.time())  # ID único para cada petición

    # Agregar a la cola (si está llena, descarta el anterior para evitar acumulación)
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
    frame_queue.put((frame_id, frame))

    # Esperar a que se procese
    timeout = 60  # segundos
    start_time = time.time()
    while time.time() - start_time < timeout:
        if frame_id in result_dict:
            result = result_dict.pop(frame_id)
            return JSONResponse(content=result)
        time.sleep(0.01)

    return JSONResponse(content={"error": "Procesamiento lento o fallo de detección"}, status_code=504)
