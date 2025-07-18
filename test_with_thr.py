import cv2
import httpx
import queue
from concurrent.futures import ThreadPoolExecutor

url = "http://192.168.100.205:8000/detect_frame"
MAX_QUEUE_SIZE = 10
frame_queue = queue.Queue()
executor = ThreadPoolExecutor(max_workers=4)

def get_response(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {
        'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg'),
    }
    try:
        with httpx.Client(timeout=15) as client:
            response = client.post(url, files=files)
            if response.status_code == 200:
                data = response.json()
                print("🔥 Fuego detectado:", data["fuego_detectado"])
                return data
            else:
                print("⚠️ Error en la respuesta:", response.status_code)
                return None
    except Exception as e:
        print(f"❌ No hay respuesta del servidor: {e}")
        return None

def frame_worker():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        get_response(frame)

def main():
    cam = cv2.VideoCapture(0)

    # Lanzamos algunos hilos de trabajo
    for _ in range(4):
        executor.submit(frame_worker)

    try:
        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                break

            if frame_queue.qsize() < MAX_QUEUE_SIZE:
                frame_queue.put(frame)

            cv2.imshow("Prueba de captura", frame)
            if cv2.waitKey(1) == 27:  # Tecla ESC
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        
        # Cerramos el pool de threads
        for _ in range(4):
            frame_queue.put(None)
        executor.shutdown(wait=True)

if __name__ == "__main__":
    main()

"""
import cv2
import httpx
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import base64

url = "http://192.168.100.205:8000/detect_frame"
MAX_QUEUE_SIZE = 10
frame_queue = queue.Queue()
executor = ThreadPoolExecutor(max_workers=4)

# Variables compartidas
processed_frame = None
frame_lock = threading.Lock()

def get_response(frame):
    global processed_frame
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {
        'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg'),
    }
    try:
        with httpx.Client(timeout=15) as client:
            response = client.post(url, files=files)
            if response.status_code == 200:
                data = response.json()
                img_bytes = base64.b64decode(data["imagen"])
                nparr = np.frombuffer(img_bytes, np.uint8)
                processed = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                print("🔥 Fuego detectado:", data["fuego_detectado"])

                # Guardar imagen procesada para mostrarla
                with frame_lock:
                    processed_frame = processed
                return data
            else:
                print("⚠️ Error en la respuesta:", response.status_code)
                return None
    except Exception as e:
        print(f"❌ No hay respuesta del servidor: {e}")
        return None

def frame_worker():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        get_response(frame)

def main():
    global processed_frame

    cam = cv2.VideoCapture(0)

    for _ in range(4):
        executor.submit(frame_worker)

    try:
        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                break

            # Enviar a cola si hay espacio
            if frame_queue.qsize() < MAX_QUEUE_SIZE:
                frame_queue.put(frame)

            # Mostrar la imagen procesada si está disponible
            with frame_lock:
                display_frame = processed_frame if processed_frame is not None else frame.copy()

            cv2.imshow("Procesado / Original", display_frame)

            if cv2.waitKey(1) == 27:
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

        for _ in range(4):
            frame_queue.put(None)
        executor.shutdown(wait=True)

if __name__ == "__main__":
    main()
"""