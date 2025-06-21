from ultralytics import YOLO
import cv2
import os

# --- 1. Entrenamiento ---
def train_model():
    model = YOLO("yolov8n.pt")  # Modelo base (puedes usar "yolov8s.pt" para más precisión)
    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        name="fire_accident_detection"
    )

# --- 2. Validación ---
def validate_model():
    model = YOLO("runs/detect/fire_accident_detection/weights/best.pt")  # Usa la mejor época
    metrics = model.val()
    print(metrics)

# --- 3. Prueba en tiempo real ---
def detect_live():
    model = YOLO("runs/detect/fire_accident_detection/weights/best.pt")
    cap = cv2.VideoCapture(0)  # Usar cámara web (o cambiar a una IP)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detección
        results = model(frame, conf=0.5)  # conf = umbral de confianza
        annotated_frame = results[0].plot()  # Dibuja las detecciones
        
        cv2.imshow("Detección de Incendios/Intrusos", annotated_frame)
        if cv2.waitKey(1) == 27:  # Salir con ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    train_model()     # Paso 1: Entrenar
    validate_model()  # Paso 2: Validar
    detect_live()     # Paso 3: Probar en vivo