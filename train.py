from ultralytics import YOLO
import cv2
import os

# --- 1. Entrenamiento ---
def train_model():
    model = YOLO("yolov8m.pt")  # Modelo base (puedes usar "yolov8s.pt" para más precisión)
    results = model.train(
        data="./data.yaml",
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
        results = model(frame, conf=0.8)  # conf = umbral de confianza
        annotated_frame = results[0].plot()  # Dibuja las detecciones
        
        cv2.imshow("Detección de Incendios/Intrusos", annotated_frame)
        if cv2.waitKey(1) == 27:  # Salir con ESC
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_from_image(image_path):
    model = YOLO("runs/detect/fire_accident_detection/weights/best.pt")
    image = cv2.imread(image_path)
    
    # Detección
    results = model(image, conf=0.8)  # conf = umbral de confianza
    annotated_image = results[0].plot()  # Dibuja las detecciones
    print(f"Detecciones: {len(results[0].boxes)} objetos detectados"
          f" en {image_path}")
    cv2.imshow("Deteccion de Incendios/Intrusos", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def type_object_cv2():
    cap = cv2.VideoCapture(0)  # Usar cámara web (o cambiar a una IP)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        print(type(frame))
        cv2.imshow("Detección de Incendios/Intrusos", frame)
        if cv2.waitKey(1) == 27:  # Salir con ESC
            break

if __name__ == "__main__":
    #train_model()     # Paso 1: Entrenar
    #validate_model()  # Paso 2: Validar
    #detect_live()     # Paso 3: Probar en vivo
    #detect_from_image("rio_portada.jpg")  # Probar con una imagen específica
    type_object_cv2()
