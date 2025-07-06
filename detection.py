from ultralytics import YOLO

model = YOLO("runs/detect/fire_accident_detection/weights/best.pt")

def detect_frame_and_return(frame):
    results = model(frame, conf=0.8)
    result = results[0]

    fuego_detectado = False
    porcentaje_fuego = 0.0

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]

        if class_name.lower() == "fire":
            fuego_detectado = True
            porcentaje_fuego = max(porcentaje_fuego, confidence * 100)

    annotated_frame = result.plot()

    return porcentaje_fuego, fuego_detectado, annotated_frame