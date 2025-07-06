from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import detection
import numpy as np
import cv2
import io
import base64

app = FastAPI()

@app.post("/detect_frame")
async def detect_frame_api(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    porcentaje, detectado, result_img = detection.detect_frame_and_return(frame)

    # Codificar imagen en base64 para enviarla como texto
    _, buffer = cv2.imencode('.jpg', result_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "fuego_detectado": detectado,
        "porcentaje_confianza": round(porcentaje, 2),
        "imagen": img_base64
    }