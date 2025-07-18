import asyncio
import cv2
import httpx
import time

url="http://192.168.40.66:8000/detect_frame"
delay_request = 2 # Tiempo entre peticiones

async def get_response(client, frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {
        'file': (
            'frame.jpg', img_encoded.tobytes(),
            'image/jpeg'
        ),
    }
    try:
        response = await client.post(url, files=files, timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            print("Error en la respuesta")
            return None
    except Exception as e:
        print(f"No hay respuesta {e}")
        return None

async def main():
    cam = cv2.VideoCapture(0)
    async with httpx.AsyncClient() as client:
        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                break
            data = await get_response(client, frame)
            
            if data["fuego_detectado"]:
                print(f'{data["fuego_detectado"]} Tiempo: {time.time()} Confianza: {data["porcentaje_confianza"]}')
            cv2.imshow("Prueba de captura", frame)
            if cv2.waitKey(1) == 27:
                break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())