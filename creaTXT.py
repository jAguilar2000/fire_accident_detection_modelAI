import os

# Definir los directorios
image_dir = './data/images/val'
output_dir = './data/labels/val'

# Crear el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Recorrer los archivos en el directorio de imágenes
for filename in os.listdir(image_dir):
    print("entro")
    if filename.lower().endswith('.png'):
        # Obtener el nombre sin extensión
        name_without_ext = os.path.splitext(filename)[0]
        
        # Definir la ruta del archivo .txt que se va a crear
        txt_path = os.path.join(output_dir, f"{name_without_ext}.txt")
        print(txt_path)
        # Crear el archivo .txt vacío o con contenido opcional
        with open(txt_path, 'w') as f:
            pass  # Puedes escribir algo dentro si lo deseas

print("Archivos .txt creados para cada imagen .jpg.")