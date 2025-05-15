import os
import shutil
import random

# Ruta a la carpeta original que contiene todas las clases
ruta_original = "aves"  # aquí están tus carpetas de especies
ruta_destino = "dataset"

# Porcentaje para validación
porcentaje_validacion = 0.2

# Crear las carpetas train/ y valid/
os.makedirs(os.path.join(ruta_destino, "train"), exist_ok=True)
os.makedirs(os.path.join(ruta_destino, "valid"), exist_ok=True)

# Recorrer cada clase
for clase in os.listdir(ruta_original):
    ruta_clase = os.path.join(ruta_original, clase)
    if os.path.isdir(ruta_clase):
        imagenes = os.listdir(ruta_clase)
        random.shuffle(imagenes)

        # Cálculo de cuántas imágenes van a validación
        num_validacion = int(len(imagenes) * porcentaje_validacion)
        imagenes_valid = imagenes[:num_validacion]
        imagenes_train = imagenes[num_validacion:]

        # Crear subcarpetas por clase
        os.makedirs(os.path.join(ruta_destino, "train", clase), exist_ok=True)
        os.makedirs(os.path.join(ruta_destino, "valid", clase), exist_ok=True)

        # Copiar imágenes
        for img in imagenes_train:
            shutil.copy2(os.path.join(ruta_clase, img), os.path.join(ruta_destino, "train", clase, img))

        for img in imagenes_valid:
            shutil.copy2(os.path.join(ruta_clase, img), os.path.join(ruta_destino, "valid", clase, img))

        print(f"✅ {clase}: {len(imagenes_train)} train, {len(imagenes_valid)} valid")

print("\n Separación completada correctamente")
