from PIL import Image
import os

def convertir_jfif_a_jpg(base_path):
    for clase in os.listdir(base_path):
        clase_path = os.path.join(base_path, clase)
        if os.path.isdir(clase_path):
            for archivo in os.listdir(clase_path):
                if archivo.lower().endswith('.jfif'):
                    original_path = os.path.join(clase_path, archivo)
                    nuevo_path = os.path.join(clase_path, archivo.replace(".jfif", ".jpg"))
                    try:
                        imagen = Image.open(original_path).convert("RGB")
                        imagen.save(nuevo_path, "JPEG")
                        os.remove(original_path)
                        print(f"✅ Convertido: {original_path} -> {nuevo_path}")
                    except Exception as e:
                        print(f"❌ Error con {original_path}: {e}")

# Ejecutar para train y valid
convertir_jfif_a_jpg("dataset/train")
convertir_jfif_a_jpg("dataset/valid")