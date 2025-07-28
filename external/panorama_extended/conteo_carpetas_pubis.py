import os

def contar_subcarpetas_hijas(ruta):
    subcarpetas = [
        nombre for nombre in os.listdir(ruta)
        if os.path.isdir(os.path.join(ruta, nombre))
    ]
    print(f"Número de subcarpetas hijas: {len(subcarpetas)}")
    return subcarpetas

# Cambia esta ruta por la ruta a tu carpeta
ruta_carpeta = "/media/juanan/TOSHIBA EXT/PubisObj/pubis"
contar_subcarpetas_hijas(ruta_carpeta)
