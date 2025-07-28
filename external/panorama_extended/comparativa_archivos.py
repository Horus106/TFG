import os
import re

def obtener_numeros_pe_de_archivos(carpeta):
    patron = re.compile(r'pe(\d+)_')
    numeros = set()

    for nombre in os.listdir(carpeta):
        ruta = os.path.join(carpeta, nombre)
        if os.path.isfile(ruta):
            coincidencia = patron.search(nombre)
            if coincidencia:
                numeros.add(int(coincidencia.group(1)))
    
    return numeros

def obtener_numeros_de_subcarpetas(carpeta):
    subcarpetas = [
        nombre for nombre in os.listdir(carpeta)
        if os.path.isdir(os.path.join(carpeta, nombre)) and nombre.isdigit()
    ]
    return set(map(int, subcarpetas))

def comparar_pe_y_subcarpetas(carpeta1, carpeta2):
    numeros_pe = obtener_numeros_pe_de_archivos(carpeta1)
    numeros_carpetas = obtener_numeros_de_subcarpetas(carpeta2)

    discrepancias = numeros_carpetas - numeros_pe

    print(f"Números únicos en nombres 'pe' de archivos en carpeta1: {sorted(numeros_pe)}")
    print(f"Números únicos en nombres de subcarpetas en carpeta2: {sorted(numeros_carpetas)}")
    print(f"Discrepancias (en carpeta2 pero no en carpeta1): {sorted(discrepancias)}")
    print(f"Total discrepancias: {len(discrepancias)}")

    return discrepancias

# Cambia estas rutas por las reales
carpeta1 = "Age-At-Death-Estimation-from-3D-Models-of-the-Pubic-Symphysis-A-Deep-Multi-View-Learning-Approach/external/panorama_extended/out"
carpeta2 = "/media/juanan/TOSHIBA EXT/PubisObj/pubis"

comparar_pe_y_subcarpetas(carpeta1, carpeta2)
