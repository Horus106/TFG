import os
import re

def contar_pe_en_directorio(raiz):
    patron = re.compile(r'pe(\d+)_')
    numeros_unicos = set()
    total_coincidencias = 0

    for carpeta_actual, _, archivos in os.walk(raiz):
        for archivo in archivos:
            coincidencia = patron.search(archivo)
            if coincidencia:
                total_coincidencias += 1
                numero = int(coincidencia.group(1))
                numeros_unicos.add(numero)

    print(f"Total de archivos que coinciden: {total_coincidencias}")
    print(f"Número de carpetas únicas (números después de 'pe'): {len(numeros_unicos)}")
    print(f"Números únicos encontrados: {sorted(numeros_unicos)}")

    return total_coincidencias, numeros_unicos

# Cambia esta ruta por la ruta a tu carpeta base
directorio = 'Age-At-Death-Estimation-from-3D-Models-of-the-Pubic-Symphysis-A-Deep-Multi-View-Learning-Approach/external/panorama_extended/out'
contar_pe_en_directorio(directorio)
