import os

# Lista de rutas de archivos concretos a revisar
archivos_csv = [
    "/root/Age-At-Death-Estimation-from-3D-Models-of-the-Pubic-Symphysis-A-Deep-Multi-View-Learning-Approach/comparativa_visual_modelos_3D/th_01/base_jaccard.csv",
    "/root/Age-At-Death-Estimation-from-3D-Models-of-the-Pubic-Symphysis-A-Deep-Multi-View-Learning-Approach/comparativa_visual_modelos_3D/th_02/base_jaccard.csv",
    "/root/Age-At-Death-Estimation-from-3D-Models-of-the-Pubic-Symphysis-A-Deep-Multi-View-Learning-Approach/comparativa_visual_modelos_3D/th_03/base_jaccard.csv",
    "/root/Age-At-Death-Estimation-from-3D-Models-of-the-Pubic-Symphysis-A-Deep-Multi-View-Learning-Approach/comparativa_visual_modelos_3D/th_04/base_jaccard.csv",
    "/root/Age-At-Death-Estimation-from-3D-Models-of-the-Pubic-Symphysis-A-Deep-Multi-View-Learning-Approach/comparativa_visual_modelos_3D/th_05/base_jaccard.csv",
    "/root/Age-At-Death-Estimation-from-3D-Models-of-the-Pubic-Symphysis-A-Deep-Multi-View-Learning-Approach/comparativa_visual_modelos_3D/th_06/base_jaccard.csv",
    "/root/Age-At-Death-Estimation-from-3D-Models-of-the-Pubic-Symphysis-A-Deep-Multi-View-Learning-Approach/comparativa_visual_modelos_3D/th_07/base_jaccard.csv",
    "/root/Age-At-Death-Estimation-from-3D-Models-of-the-Pubic-Symphysis-A-Deep-Multi-View-Learning-Approach/comparativa_visual_modelos_3D/th_08/base_jaccard.csv",
    "/root/Age-At-Death-Estimation-from-3D-Models-of-the-Pubic-Symphysis-A-Deep-Multi-View-Learning-Approach/comparativa_visual_modelos_3D/th_09/base_jaccard.csv"
]

# Patrones que indican vistas individuales
vistas_indeseadas = ["_X", "_Y", "_Z"]

# Procesar uno a uno
for archivo in archivos_csv:
    nombre = os.path.basename(archivo)
    if any(vista in nombre for vista in vistas_indeseadas):
        os.remove(archivo)
        print(f"Eliminado: {nombre}")
    else:
        print(f"Conservado: {nombre}")

print("Proceso de limpieza finalizado.")
