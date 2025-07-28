import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.layers import Average
from tensorflow.keras.models import Model
import sys
import os
import functions as fn
from tensorflow.keras.preprocessing.image import ImageDataGenerator

### XAI section ###
import innvestigate
import pickle as pkl
import numpy as np
import time
import tensorflow.keras.backend as K
import gc
import argparse

# CRITICAL: Clear any existing sessions before starting
K.clear_session()
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution() # for use iNNvestigate

# Argumentos del script
parser = argparse.ArgumentParser(description="Generador de mapas XAI por lotes para 2 vistas")
parser.add_argument('--lote', type=int, required=True, choices=[1, 2, 3, 4, 5],
                    help='Número de lote a procesar (1-5)')
parser.add_argument('--projection1', type=str, required=True, choices=['X', 'Y', 'Z'],
                    help='Primera proyección a usar')
parser.add_argument('--projection2', type=str, required=True, choices=['X', 'Y', 'Z'],
                    help='Segunda proyección a usar')
args = parser.parse_args()

COLORMODE = "rgb"
RESNET = True

IMG_SHAPE = (36,108)
IMG_SHAPE2 = (108, 36)
depth = 3

model_weights_base = "/mnt/homeGPU/jarodriguez/codigo/Age-At-Death-Estimation-from-3D-Models-of-the-Pubic-Symphysis-A-Deep-Multi-View-Learning-Approach/separate_planes_ensembles"

projections_combo = f"{args.projection1}{args.projection2}"
processing_label = f"ensemble {projections_combo}"
print(f"Procesando {processing_label}")

ensemble_folder = f"/ensembles_{projections_combo}/221512/model_rgb_resnet_ft_{projections_combo}/model_rgb_resnet_ft_{projections_combo}"

if RESNET:
    cnn = fn.create_CNN_Resnet
    model_weights = f"{model_weights_base}{ensemble_folder}"
else:
    cnn = fn.create_CNN
    model_weights = "/content/drive/MyDrive/XAI_forense/training_models/221829/model_rgb_panoramacnn_2views/model_rgb_panoramacnn_2views"

print(f"Model weights: {model_weights}")
print(f"Using projections: {args.projection1} and {args.projection2}")

### Verificar que el archivo de pesos existe ###
if not os.path.exists(model_weights + ".index") and not os.path.exists(model_weights + ".h5"):
    print(f"[ERROR] Model weights file not found: {model_weights}")
    sys.exit(1)

### Create all models in a single graph ###
print("[INFO] Creating models in single TensorFlow graph...")

try:
    # Create ensemble model first
    cnn_0 = cnn(0, IMG_SHAPE[0], IMG_SHAPE[1], depth)  # width=108, height=36
    cnn_1 = cnn(1, IMG_SHAPE[0], IMG_SHAPE[1], depth)

    full_y = Average()([cnn_0.output, cnn_1.output])
    ensemble_model = Model(inputs=[cnn_0.input, cnn_1.input], outputs=full_y)
    ensemble_model.compile(loss=MeanAbsoluteError(), optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), metrics=['mae'])

    ensemble_model.load_weights(model_weights)
    print("...loaded ensemble weights")
    
    # Create individual models from the same graph components
    # Model for first view (using cnn_0 architecture)
    model_view_1 = Model(inputs=cnn_0.input, outputs=cnn_0.output)
    model_view_1.compile(loss=MeanAbsoluteError(), 
                        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), 
                        metrics=['mae'])
    
    # Model for second view (using cnn_1 architecture)  
    model_view_2 = Model(inputs=cnn_1.input, outputs=cnn_1.output)
    model_view_2.compile(loss=MeanAbsoluteError(), 
                        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), 
                        metrics=['mae'])
    
    print(f"[DEBUG] Modelo view1 espera: {model_view_1.input_shape}")
    print(f"[DEBUG] Modelo view2 espera: {model_view_2.input_shape}")
    print("[INFO] All models created successfully in same graph")

except Exception as e:
    print(f"[ERROR] Failed to create models: {e}")
    sys.exit(1)

### Improved XAI function with better error handling ###
def ig_xai_safe(analyzer, data_generator, outfile, max_retries=3):
    """
    Función mejorada para análisis XAI con manejo de errores y gestión de grafos
    """
    print(f"[XAI] Preparando entrada para: {outfile}", flush=True)
    
    for attempt in range(max_retries):
        try:
            # Get data from generator
            data = next(data_generator)
            
            # Handle tuple format (images, labels)
            if isinstance(data, tuple):
                images = data[0]
            else:
                images = data
            
            # Ensure numpy array format
            if not isinstance(images, np.ndarray):
                images = np.array(images)
            
            # Verify shape
            expected_shape = (1, 108, 36, depth)
            if images.shape != expected_shape:
                raise ValueError(f"[ERROR] Imagen con shape {images.shape}, pero se esperaba {expected_shape}")
            
            print(f"[XAI] Iniciando análisis de: {outfile} (intento {attempt + 1})", flush=True)
            start = time.time()
            
            # Perform XAI analysis
            result = analyzer.analyze(images)
            
            duration = time.time() - start
            print(f"[XAI] Finalizado en {duration:.2f} segundos: {outfile}", flush=True)
            
            # Save result
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            with open(outfile, 'wb') as f:
                pkl.dump(result, f)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Intento {attempt + 1} falló para {outfile}: {e}", flush=True)
            if attempt < max_retries - 1:
                print(f"[INFO] Reintentando en 2 segundos...", flush=True)
                time.sleep(2)
                # Don't clear session here as it would break the analyzers
                gc.collect()
            else:
                print(f"[ERROR] Todos los intentos fallaron para {outfile}", flush=True)
                return False

### Create analyzers once for efficiency ###
print("[INFO] Creating XAI analyzers...")
try:
    mth = 'integrated_gradients'
    analyzer_view1 = innvestigate.create_analyzer(mth, model_view_1)
    analyzer_view2 = innvestigate.create_analyzer(mth, model_view_2)
    print("[INFO] XAI analyzers created successfully")
except Exception as e:
    print(f"[ERROR] Failed to create analyzers: {e}")
    sys.exit(1)

# Datos de individuos
individuos_id = [14,15,48,64,141,271,367,504,225,558]
individuos_name = [("14_Dch",33),("15_Izq",23),("48_Dch",30),("64_Dch",52),
                  ("141_Izq",20),("271_Izq",47),("367_Izq",18),("504_Izq",35),("225_Dch",41),("558_Izq",26)]

# individuos_id = [225, 558]
# individuos_name = [("225_Dch",41),("558_Izq",26)]


# División en lotes: 4 + 4 + 2
lotes = [
    individuos_name[0:4],
    individuos_name[4:8],
    individuos_name[8:10]
]

individuos_name_lote = lotes[args.lote - 1]
print(f"Procesando lote {args.lote}: {[i[0] for i in individuos_name_lote]}")

model_name = f"ensemble_{projections_combo}"
folder = "/mnt/homeGPU/jarodriguez/panorama_extended/pubis_data_proc_25_panorama_norm"

# Mapeo de proyecciones
projection_suffixes = {
    'X': '_panorama_ext_X.png',
    'Y': '_panorama_ext_Y.png', 
    'Z': '_panorama_ext_Z.png'
}

suffix_1 = projection_suffixes[args.projection1]
suffix_2 = projection_suffixes[args.projection2]

# Verificar que las rutas de imágenes existen
if not os.path.exists(folder):
    print(f"[ERROR] Folder de imágenes no encontrado: {folder}")
    sys.exit(1)

### Procesamiento principal ###
for indv in range(len(individuos_name_lote)):
    name = individuos_name_lote[indv][0]
    age = individuos_name_lote[indv][1]
    ruta = f"inferencia_img/{model_name}/2d_heatmaps/{name}/"
    
    print(f"\n[INFO] Procesando individuo: {name} (edad: {age})")

    # Verificar que las imágenes existen
    img_path_1 = os.path.join(folder, name, name + "_0" + suffix_1)
    img_path_2 = os.path.join(folder, name, name + "_0" + suffix_2)
    
    if not os.path.exists(img_path_1):
        print(f"[ERROR] Imagen no encontrada: {img_path_1}")
        continue
    if not os.path.exists(img_path_2):
        print(f"[ERROR] Imagen no encontrada: {img_path_2}")
        continue

    # Crear directorio de salida
    os.makedirs(ruta, exist_ok=True)

    # Preparar datos
    test_view_1 = [[name + "/" + name + "_0" + suffix_1, age]]
    test_view_2 = [[name + "/" + name + "_0" + suffix_2, age]]

    success_view1 = False
    success_view2 = False
    
    # === Primera vista ===
    try:
        datagen = ImageDataGenerator()
        datagen_view1_for_pred = fn.image_generator2(test_view_1, folder, 1, datagen, IMG_SHAPE2, 
                                                   colormode=COLORMODE, shuffle=False, weights=False)
        
        est_view1 = model_view_1.predict(datagen_view1_for_pred, verbose=0, steps=1)
        print(f"Age estimation by {args.projection1}: {est_view1[0][0]:.2f}")

        output_file = os.path.join(ruta, name + f'_{args.projection1}.pkl')
        if os.path.exists(output_file):
            print(f"[INFO] Ya existe {output_file}, se omite.")
            success_view1 = True
        else:
            # Recreate generator for XAI
            datagen_view1_for_xai = fn.image_generator2(test_view_1, folder, 1, ImageDataGenerator(), 
                                                       IMG_SHAPE2, colormode=COLORMODE, shuffle=False, weights=False)
            
            success_view1 = ig_xai_safe(analyzer_view1, datagen_view1_for_xai, output_file)
            
    except Exception as e:
        print(f"[ERROR] Error procesando vista 1 para {name}: {e}")

    # === Segunda vista ===
    try:
        datagen = ImageDataGenerator()
        datagen_view2_for_pred = fn.image_generator2(test_view_2, folder, 1, datagen, IMG_SHAPE2, 
                                                   colormode=COLORMODE, shuffle=False, weights=False)
        
        est_view2 = model_view_2.predict(datagen_view2_for_pred, verbose=0, steps=1)
        print(f"Age estimation by {args.projection2}: {est_view2[0][0]:.2f}")

        output_file = os.path.join(ruta, name + f'_{args.projection2}.pkl')
        if os.path.exists(output_file):
            print(f"[INFO] Ya existe {output_file}, se omite.")
            success_view2 = True
        else:
            # Recreate generator for XAI
            datagen_view2_for_xai = fn.image_generator2(test_view_2, folder, 1, ImageDataGenerator(), 
                                                       IMG_SHAPE2, colormode=COLORMODE, shuffle=False, weights=False)
            
            success_view2 = ig_xai_safe(analyzer_view2, datagen_view2_for_xai, output_file)
            
    except Exception as e:
        print(f"[ERROR] Error procesando vista 2 para {name}: {e}")

    # === Predicción del ensemble ===
    if success_view1 and success_view2:
        try:
            # Create separate generators for each view
            test_view_1_ens = [[name + "/" + name + "_0" + suffix_1, age]]
            test_view_2_ens = [[name + "/" + name + "_0" + suffix_2, age]]
            
            # Generate data for each view separately
            datagen_view1_ens = fn.image_generator2(test_view_1_ens, folder, 1, ImageDataGenerator(), 
                                                   IMG_SHAPE2, colormode=COLORMODE, shuffle=False, weights=False)
            datagen_view2_ens = fn.image_generator2(test_view_2_ens, folder, 1, ImageDataGenerator(), 
                                                   IMG_SHAPE2, colormode=COLORMODE, shuffle=False, weights=False)
            
            # Get data from each generator
            view1_data = next(datagen_view1_ens)
            view2_data = next(datagen_view2_ens)
            
            # Handle the case where generators return (X, y) tuples
            if isinstance(view1_data, tuple):
                view1_data = view1_data[0]
            if isinstance(view2_data, tuple):
                view2_data = view2_data[0]
            
            # Predict with ensemble model
            est_ensemble = ensemble_model.predict([view1_data, view2_data], verbose=0)
            
            print(f"Age estimation by ensemble ({args.projection1}+{args.projection2}): {est_ensemble[0][0]:.2f}")
            
            # Guardar predicciones
            predictions_file = os.path.join(ruta, f"{name}_predictions.txt")
            with open(predictions_file, 'w') as f:
                f.write(f"True age: {age}\n")
                f.write(f"{args.projection1} prediction: {est_view1[0][0]:.2f}\n")
                f.write(f"{args.projection2} prediction: {est_view2[0][0]:.2f}\n")
                f.write(f"Ensemble prediction: {est_ensemble[0][0]:.2f}\n")
                f.write(f"MAE {args.projection1}: {abs(age - est_view1[0][0]):.2f}\n")
                f.write(f"MAE {args.projection2}: {abs(age - est_view2[0][0]):.2f}\n")
                f.write(f"MAE Ensemble: {abs(age - est_ensemble[0][0]):.2f}\n")
            
            print(f"[OK] Predicciones guardadas en: {predictions_file}")
                
        except Exception as e:
            print(f"[WARNING] No se pudo calcular predicción del ensemble: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[WARNING] Saltando ensemble para {name} - faltan análisis XAI individuales")

### Clean up analyzers ###
del analyzer_view1, analyzer_view2
gc.collect()

### Postprocesado ###
print(f"\n--- Postprocesado: generación de mapas 2D normalizados para {args.projection1} y {args.projection2} ---\n")

def normalized(nparray):
    """Normalizar array a rango [0,1]"""
    max_val = np.max(np.abs(nparray))
    if max_val > 0:
        nparray = nparray / max_val  # normaliza [-1, 1]
        nparray[nparray < 0] = 0     # [0,1]
    return nparray

selected_axes = [args.projection1, args.projection2]
processed_files = 0

for name, _ in individuos_name_lote:
    for axis in selected_axes:
        indv_name = f"{name}_{axis}"
        pkl_path = f"inferencia_img/{model_name}/2d_heatmaps/{name}/{indv_name}.pkl"
        pkl_out = f"inferencia_img/{model_name}/2d_heatmaps/{name}/{indv_name}_2channels.pkl"

        if not os.path.exists(pkl_path):
            print(f"[AVISO] Archivo no encontrado, se omite: {pkl_path}")
            continue

        try:
            with open(pkl_path, 'rb') as f:
                hmap_3_channels = pkl.load(f)

            # Verificar shape
            if len(hmap_3_channels.shape) == 4:  # (1, H, W, 3)
                hmap_3_channels = hmap_3_channels[0]  # (H, W, 3)
            
            # Convertir de 3 canales a 2D
            if hmap_3_channels.shape[-1] == 3:  # RGB
                hmap_2_channels = hmap_3_channels.sum(axis=-1)  # Sumar canales RGB
            else:
                hmap_2_channels = hmap_3_channels
            
            hmap_2_channels = normalized(hmap_2_channels)
            
            # Asegurar dimensiones correctas (36, 108)
            if hmap_2_channels.shape == (108, 36):
                hmap_2_channels = np.transpose(hmap_2_channels)
            
            with open(pkl_out, 'wb') as f:
                pkl.dump(hmap_2_channels, f)

            print(f"[OK] Guardado: {pkl_out} - Shape: {hmap_2_channels.shape}")
            processed_files += 1
            
        except Exception as e:
            print(f"[ERROR] Error procesando {pkl_path}: {e}")

print(f"\n[INFO] Archivos procesados: {processed_files}")

### Colección final ###
print("\n--- Colección de mapas de calor 2D ---\n")

hmaps_2_channels = []
hmap_info = []

for name, age in individuos_name_lote:
    for axis in selected_axes:
        pkl_file = f"inferencia_img/{model_name}/2d_heatmaps/{name}/{name}_{axis}_2channels.pkl"
        if not os.path.exists(pkl_file):
            print(f"[AVISO] Fichero no encontrado: {pkl_file}")
            continue
            
        try:
            with open(pkl_file, 'rb') as f:
                hmap_2_channels = pkl.load(f)
                hmaps_2_channels.append(hmap_2_channels)
                hmap_info.append({
                    'name': name,
                    'age': age,
                    'axis': axis,
                    'file': pkl_file,
                    'shape': hmap_2_channels.shape
                })
                print(f"[OK] Cargado: {pkl_file} - Shape: {hmap_2_channels.shape}")
        except Exception as e:
            print(f"[ERROR] Error cargando {pkl_file}: {e}")

print(f"\nTotal de mapas de calor cargados: {len(hmaps_2_channels)}")
print(f"Proyecciones procesadas: {selected_axes}")

# Guardar colección completa
if hmaps_2_channels:
    collection_file = f"inferencia_img/{model_name}/2d_heatmaps/heatmaps_collection_lote_{args.lote}_{args.projection1}_{args.projection2}.pkl"
    os.makedirs(os.path.dirname(collection_file), exist_ok=True)
    
    collection_data = {
        'heatmaps': hmaps_2_channels,
        'info': hmap_info,
        'projections': selected_axes,
        'lote': args.lote,
        'total_files': len(hmaps_2_channels)
    }
    
    try:
        with open(collection_file, 'wb') as f:
            pkl.dump(collection_data, f)
        print(f"[OK] Colección guardada en: {collection_file}")
    except Exception as e:
        print(f"[ERROR] Error guardando colección: {e}")

print(f"\n=== Procesamiento completado para lote {args.lote} con proyecciones {args.projection1} y {args.projection2} ===")
print(f"Archivos XAI generados: {processed_files}")
print(f"Mapas de calor en colección: {len(hmaps_2_channels)}")

print("\n=== Verificación de formas de entrada de los modelos ===")
print(f"model_view_1 input shape: {model_view_1.input_shape}")
print(f"model_view_2 input shape: {model_view_2.input_shape}")
print(f"ensemble model input shapes: {[input_tensor.shape for input_tensor in ensemble_model.input]}")