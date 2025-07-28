import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Average
import functions as fn  # Asegúrate de que functions.py esté en el path

# === Función robusta para extraer submodelos ===
def extract_submodels_from_ensemble(model_ft):
    """
    Extrae submodelos individuales del modelo ensemble entrenado (model_ft) asumiendo dos ramas.
    Devuelve dos modelos: model_x y model_y
    """
    assert isinstance(model_ft.input, list) and len(model_ft.input) == 2, "Se esperaban dos entradas en el modelo ensemble"

    # Buscar la capa 'Average'
    for layer in reversed(model_ft.layers):
        if isinstance(layer, Average):
            input_tensors = layer.input
            break
    else:
        raise RuntimeError("No se encontró una capa 'Average' al final del modelo")

    model_1 = Model(inputs=model_ft.input[0], outputs=input_tensors[0], name="model_1")
    model_2 = Model(inputs=model_ft.input[1], outputs=input_tensors[1], name="model_2")
    return model_1, model_2

# === Configuración ===
combinations = [('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]
base_path = "ensembles_{proj}/221512/model_rgb_resnet_ft_{proj}"
IMG_SHAPE = (36, 108)
DEPTH = 3

# === Procesamiento de ensembles ===
for proj1, proj2 in combinations:
    proj_key = f"{proj1}{proj2}"
    print(f"\n==================== Procesando modelo {proj_key} ====================")

    # Rutas
    model_dir = base_path.format(proj=proj_key)
    checkpoint_path = os.path.join(model_dir, f"model_rgb_resnet_ft_{proj_key}")
    output_path_1 = os.path.join(model_dir, f"model_{proj1.lower()}")
    output_path_2 = os.path.join(model_dir, f"model_{proj2.lower()}")

    # Crear modelo ensemble
    model_1 = fn.create_CNN_Resnet(0, IMG_SHAPE[1], IMG_SHAPE[0], DEPTH, trainable=True)
    model_2 = fn.create_CNN_Resnet(1, IMG_SHAPE[1], IMG_SHAPE[0], DEPTH, trainable=True)
    merged = tf.keras.layers.Average()([model_1.output, model_2.output])
    model_ft = Model(inputs=[model_1.input, model_2.input], outputs=merged)

    # Cargar pesos
    print(f"[INFO] Cargando checkpoint desde: {checkpoint_path}")
    if not os.path.exists(checkpoint_path + ".index"):
        print(f"[ERROR] Checkpoint no encontrado para {proj_key}. Se omite.")
        continue
    model_ft.load_weights(checkpoint_path).expect_partial()
    print(f"[OK] Pesos cargados correctamente para {proj_key}")

    # Extraer submodelos individuales
    model_sub1, model_sub2 = extract_submodels_from_ensemble(model_ft)

    # Guardar pesos
    model_sub1.save_weights(output_path_1)
    model_sub2.save_weights(output_path_2)

    print(f"[✓] Submodelo {proj1} guardado en: {output_path_1}")
    print(f"[✓] Submodelo {proj2} guardado en: {output_path_2}")
