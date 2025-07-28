import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import random
import os
import argparse
import logging
import json
import functions as fn
from tensorflow.keras.layers import Average


def main(test_path, data_path, color_mode, img_shape, depth, cnn, verbose_level, model_folder, model_name, output, projection_1, projection_2):
    trained_models = [
        item for item in os.listdir(model_folder)
        if os.path.isdir(os.path.join(model_folder, item)) and item.lower() != "checkpoint"
    ]

    for model_code in trained_models:
        pretrained_model = os.path.join(model_folder, model_code, model_code)

        # Mapeo de proyecciones a índices
        projection_map = {'X': 0, 'Y': 1, 'Z': 2}
        proj1_idx = projection_map[projection_1]
        proj2_idx = projection_map[projection_2]

        # Crear modelo multivista con solo 2 proyecciones
        cnn_proj1 = cnn(proj1_idx, img_shape[0], img_shape[1], depth)
        cnn_proj2 = cnn(proj2_idx, img_shape[0], img_shape[1], depth)
        x = Average()([cnn_proj1.output, cnn_proj2.output])
        model = Model(inputs=[cnn_proj1.input, cnn_proj2.input], outputs=x)

        model.compile(loss=MeanAbsoluteError(), optimizer=Adam(), metrics=['mae'])

        print(f"[INFO] Cargando pesos del modelo: {pretrained_model}")
        model.load_weights(pretrained_model)
        print("[DEBUG] Sumatoria de pesos cargados:", np.sum([np.sum(w.numpy()) for w in model.weights]))

        # Preparar datos de test multivista con 2 proyecciones
        test_df = pd.read_csv(test_path)
        test_not_augment = fn.get_images_list(test_df, color_mode, projection_1, projection_2, augment=False, weights=False)
        test_augment = fn.get_images_list(test_df, color_mode, projection_1, projection_2, augment=True, weights=False)

        datagen = ImageDataGenerator()
        
        # Usar la función corregida image_generator_fixed
        datagen_test_not_augment = fn.image_generator_fixed(
            test_not_augment, data_path, 1, datagen, img_shape, colormode=color_mode,
            shuffle=False, weights=False
        )
        datagen_test_augment = fn.image_generator_fixed(
            test_augment, data_path, 1, datagen, img_shape, colormode=color_mode,
            shuffle=False, weights=False
        )

        # Predicción
        prediction = model.predict(datagen_test_not_augment, verbose=verbose_level, steps=len(test_not_augment))

        # === Predicción con datos aumentados ===
        prediction_aug = model.predict(datagen_test_augment, verbose=verbose_level, steps=len(test_augment))
        true_aug = np.array([float(n[2]) for n in test_augment])
        pred_aug = prediction_aug.flatten()
        names_aug = np.array([n[0].split("/")[0].split("_") for n in test_augment]).T

        prediction_dict_aug = {
            "SAMPLE": names_aug[0],
            "SIDE": names_aug[1],
            "TRUE": true_aug,
            "PREDICTED": pred_aug,
            "ERROR": true_aug - pred_aug
        }

        df_aug = pd.DataFrame.from_dict(prediction_dict_aug)

        output_file_aug = os.path.join(output, f"age_prediction_{model_code}_{model_name}_{proj_suffix}_augment.csv")
        print(f"[INFO] Guardando predicciones con augment en: {output_file_aug}")
        df_aug.to_csv(output_file_aug, index=False)


        x_batch, _ = next(datagen_test_not_augment)
        print("[DEBUG] Tipo de entrada:", type(x_batch))
        print("[DEBUG] Tamaño batch entrada:", len(x_batch))
        print("[DEBUG] Shape vista 0:", x_batch[0].shape)
        if len(x_batch) > 1:
            print("[DEBUG] Shape vista 1:", x_batch[1].shape)

        # Ajustar índice para obtener las etiquetas verdaderas (ahora en posición 2)
        true = np.array([float(n[2]) for n in test_not_augment])  # Cambiado de n[len(n)-1] a n[2]
        pred = prediction.flatten()

        # Ajustar parsing de nombres para el nuevo formato
        names = np.array([n[0].split("/")[0].split("_") for n in test_not_augment]).T
        prediction_dict = {
            "SAMPLE": names[0],
            "SIDE": names[1],
            "TRUE": true,
            "PREDICTED": pred,
            "ERROR": true - pred
        }

        dataframe = pd.DataFrame.from_dict(prediction_dict)

        # Incluir información de proyecciones en el nombre del archivo
        proj_suffix = f"{projection_1}{projection_2}"
        output_file = os.path.join(output, f"age_prediction_{model_code}_{model_name}_{proj_suffix}.csv")
        print(f"[INFO] Guardando predicciones en: {output_file}")

        os.makedirs(output, exist_ok=True)
        dataframe.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parameters", help="JSON file with parameters configuration", type=str, required=True)
    args = parser.parse_args()

    with open(args.parameters, "r") as json_file:
        args = json.load(json_file)

    # Validar que las proyecciones estén especificadas
    if "projection_1" not in args or "projection_2" not in args:
        raise ValueError("projection_1 and projection_2 must be specified in the JSON parameters")
    
    # Validar que las proyecciones sean válidas
    valid_projections = ['X', 'Y', 'Z']
    if args["projection_1"] not in valid_projections or args["projection_2"] not in valid_projections:
        raise ValueError("Projections must be one of: X, Y, Z")
    
    # Validar que las proyecciones sean diferentes
    if args["projection_1"] == args["projection_2"]:
        raise ValueError("projection_1 and projection_2 must be different")

    if args["color_mode"] == 'rgb':
        img_shape = (36, 108)
        depth = 3
        cnn = fn.create_CNN_Resnet if args["resnet"] else fn.create_CNN
    elif args["color_mode"] == 'grayscale':
        img_shape = (108, 108)
        depth = 1
        cnn = fn.create_CNN
    else:
        raise RuntimeError("Color mode option unknown")

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    tf.random.set_seed(args["seed"])

    log_level = logging.WARNING
    if args["verbose"] == 1: log_level = logging.INFO
    elif args["verbose"] == 2: log_level = logging.DEBUG

    logging.getLogger().setLevel(log_level)

    main(
        args["test"],
        args["data"],
        args["color_mode"],
        img_shape,
        depth,
        cnn,
        args["verbose_keras"],
        args["model_folder"],
        args["model_name"],
        args["output"],
        args["projection_1"],
        args["projection_2"]
    )