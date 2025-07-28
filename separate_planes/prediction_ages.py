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

def main(test_path, data_path, color_mode, img_shape, depth, cnn, verbose_level, model_folder, model_name, output, view):
    trained_models = os.listdir(model_folder)
    trained_models = [item for item in trained_models if os.path.isdir(os.path.join(model_folder, item))]

    for model_code in trained_models:
        pretrained_model = os.path.join(model_folder, model_name, model_name)

        model_cnn = cnn(0, img_shape[0], img_shape[1], depth)
        model = Model(inputs=model_cnn.input, outputs=model_cnn.output)
        model.compile(loss=MeanAbsoluteError(), optimizer=Adam(), metrics=['mae'])

        print(f"[INFO] Cargando pesos del modelo: {pretrained_model}")
        model.load_weights(pretrained_model)

        test_df = pd.read_csv(test_path)
        test_not_augment = fn.get_images_list(test_df, color_mode, view=view, augment=False, weights=False)

        datagen = ImageDataGenerator()
        datagen_test_not_augment = fn.image_generator(test_not_augment, data_path, 1, datagen, img_shape, colormode=color_mode, shuffle=False, weights=False)

        prediction = model.predict(datagen_test_not_augment, verbose=verbose_level, steps=len(test_not_augment))
        true = np.array([float(n[1]) for n in test_not_augment])
        pred = prediction.flatten()

        names = np.array([n[0].split("/")[0].split("_") for n in test_not_augment]).T

        prediction_dict = {
            "SAMPLE": names[0],
            "SIDE": names[1],
            "TRUE": true,
            "PREDICTED": pred,
            "ERROR": true - pred
        }

        dataframe = pd.DataFrame.from_dict(prediction_dict)

        output_file = os.path.join(output, f"age_prediction_{model_code}_{model_name}.csv")
        print(f"[INFO] Guardando predicciones en: {output_file}")

        os.makedirs(output, exist_ok=True)  # 👈 crea el directorio si no existe
        dataframe.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parameters", help="JSON file with parameters configuration", type=str, required=True)
    args = parser.parse_args()

    with open(args.parameters, "r") as json_file:
        args = json.load(json_file)

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
        args["view"]
    )
