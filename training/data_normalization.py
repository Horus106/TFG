import pandas as pd
import os
import argparse
import logging


import functions as fn
IMG_SHAPE_RGB = (36,108)
IMG_SHAPE_GRAY = (108,108)

def main(train_path, validation_path, test_path, data_path, output_path):
    print('Estandarizando train...')
    train = pd.read_csv(train_path)
    train_list_rgb = fn.get_images_list(train,'rgb')
    # train_list_gray = fn.get_images_list(train,'grayscale')

    norm_params_rgb = fn.load_image_normalize_rgb(train_list_rgb, data_path, IMG_SHAPE_RGB, output_path)
    # norm_params_gray = fn.load_image_normalize_grayscale(train_list_gray, data_path, IMG_SHAPE_GRAY, output_path)

    print('Estandarizando validacion...')
    validation = pd.read_csv(validation_path)
    validation_list_rgb = fn.get_images_list(validation,'rgb', augment=True, weights=False)
    # validation_list_gray = fn.get_images_list(validation,'grayscale', augment=False, weights=False)

    fn.load_image_normalize_rgb(validation_list_rgb, data_path, IMG_SHAPE_RGB, output_path, fit_data=False, list_values=norm_params_rgb)
    # fn.load_image_normalize_grayscale(validation_list_gray, data_path, IMG_SHAPE_GRAY, output_path, fit_data=False, list_values=norm_params_rgb)

    print('Estandarizando test...')
    test = pd.read_csv(test_path)
    test_list_rgb = fn.get_images_list(test,'rgb', augment=True, weights=False)
    # test_list_gray = fn.get_images_list(test,'grayscale', augment=False, weights=False)

    fn.load_image_normalize_rgb(test_list_rgb, data_path, IMG_SHAPE_RGB, output_path, fit_data=False, list_values=norm_params_rgb)
    # fn.load_image_normalize_grayscale(test_list_gray, data_path, IMG_SHAPE_GRAY, output_path, fit_data=False, list_values=norm_params_rgb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "--val",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "--test",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "--data",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "--output",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "-v", 
        "--verbose", 
        type=int, 
        required=False, 
        default=0
    )

    args = parser.parse_args()

    log_level = logging.WARNING
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose == 2:
        log_level = logging.DEBUG
    else:
        logging.warning('Log level not recognised. Using WARNING as default')

    logging.getLogger().setLevel(log_level)

    logging.warning("Verbose level set to {}".format(logging.root.level))

    os.makedirs(args.output, exist_ok=True)

    main(args.train, args.val, args.test, args.data, args.output)
