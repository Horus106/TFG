import argparse
import logging
import os
import cv2
import numpy as np

def rgb_to_gray(img):
    img_gray = cv2.cvtColor(img, cv2.cv2.COLOR_BGR2GRAY)
    return img_gray

def mask_image(img):
    if len(img.shape) != 2:
        img = cv2.cvtColor(img, cv2.cv2.COLOR_BGR2GRAY)
    img_mask = np.where(img > 0, 255.0, 0.0)
    return img_mask

def shrink_image(img, factor=0.2):
    img = cv2.resize(img, (0, 0), fx = factor, fy = factor)
    print(img.shape)
    return img

def main(input, convert, mask, shrink, output):
    for root, folders, files in os.walk(input):
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            img_mod = img.copy()
            name_mod = ""

            if convert:
                img_mod = rgb_to_gray(img_mod)
                name_mod += "_gray"

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    cv2.imshow(file, img)
                    cv2.imshow(file+"convert", img_mod)

            if mask:
                img_mod = mask_image(img_mod)
                name_mod += "_mask"

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    cv2.imshow(file, img)
                    cv2.imshow(file+"mask", img_mod)

            if shrink:
                img_mod = shrink_image(img_mod)
                name_mod += "_input"

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    cv2.imshow(file, img)
                    cv2.imshow(file+"shrink", img_mod)

            filename, ext = os.path.splitext(file)
            file_mod = filename + name_mod + ext
            file_mod = os.path.join(root, file_mod)
            file_mod = file_mod.replace(input, output)

            logging.debug(file_mod)

            os.makedirs(os.path.dirname(file_mod), exist_ok=True)
            cv2.imwrite(file_mod, img_mod)

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                cv2.imshow(file, img)
                cv2.imshow(file_mod, img_mod)

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "-c",
        "--convert",
        action="store_true",
        help="",
        required=False,
        default=False
    )

    parser.add_argument(
        "-m",
        "--mask",
        action="store_true",
        help="",
        required=False,
        default=False
    )
    
    parser.add_argument(
        "-s",
        "--shrink",
        action="store_true",
        help="",
        required=False,
        default=False
    )

    parser.add_argument(
        "-o",
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

    

    main(args.input, args.convert, args.mask, args.shrink, args.output)
