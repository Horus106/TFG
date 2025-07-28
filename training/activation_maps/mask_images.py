import argparse
import logging
import os
import cv2
import numpy as np

def main(input, output):
    for root, folders, files in os.walk(input):
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            img = cv2.cvtColor(img, cv2.cv2.COLOR_BGR2GRAY)
            img_mask = np.where(img > 0, 255.0, 0.0)

            filename, ext = os.path.splitext(file)
            file_mask = filename + "_mask" + ext
            file_mask = os.path.join(root, file_mask)
            file_mask = file_mask.replace(input, output)
            logging.debug(file_mask)
            os.makedirs(os.path.dirname(file_mask), exist_ok=True)
            cv2.imwrite(file_mask, img_mask)

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                cv2.imshow(file, img)
                cv2.imshow("file_mask", img_mask)

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

    

    main(args.input, args.output)
