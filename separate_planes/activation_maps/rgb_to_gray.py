import argparse
import logging
import os
import cv2

def main(input, output):
    for root, folders, files in os.walk(input):
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            img_gray = cv2.cvtColor(img, cv2.cv2.COLOR_BGR2GRAY)

            filename, ext = os.path.splitext(file)
            file_gray = filename + "_gray" + ext
            file_gray = os.path.join(root, file_gray)
            file_gray = file_gray.replace(input, output)
            logging.debug(file_gray)
            os.makedirs(os.path.dirname(file_gray), exist_ok=True)
            cv2.imwrite(file_gray, img_gray)

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                cv2.imshow(file, img)
                cv2.imshow(file_gray, img_gray)

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
