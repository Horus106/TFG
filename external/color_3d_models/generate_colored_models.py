import argparse
import logging
import os
import pandas as pd
import subprocess
import multiprocessing

def work(cmd):
    return subprocess.run(cmd, shell=False, capture_output=True)

def main(input, output):
    relevance_maps_df = pd.read_csv(input, sep=";")


    args_list_parallel = []

    exe = "bin/panorama_extended"
    
    for i, row in relevance_maps_df.iterrows():
        net = row["net"]
        sample = row["sample"]
        side = row["side"]
        axis = row["axis"]
        obj = row["obj"]
        # izq = row["izq"]
        # dch = row["dch"]
        relevance = row["relevance"]
        threshold = "0.0"
        color_output = os.path.join(output, net, "{}_{}".format(str(sample),side))
        os.makedirs(color_output, exist_ok=True)

        args = [exe, str(sample)+"_"+side, axis, obj, relevance, color_output, threshold]

        call = ""
        for arg in args:
            call += arg + " "

        print(call)
        print()

        args_list_parallel.append(args)

    print(args_list_parallel)

    pool = multiprocessing.Pool(processes=10)
    pool.map(work, args_list_parallel)

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
