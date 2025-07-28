import argparse
import logging
import os
import multiprocessing

def combine_color(old, new):
    new_colors = []
    for channel_old, channel_new in zip(old, new):
        # print("Old:", channel_old, "New:", channel_new)
        # old_color = channel_old - 0.25
        # new_color = channel_new - 0.25
        # new_color += old_color
        # new_color += 0.25
        new_color = channel_new
        new_color += channel_old
        if new_color > 1.0:
            new_color = 1.0
        new_colors.append(new_color)
        # print("New:", new_color)

    return new_colors

def process_folder(folder, name, models):
    v_colors = {}
    fixed_obj_txt = []

    for model_path in models:
        with open(model_path, "r") as model_obj:
            obj_lines = model_obj.readlines()
            # print(len(obj_lines))

        for i, line in enumerate(obj_lines):
            if line.startswith("v "):
                line = [ float(i) for i in line.split(" ")[4:] ]
                if i in v_colors.keys():
                    v_colors[i] = combine_color(v_colors[i], line)
                else:
                    v_colors[i] = line
                    
                # print(line)

    with open(models[0], "r") as model_obj:
        obj_lines = model_obj.readlines()
        # print(len(obj_lines))
        # print(os.path.basename(os.path.dirname(models[0])))
    
    for i, line in enumerate(obj_lines):
        line = " ".join(line.split(" ")[:4])
        if line.startswith("v "):
            # print(i)
            # print(line)
            # print(v_colors[i])
            line = " ".join([line, *[ str(x) for x in v_colors[i] ] ]) + "\n"
            # print(line)
        fixed_obj_txt.append(line)
    # print(v_colors)

    name = os.path.basename(os.path.dirname(models[0]))
    # print(name)
    output_obj = os.path.join(os.path.dirname(folder), name+"_colored.obj")

    with open(output_obj, "w") as file:
        for l in fixed_obj_txt:
            file.write(l)


def main(folder):
    
    args_list_parallel = []
    for root, folders, files in os.walk(folder):
        for folder in folders:
            list_obj = os.listdir(os.path.join(root,folder))
            list_obj = [ x for x in list_obj if (".obj" in x and not "full" in x) ]
            list_obj = [ os.path.join(root, folder, x) for x in list_obj ]
            args_list_parallel.append([os.path.join(root, folder), folder, list_obj])

    pool = multiprocessing.Pool(processes=10)
    pool.starmap(process_folder, args_list_parallel)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        help="",
        required=True
    )

    # parser.add_argument(
    #     "-o",
    #     "--output",
    #     type=str,
    #     help="",
    #     required=True
    # )

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

    main(args.folder)
