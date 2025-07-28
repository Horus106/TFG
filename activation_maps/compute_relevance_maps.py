import argparse
import logging
import pickle as pkl
import cv2 as cv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

def find_obj(obj_folder, sample, side):
    for root, folders, files in os.walk(obj_folder):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext == ".obj":
                file_sample, file_side, _ = file.split("_")
                if sample == file_sample and side == file_side:
                    return os.path.join(os.path.abspath(root), file)

def heatmap_on_image(heatmap, image, w=108, h=36, hcmap="jet"):
    plt.figure(figsize=(w,h),dpi=1)
    if hcmap == "jet":
        min = 0
    else:
        min = -1
    hmax = sns.heatmap(heatmap, vmin=min, vmax=1,
                    cmap=hcmap,
                    xticklabels=False, yticklabels=False, cbar=False, 
                    alpha=1, zorder=1)
    # plt.savefig("os.path.join(relevance_maps_path, "heatmap.png"))
    hmax.imshow(image,
                cmap="gray",
                alpha=0.5,
                aspect = hmax.get_aspect(),
                extent = hmax.get_xlim() + hmax.get_ylim(),
                zorder = 2)
    plt.tight_layout()
    # plt.savefig("os.path.join(relevance_maps_path, "heatmap_background.png"))

    ax = plt.gca()
    canvas = ax.figure.canvas
    canvas.draw()
    hoi = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    hoi = np.reshape(hoi, (h,w,3))
    hoi = cv.cvtColor(hoi,cv.COLOR_BGR2RGBA)
    plt.close()
    return hoi


def main(width, maps_path, data_path, obj_paths, threshold, as_png):
    height = int(width / 3)
    relevance_maps = {}
    i = 0
    for root, folders, files in os.walk(maps_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext == ".pkl":
                file_path = os.path.join(root, file)

                sample, side, axis, _ = file.split("_")

                print(sample, side, axis, file_path)

                if as_png:
                    relevance_map = cv.imread(file_path, 0)/255
                else:
                    with open(file_path, 'rb') as f:
                        data = pkl.load(f)

                    relevance_map = np.array(data).T


                relevance_map = cv.resize(relevance_map, (width, height), cv.INTER_CUBIC)
                # relevance_map_max = np.max(relevance_map)
                # relevance_map = relevance_map / relevance_map_max
                # relevance_map = np.where(relevance_map >= threshold, relevance_map, 0)
                print(np.min(relevance_map), np.max(relevance_map))

                relevance_map_orig = relevance_map.copy()
                relevance_map_orig_max = np.max(relevance_map_orig)
                relevance_map_orig = relevance_map_orig / relevance_map_orig_max

                mask = cv.imread(os.path.join(data_path, "{}_{}".format(sample,side), "{}_{}_0_panorama_ext_{}_gray_mask.png".format(sample,side,axis)), cv.IMREAD_GRAYSCALE)
                mask = cv.resize(mask, (width, height), cv.INTER_CUBIC)

                mask = mask / 255.0
                relevance_map = relevance_map * mask
                relevance_map = relevance_map.astype('float32')

                cv.imwrite(os.path.join(root,"relevance_map_{}_{}_{}_complete.png".format(sample,side,axis)), relevance_map*255)

                # relevance_map = relevance_map/2
                relevance_map[:,:height] += relevance_map[:,2*height:]
                relevance_map = relevance_map[:,:2*height]
                relevance_map_max = np.max(relevance_map)
                relevance_map = relevance_map / relevance_map_max
                relevance_map = np.where(relevance_map >= threshold, relevance_map, 0)


                relevance_maps_path = os.path.join(os.path.abspath(root),"relevance_map_{}_{}_{}.png".format(sample,side,axis))

                panorama = cv.imread(os.path.join(data_path, "{}_{}".format(sample,side), "{}_{}_0_panorama_ext_{}_gray.png".format(sample,side,axis)), cv.IMREAD_GRAYSCALE)
                panorama = cv.resize(panorama, (width, height), cv.INTER_CUBIC)

                for cmap in ["jet", "bwr"]:

                    heatmap = heatmap_on_image(cv.resize(relevance_map_orig, (width, height), cv.INTER_CUBIC), panorama, w=width, h=height, hcmap=cmap)
                    cv.imwrite(os.path.join(root,"heatmap_{}_{}_{}_complete_{}.png".format(sample,side,axis,cmap)), heatmap)

                    panorama_no_extend = panorama[:,:2*height]
                    heatmap = heatmap_on_image(relevance_map, panorama_no_extend, w=2*height, h=height, hcmap=cmap)

                    cv.imwrite(os.path.join(root,"heatmap_{}_{}_{}_{}.png".format(sample,side,axis, cmap)), heatmap)

                relevance_map = cv.flip(relevance_map, 0)
                cv.imwrite(os.path.join(root,"relevance_map_{}_{}_{}.png".format(sample,side,axis)), relevance_map*255)
                
                if "resnet" in file_path:
                        net = "Resnet"
                else:
                    net = "Panorama"

                obj = find_obj(obj_paths, sample, side)

                relevance_maps[i] = {
                    "net": net,
                    "sample" : sample,
                    "side" : side,
                    "axis" : axis,
                    "obj" : obj,
                    "relevance": relevance_maps_path,
                    "max" : relevance_map_max

                }
                i += 1

                # cv.waitKey(0)
                # cv.destroyAllWindows()

    relevance_maps_df = pd.DataFrame.from_dict(relevance_maps).T
    print(relevance_maps_df)
    print(os.path.join(maps_path,"relevance_maps.csv"))
    relevance_maps_df.to_csv(os.path.join(maps_path,"relevance_maps.csv"), sep=";", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-w",
        "--width",
        type=int,
        help="",
        required=False,
        default=108
    )

    parser.add_argument(
        "-thr",
        "--threshold",
        type=float,
        help="",
        required=False,
        default=0.0
    )


    parser.add_argument(
        "-m",
        "--maps",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "-obj",
        "--obj_paths",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "--as_png",
        action="store_true",
        help="",
        required=False,
        default=False
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

    main(args.width, args.maps, args.data, args.obj_paths, args.threshold, args.as_png)
