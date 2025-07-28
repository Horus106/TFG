import argparse
import logging
import pickle as pkl
import cv2 as cv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import json

def find_obj(obj_folder, sample, side):
    if obj_folder is None:
        return None
    for root, folders, files in os.walk(obj_folder):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext == ".obj":
                file_sample, file_side, _ = file.split("_")
                if sample == file_sample and side == file_side:
                    return os.path.join(os.path.abspath(root), file)
    return None

def heatmap_on_image(heatmap, image, w=108, h=36, hcmap="jet"):
    plt.figure(figsize=(w,h),dpi=1)
    vmin = 0 if hcmap == "jet" else -1
    hmax = sns.heatmap(heatmap, vmin=vmin, vmax=1,
                    cmap=hcmap,
                    xticklabels=False, yticklabels=False, cbar=False, 
                    alpha=1, zorder=1)
    hmax.imshow(image,
                cmap="gray",
                alpha=0.5,
                aspect=hmax.get_aspect(),
                extent=hmax.get_xlim() + hmax.get_ylim(),
                zorder=2)
    plt.tight_layout()
    ax = plt.gca()
    canvas = ax.figure.canvas
    canvas.draw()
    hoi = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    hoi = np.reshape(hoi, (h,w,3))
    hoi = cv.cvtColor(hoi, cv.COLOR_BGR2RGBA)
    plt.close()
    return hoi

def main(width, maps_path, data_path, obj_paths, threshold):
    height = int(width / 3)
    relevance_maps = {}
    i = 0
    skipped_files = []

    for root, folders, files in os.walk(maps_path):
        for file in files:
            if not file.endswith("_2channels.pkl"):
                continue

            file_path = os.path.join(root, file)

            try:
                sample, side, axis, _ = file.split("_")
            except ValueError:
                print(f"[AVISO] Nombre de archivo no compatible: {file}")
                continue

            print(f"Procesando: {sample}, {side}, {axis}, {file_path}")

            # Check if pickle file exists and can be loaded
            try:
                with open(file_path, 'rb') as f:
                    relevance_map = pkl.load(f)  # ya tiene forma (36, 108)
            except Exception as e:
                print(f"[ERROR] No se pudo cargar {file_path}: {e}")
                skipped_files.append(f"{file} - Error loading pickle: {e}")
                continue

            relevance_map = cv.resize(relevance_map, (width, height), cv.INTER_CUBIC)
            print(f"Relevance map range: {np.min(relevance_map)} - {np.max(relevance_map)}")

            relevance_map_orig = relevance_map.copy()
            relevance_map_orig_max = np.max(relevance_map_orig)
            if relevance_map_orig_max != 0:
                relevance_map_orig = relevance_map_orig / relevance_map_orig_max

            # Check mask file
            mask_path = os.path.join(data_path, f"{sample}_{side}", f"{sample}_{side}_0_panorama_ext_{axis}_gray_mask.png")
            if not os.path.exists(mask_path):
                print(f"[WARNING] Mask file not found: {mask_path}")
                skipped_files.append(f"{file} - Mask not found: {mask_path}")
                continue
            
            mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[ERROR] Could not read mask file: {mask_path}")
                skipped_files.append(f"{file} - Could not read mask: {mask_path}")
                continue
            
            mask = cv.resize(mask, (width, height), cv.INTER_CUBIC)
            mask = mask / 255.0

            relevance_map = relevance_map * mask
            relevance_map = relevance_map.astype('float32')

            # Save complete relevance map
            try:
                cv.imwrite(os.path.join(root, f"relevance_map_{sample}_{side}_{axis}_complete.png"), relevance_map*255)
            except Exception as e:
                print(f"[WARNING] Could not save complete relevance map: {e}")

            relevance_map[:, :height] += relevance_map[:, 2*height:]
            relevance_map = relevance_map[:, :2*height]

            relevance_map_max = np.max(relevance_map)
            if relevance_map_max != 0:
                relevance_map = relevance_map / relevance_map_max

            relevance_map = np.where(relevance_map >= threshold, relevance_map, 0)

            relevance_maps_path = os.path.join(os.path.abspath(root), f"relevance_map_{sample}_{side}_{axis}.png")

            # Check panorama file
            panorama_path = os.path.join(data_path, f"{sample}_{side}", f"{sample}_{side}_0_panorama_ext_{axis}.png")
            if not os.path.exists(panorama_path):
                print(f"[WARNING] Panorama file not found: {panorama_path}")
                skipped_files.append(f"{file} - Panorama not found: {panorama_path}")
                continue
            
            panorama = cv.imread(panorama_path, cv.IMREAD_GRAYSCALE)
            if panorama is None:
                print(f"[ERROR] Could not read panorama file: {panorama_path}")
                skipped_files.append(f"{file} - Could not read panorama: {panorama_path}")
                continue
            
            panorama = cv.resize(panorama, (width, height), cv.INTER_CUBIC)

            # Generate heatmaps
            try:
                for cmap in ["jet", "bwr"]:
                    heatmap = heatmap_on_image(cv.resize(relevance_map_orig, (width, height), cv.INTER_CUBIC), panorama, w=width, h=height, hcmap=cmap)
                    cv.imwrite(os.path.join(root, f"heatmap_{sample}_{side}_{axis}_complete_{cmap}.png"), heatmap)

                    panorama_no_extend = panorama[:, :2*height]
                    heatmap = heatmap_on_image(relevance_map, panorama_no_extend, w=2*height, h=height, hcmap=cmap)
                    cv.imwrite(os.path.join(root, f"heatmap_{sample}_{side}_{axis}_{cmap}.png"), heatmap)
            except Exception as e:
                print(f"[WARNING] Could not generate heatmaps: {e}")

            relevance_map = cv.flip(relevance_map, 0)
            
            try:
                cv.imwrite(os.path.join(root, f"relevance_map_{sample}_{side}_{axis}.png"), relevance_map*255)
            except Exception as e:
                print(f"[WARNING] Could not save final relevance map: {e}")

            net = "Resnet" if "resnet" in file_path else "Panorama"
            obj = find_obj(obj_paths, sample, side) if obj_paths else None

            relevance_maps[i] = {
                "net": net,
                "sample": sample,
                "side": side,
                "axis": axis,
                "obj": obj,
                "relevance": relevance_maps_path,
                "max": relevance_map_max
            }
            i += 1

    # Print summary
    print(f"\n=== RESUMEN ===")
    print(f"Archivos procesados exitosamente: {i}")
    print(f"Archivos omitidos: {len(skipped_files)}")
    
    if skipped_files:
        print(f"\nArchivos omitidos:")
        for skipped in skipped_files:
            print(f"  - {skipped}")

    if relevance_maps:
        relevance_maps_df = pd.DataFrame.from_dict(relevance_maps).T
        print(f"\nDataFrame generado:")
        print(relevance_maps_df)
        csv_path = os.path.join(maps_path, "relevance_maps.csv")
        print(f"Guardando CSV en: {csv_path}")
        relevance_maps_df.to_csv(csv_path, sep=";", index=False)
    else:
        print(f"\n[WARNING] No se procesó ningún archivo exitosamente.")

def load_args_from_json(json_file):
    """Carga argumentos desde un archivo JSON"""
    with open(json_file, 'r') as f:
        args_dict = json.load(f)
    return args_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Argumentos normales
    parser.add_argument("-w", "--width", type=int, default=108, help="")
    parser.add_argument("-thr", "--threshold", type=float, default=0.0, help="")
    parser.add_argument("-m", "--maps", type=str, required=False, help="")
    parser.add_argument("-d", "--data", type=str, required=False, help="")
    parser.add_argument("-obj", "--obj_paths", type=str, required=False, default=None, help="(Opcional) Ruta a los .obj para incluir en el CSV")
    parser.add_argument("-v", "--verbose", type=int, default=0, help="")
    
    # Nuevo argumento para archivo JSON
    parser.add_argument("--parameters", type=str, help="Archivo JSON con parámetros")

    args = parser.parse_args()

    # Si se proporciona un archivo JSON, cargar parámetros desde ahí
    if args.parameters:
        try:
            json_args = load_args_from_json(args.parameters)
            
            # Actualizar args con valores del JSON
            for key, value in json_args.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                else:
                    print(f"[AVISO] Parámetro desconocido en JSON: {key}")
            
            print(f"Parámetros cargados desde {args.parameters}")
            
        except Exception as e:
            print(f"Error al cargar parámetros desde JSON: {e}")
            exit(1)

    # Validar que los argumentos requeridos estén presentes
    if not args.maps:
        print("Error: Se requiere el parámetro 'maps'")
        exit(1)
    if not args.data:
        print("Error: Se requiere el parámetro 'data'")
        exit(1)

    log_level = [logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)]
    logging.getLogger().setLevel(log_level)
    logging.warning("Verbose level set to {}".format(logging.root.level))

    print(f"Ejecutando con parámetros:")
    print(f"  width: {args.width}")
    print(f"  threshold: {args.threshold}")
    print(f"  maps: {args.maps}")
    print(f"  data: {args.data}")
    print(f"  obj_paths: {args.obj_paths}")
    print(f"  verbose: {args.verbose}")

    main(args.width, args.maps, args.data, args.obj_paths, args.threshold)