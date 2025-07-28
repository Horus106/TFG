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

def load_relevance_map(file_path):
    """Carga un mapa de relevancia desde un archivo pickle"""
    try:
        with open(file_path, 'rb') as f:
            relevance_map = pkl.load(f)
        return relevance_map
    except Exception as e:
        print(f"[ERROR] No se pudo cargar {file_path}: {e}")
        return None

def combine_relevance_maps(map1, map2, method='average'):
    """
    Combina dos mapas de relevancia usando diferentes métodos
    
    Args:
        map1, map2: arrays numpy con los mapas de relevancia
        method: 'average', 'max', 'weighted_average'
    """
    if method == 'average':
        return (map1 + map2) / 2.0
    elif method == 'max':
        return np.maximum(map1, map2)
    elif method == 'weighted_average':
        # Puede ser personalizado según necesidades
        return (map1 * 0.6 + map2 * 0.4)
    else:
        raise ValueError(f"Método no reconocido: {method}")

def find_ensemble_files(maps_path, ensemble_views, sample, side):
    """
    Construye las rutas directamente basándose en el patrón conocido:
    maps_path/SAMPLE_SIDE/SAMPLE_SIDE_AXIS_2channels.pkl
    
    Args:
        maps_path: ruta base donde buscar
        ensemble_views: tupla con las vistas (ej: ('X', 'Y'))
        sample: nombre de la muestra
        side: lado (L/R)
    
    Returns:
        dict con las rutas de los archivos encontrados
    """
    found_files = {}
    
    print(f"[DEBUG] Buscando archivos para {sample}_{side} en {maps_path}")
    print(f"[DEBUG] Vistas objetivo: {ensemble_views}")
    
    # Construir ruta de la carpeta específica: SAMPLE_SIDE
    ensemble_name = ''.join(ensemble_views)  # ('X', 'Y') -> 'XY'
    
    sample_folder = f"ensemble_{ensemble_name}/2d_heatmaps/{sample}_{side}"
    sample_path = os.path.join(maps_path, sample_folder)
    
    print(f"[DEBUG] Verificando carpeta: {sample_path}")
    
    # Verificar que la carpeta existe
    if not os.path.exists(sample_path):
        print(f"[DEBUG] ✗ Carpeta no encontrada: {sample_path}")
        return found_files
    
    if not os.path.isdir(sample_path):
        print(f"[DEBUG] ✗ La ruta no es una carpeta: {sample_path}")
        return found_files
    
    # Buscar cada vista requerida
    for view in ensemble_views:
        # Construir nombre esperado del archivo
        expected_filename = f"{sample}_{side}_{view}_2channels.pkl"
        expected_path = os.path.join(sample_path, expected_filename)
        
        print(f"[DEBUG] Buscando: {expected_path}")
        
        # Verificar que el archivo existe
        if os.path.exists(expected_path) and os.path.isfile(expected_path):
            found_files[view] = expected_path
            print(f"[DEBUG] ✓ Encontrado {view}: {expected_path}")
        else:
            print(f"[DEBUG] ✗ No encontrado {view}: {expected_path}")
    
    print(f"[DEBUG] Archivos encontrados para {sample}_{side}: {found_files}")
    return found_files

def process_ensemble_view(maps_path, data_path, obj_paths, ensemble_views, width, height, threshold, combination_method='average'):
    """
    Procesa un ensemble de vistas específico (ej: XY, XZ, YZ)
    """
    relevance_maps = {}
    i = 0
    skipped_files = []
    ensemble_name = ''.join(ensemble_views)
    
    print(f"\n=== PROCESANDO ENSEMBLE {ensemble_name} ===")
    
    # Encontrar todas las combinaciones únicas de sample_side
    samples_sides = set()
    
    for root, folders, files in os.walk(maps_path):
        for file in files:
            if not file.endswith("_2channels.pkl"):
                continue
                
            try:
                sample, side, axis, _ = file.split("_")
                if axis in ensemble_views:
                    samples_sides.add((sample, side))
            except ValueError:
                continue
    
    print(f"Encontradas {len(samples_sides)} combinaciones sample_side para procesar")
    
    # Procesar cada combinación sample_side
    for sample, side in samples_sides:
        print(f"\nProcesando: {sample}_{side}")
        
        # Buscar archivos para este ensemble
        ensemble_files = find_ensemble_files(maps_path, ensemble_views, sample, side)
        
        # Verificar que tengamos todos los archivos necesarios
        missing_views = set(ensemble_views) - set(ensemble_files.keys())
        if missing_views:
            print(f"[AVISO] Faltan vistas {missing_views} para {sample}_{side}")
            skipped_files.append(f"{sample}_{side}_{ensemble_name} - Faltan vistas: {missing_views}")
            continue
        
        print(f"Archivos encontrados: {ensemble_files}")
        
        # Cargar mapas de relevancia
        maps_to_combine = {}
        all_loaded = True
        
        for view, file_path in ensemble_files.items():
            relevance_map = load_relevance_map(file_path)
            if relevance_map is None:
                skipped_files.append(f"{sample}_{side}_{ensemble_name} - Error cargando {view}: {file_path}")
                all_loaded = False
                break
            
            # Redimensionar y normalizar individualmente
            relevance_map = cv.resize(relevance_map, (width, height), cv.INTER_CUBIC)
            maps_to_combine[view] = relevance_map
            print(f"Mapa {view} cargado. Rango: {np.min(relevance_map)} - {np.max(relevance_map)}")
        
        if not all_loaded:
            continue
        
        # Combinar mapas de relevancia
        view_list = list(ensemble_views)
        combined_map = combine_relevance_maps(
            maps_to_combine[view_list[0]], 
            maps_to_combine[view_list[1]], 
            method=combination_method
        )
        
        print(f"Mapa combinado. Rango: {np.min(combined_map)} - {np.max(combined_map)}")
        
        # Guardar copia del mapa original combinado
        combined_map_orig = combined_map.copy()
        combined_map_orig_max = np.max(combined_map_orig)
        if combined_map_orig_max != 0:
            combined_map_orig = combined_map_orig / combined_map_orig_max
        
        # Aplicar máscara (usando la primera vista para la máscara)
        first_view = view_list[0]
        mask_path = os.path.join(data_path, f"{sample}_{side}", f"{sample}_{side}_0_panorama_ext_{first_view}_gray_mask.png")
        
        if not os.path.exists(mask_path):
            print(f"[WARNING] Mask file not found: {mask_path}")
            skipped_files.append(f"{sample}_{side}_{ensemble_name} - Mask not found: {mask_path}")
            continue
        
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[ERROR] Could not read mask file: {mask_path}")
            skipped_files.append(f"{sample}_{side}_{ensemble_name} - Could not read mask: {mask_path}")
            continue
        
        mask = cv.resize(mask, (width, height), cv.INTER_CUBIC)
        mask = mask / 255.0
        
        combined_map_masked = combined_map * mask
        combined_map_masked = combined_map_masked.astype('float32')
        
        # Crear directorio de salida
        output_dir = os.path.join(maps_path, f"ensemble_{ensemble_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar mapa completo
        try:
            cv.imwrite(os.path.join(output_dir, f"relevance_map_{sample}_{side}_{ensemble_name}_complete.png"), 
                      combined_map_masked*255)
        except Exception as e:
            print(f"[WARNING] Could not save complete relevance map: {e}")
        
        # Normalizar y aplicar threshold
        combined_map_max = np.max(combined_map_masked)
        if combined_map_max != 0:
            combined_map_masked = combined_map_masked / combined_map_max
        
        combined_map_thresholded = np.where(combined_map_masked >= threshold, combined_map_masked, 0)
        
        # Rutas de salida
        relevance_maps_path = os.path.join(output_dir, f"relevance_map_{sample}_{side}_{ensemble_name}.png")
        
        # Cargar panorama (usando la primera vista)
        panorama_path = os.path.join(data_path, f"{sample}_{side}", f"{sample}_{side}_0_panorama_ext_{first_view}.png")
        
        if not os.path.exists(panorama_path):
            print(f"[WARNING] Panorama file not found: {panorama_path}")
            skipped_files.append(f"{sample}_{side}_{ensemble_name} - Panorama not found: {panorama_path}")
            continue
        
        panorama = cv.imread(panorama_path, cv.IMREAD_GRAYSCALE)
        if panorama is None:
            print(f"[ERROR] Could not read panorama file: {panorama_path}")
            skipped_files.append(f"{sample}_{side}_{ensemble_name} - Could not read panorama: {panorama_path}")
            continue
        
        panorama = cv.resize(panorama, (width, height), cv.INTER_CUBIC)
        
        # Generar heatmaps
        try:
            for cmap in ["jet", "bwr"]:
                # Heatmap completo
                heatmap = heatmap_on_image(
                    cv.resize(combined_map_orig, (width, height), cv.INTER_CUBIC), 
                    panorama, w=width, h=height, hcmap=cmap
                )
                cv.imwrite(os.path.join(output_dir, f"heatmap_{sample}_{side}_{ensemble_name}_complete_{cmap}.png"), heatmap)
                
                # Heatmap con threshold
                heatmap = heatmap_on_image(
                    combined_map_thresholded, panorama, w=width, h=height, hcmap=cmap
                )
                cv.imwrite(os.path.join(output_dir, f"heatmap_{sample}_{side}_{ensemble_name}_{cmap}.png"), heatmap)
        except Exception as e:
            print(f"[WARNING] Could not generate heatmaps: {e}")
        
        # Flip y guardar mapa final
        combined_map_final = cv.flip(combined_map_thresholded, 0)
        
        try:
            cv.imwrite(relevance_maps_path, combined_map_final*255)
        except Exception as e:
            print(f"[WARNING] Could not save final relevance map: {e}")
        
        # Determinar red (asumiendo que es consistente entre archivos)
        first_file_path = list(ensemble_files.values())[0]
        net = "Resnet" if "resnet" in first_file_path else "Panorama"
        obj = find_obj(obj_paths, sample, side) if obj_paths else None
        
        relevance_maps[i] = {
            "net": net,
            "sample": sample,
            "side": side,
            "axis": ensemble_name,
            "view": ensemble_name,
            "obj": obj,
            "relevance": relevance_maps_path,
            "max": combined_map_max,
            "combination_method": combination_method
        }
        i += 1
    
    return relevance_maps, skipped_files

def main(width, maps_path, data_path, obj_paths, threshold, ensembles, combination_method='average'):
    height = int(width / 3)
    
    # Definir ensembles disponibles
    available_ensembles = ['XY', 'XZ', 'YZ']
    
    # Validar ensembles
    if isinstance(ensembles, str):
        ensembles = [ensembles]
    
    for ensemble in ensembles:
        if ensemble not in available_ensembles:
            print(f"[ERROR] Ensemble '{ensemble}' no válido. Ensembles disponibles: {available_ensembles}")
            return
    
    all_relevance_maps = {}
    all_skipped_files = []
    total_processed = 0
    
    # Procesar ensembles
    for ensemble in ensembles:
        print(f"\n{'='*50}")
        print(f"PROCESANDO ENSEMBLE: {ensemble}")
        print(f"{'='*50}")
        
        ensemble_views = tuple(ensemble)  # 'XY' -> ('X', 'Y')
        
        relevance_maps, skipped_files = process_ensemble_view(
            maps_path, data_path, obj_paths, ensemble_views, 
            width, height, threshold, combination_method
        )
        
        # Actualizar índices para evitar conflictos
        for key, value in relevance_maps.items():
            all_relevance_maps[total_processed + key] = value
        
        all_skipped_files.extend(skipped_files)
        total_processed += len(relevance_maps)
        
        # Guardar CSV específico para este ensemble
        if relevance_maps:
            relevance_maps_df = pd.DataFrame.from_dict(relevance_maps).T
            csv_path = os.path.join(maps_path, f"relevance_maps_ensemble_{ensemble}.csv")
            print(f"Guardando CSV para ensemble {ensemble} en: {csv_path}")
            relevance_maps_df.to_csv(csv_path, sep=";", index=False)

    # Print summary
    print(f"\n{'='*50}")
    print(f"RESUMEN FINAL")
    print(f"{'='*50}")
    print(f"Ensembles procesados: {', '.join(ensembles)}")
    print(f"Método de combinación: {combination_method}")
    print(f"Archivos procesados exitosamente: {total_processed}")
    print(f"Archivos omitidos: {len(all_skipped_files)}")
    
    if all_skipped_files:
        print(f"\nArchivos omitidos:")
        for skipped in all_skipped_files:
            print(f"  - {skipped}")

    # Guardar CSV completo con todos los ensembles
    if all_relevance_maps:
        all_relevance_maps_df = pd.DataFrame.from_dict(all_relevance_maps).T
        print(f"\nDataFrame completo generado:")
        print(all_relevance_maps_df)
        csv_path = os.path.join(maps_path, "relevance_maps_ensembles.csv")
        print(f"Guardando CSV completo en: {csv_path}")
        all_relevance_maps_df.to_csv(csv_path, sep=";", index=False)
    else:
        print(f"\n[WARNING] No se procesó ningún archivo exitosamente.")

def load_args_from_json(json_file):
    """Carga argumentos desde un archivo JSON"""
    with open(json_file, 'r') as f:
        args_dict = json.load(f)
    return args_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesador de mapas de relevancia para ensembles")

    # Argumentos básicos
    parser.add_argument("-w", "--width", type=int, default=108, help="Ancho de las imágenes")
    parser.add_argument("-thr", "--threshold", type=float, default=0.0, help="Umbral para filtrar mapas de relevancia")
    parser.add_argument("-m", "--maps", type=str, required=False, help="Ruta a los archivos .pkl de mapas de relevancia")
    parser.add_argument("-d", "--data", type=str, required=False, help="Ruta a los datos originales (máscaras y panoramas)")
    parser.add_argument("-obj", "--obj_paths", type=str, required=False, default=None, help="(Opcional) Ruta a los .obj para incluir en el CSV")
    parser.add_argument("-v", "--verbose", type=int, default=0, help="Nivel de verbosidad")
    
    # Argumentos para ensembles
    parser.add_argument("--ensembles", type=str, nargs='+', choices=['XY', 'XZ', 'YZ'],
                        required=True, help="Ensemble(s) a procesar (combinaciones de vistas)")
    
    parser.add_argument("--combination_method", type=str, choices=['average', 'max', 'weighted_average'], 
                        default='average', help="Método para combinar mapas en ensembles")
    
    # Archivo JSON de parámetros
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
    if not args.ensembles:
        print("Error: Se requiere especificar al menos un ensemble (--ensembles)")
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
    print(f"  ensembles: {args.ensembles}")
    print(f"  combination_method: {args.combination_method}")

    main(args.width, args.maps, args.data, args.obj_paths, args.threshold, 
         args.ensembles, args.combination_method)