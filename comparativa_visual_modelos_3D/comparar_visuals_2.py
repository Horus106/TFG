import os
import argparse
import pandas as pd
import re
import trimesh
import numpy as np

def buscar_archivos_obj_recursivo(directorio):
    archivos = []
    for root, _, files in os.walk(directorio):
        for file in files:
            if file.endswith(".obj"):
                archivos.append(os.path.join(root, file))
    return archivos

def extraer_id_lado(nombre_archivo):
    match = re.search(r"(\d+_(Dch|Izq))", nombre_archivo)
    return match.group(1) if match else None

def emparejar_modelos(gt_folder, pred_folder, modelo):
    gt_files = buscar_archivos_obj_recursivo(gt_folder)

    if modelo == "base":
        pred_files = buscar_archivos_obj_recursivo(pred_folder)
        pred_files = [f for f in pred_files if "_X" not in f and "_Y" not in f and "_Z" not in f]
    else:
        pred_files = buscar_archivos_obj_recursivo(pred_folder)
        pred_files = [f for f in pred_files if "full_colored_smooth" in f]

    emparejamientos = []

    for gt in gt_files:
        id_gt = extraer_id_lado(os.path.basename(gt))
        if not id_gt:
            continue

        for pred in pred_files:
            id_pred = extraer_id_lado(os.path.basename(pred))
            if id_gt == id_pred:
                emparejamientos.append((gt, pred))
                break

    print(f"\n📂 Archivos GT encontrados: {len(gt_files)}")
    print(f"📂 Archivos pred encontrados: {len(pred_files)}")
    print(f"🔗 Emparejamientos encontrados: {len(emparejamientos)}")

    return emparejamientos

def cargar_malla(path):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=g.vertices, faces=g.faces,
                            vertex_colors=g.visual.vertex_colors)
            for g in mesh.geometry.values()
        ])
    return mesh

def compute_iou(values1, values2, threshold):
    mask1 = values1 >= threshold
    mask2 = values2 >= threshold
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return 1.0 if union == 0 else inter / union

def compute_mse(values1, values2):
    diff = values1.astype(float) - values2.astype(float)
    return np.mean(diff ** 2)

def calcular_metricas(gt_path, pred_path, threshold):
    nombre = extraer_id_lado(os.path.basename(gt_path))
    mesh_gt = cargar_malla(gt_path)
    mesh_pred = cargar_malla(pred_path)

    v_gt = mesh_gt.vertices
    v_pred = mesh_pred.vertices
    c_gt = mesh_gt.visual.vertex_colors
    c_pred = mesh_pred.visual.vertex_colors

    if len(v_gt) != len(v_pred):
        print(f"[{nombre}] ❌ Diferente número de vértices. Se omite.")
        return None

    if c_gt.shape[0] != v_gt.shape[0] or c_pred.shape[0] != v_pred.shape[0]:
        print(f"[{nombre}] ❌ Vertex_colors no coinciden con vértices.")
        return None

    att_gt = c_gt[:, 0].astype(float) / 255.0
    att_pred = c_pred[:, 0].astype(float) / 255.0

    iou = compute_iou(att_pred, att_gt, threshold)
    mse = compute_mse(att_pred, att_gt)
    n_vertices = len(att_pred)

    return {
        "nombre": nombre,
        "gt_file": os.path.basename(gt_path),
        "pred_file": os.path.basename(pred_path),
        "iou": round(iou, 4),
        "mse": round(mse, 6),
        "n_vertices": n_vertices
    }

def main(gt_folder, pred_folder, output_csv, modelo, threshold):
    emparejamientos = emparejar_modelos(gt_folder, pred_folder, modelo)

    if not emparejamientos:
        print("⚠️  No se encontraron emparejamientos. Revisa los nombres.")
        return

    resultados = []
    for gt_path, pred_path in emparejamientos:
        resultado = calcular_metricas(gt_path, pred_path, threshold)
        if resultado:
            resultados.append(resultado)

    df = pd.DataFrame(resultados)
    df.sort_values("nombre").to_csv(output_csv, index=False)
    print(f"\n✅ Resultados guardados en: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_folder", help="Carpeta con los modelos GT")
    parser.add_argument("pred_folder", help="Carpeta con las predicciones")
    parser.add_argument("output_csv", help="Ruta del CSV de salida")
    parser.add_argument("modelo", help="Nombre del modelo (base, sep_X, sep_Y, ensemble_XY...)")
    parser.add_argument("threshold", type=float, help="Umbral para comparación")
    args = parser.parse_args()

    main(args.gt_folder, args.pred_folder, args.output_csv, args.modelo, args.threshold)
