import os
import re
import argparse
import trimesh
import numpy as np
import pandas as pd

def compute_iou(values1, values2, threshold=0.5):
    mask1 = values1 >= threshold
    mask2 = values2 >= threshold
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return 1.0 if union == 0 else inter / union

def compute_mse(values1, values2):
    diff = values1.astype(float) - values2.astype(float)
    return np.mean(diff ** 2)

def normalizar_vertices(vertices):
    centro = vertices.mean(axis=0)
    vertices_centrados = vertices - centro
    escala = np.linalg.norm(vertices_centrados, axis=1).max()
    return vertices_centrados / escala

def cargar_malla(path):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=g.vertices, faces=g.faces, 
                            vertex_colors=g.visual.vertex_colors)
            for g in mesh.geometry.values()
        ])
    return mesh

def buscar_archivos_obj_recursivo(folder, filtro_nombre=None):
    archivos = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".obj") and (filtro_nombre is None or filtro_nombre in f):
                archivos.append(os.path.join(root, f))
    return archivos

def emparejar_modelos(gt_folder, pred_folder):
    gt_dict = {}
    for f in buscar_archivos_obj_recursivo(gt_folder):
        match = re.match(r"(\d+_[A-Za-z]+)", os.path.basename(f))
        if match:
            nombre_base = match.group(1)
            gt_dict[nombre_base] = f

    pred_files = buscar_archivos_obj_recursivo(pred_folder, filtro_nombre="full_colored_smooth")
    print(f"\n📂 Archivos pred encontrados:")
    for p in pred_files:
        print(f" - {p}")

    emparejamientos = []
    for pred_file in pred_files:
        match = re.match(r"(\d+_[A-Za-z]+)_", os.path.basename(pred_file))
        if match:
            nombre_base = match.group(1)
            if nombre_base in gt_dict:
                emparejamientos.append((gt_dict[nombre_base], pred_file, nombre_base))

    print(f"\n🔗 Emparejamientos encontrados: {len(emparejamientos)}")
    return emparejamientos


def main(gt_folder, pred_folder, nombre_salida, threshold=0.5):
    resultados = []
    emparejamientos = emparejar_modelos(gt_folder, pred_folder)

    for gt_path, pred_path, nombre in emparejamientos:
        try:
            mesh_gt = cargar_malla(gt_path)
            mesh_pred = cargar_malla(pred_path)

            v_gt = mesh_gt.vertices
            v_pred = mesh_pred.vertices
            c_gt = mesh_gt.visual.vertex_colors
            c_pred = mesh_pred.visual.vertex_colors

            print(f"[{nombre}] GT: {len(v_gt)} vértices, {len(mesh_gt.faces)} caras, {c_gt.shape} colores")
            print(f"[{nombre}] Pred: {len(v_pred)} vértices, {len(mesh_pred.faces)} caras, {c_pred.shape} colores")

            if len(v_gt) != len(v_pred):
                print(f"[{nombre}] ❌ Diferente número de vértices. Se omite.")
                continue

            if c_gt.shape[0] != v_gt.shape[0] or c_pred.shape[0] != v_pred.shape[0]:
                print(f"[{nombre}] ❌ Los vertex_colors no coinciden con el número de vértices.")
                continue

            v1 = normalizar_vertices(v_pred)
            v2 = normalizar_vertices(v_gt)

            if not np.allclose(v1, v2, atol=1e-3):
                print(f"[{nombre}] ⚠️ Diferencias geométricas más allá de la escala.")

            att_pred = c_pred[:, 0].astype(float) / 255.0
            att_gt = c_gt[:, 0].astype(float) / 255.0

            iou = compute_iou(att_pred, att_gt, threshold)
            mse = compute_mse(att_pred, att_gt)

            resultados.append({
                "nombre": nombre,
                "gt_file": os.path.relpath(gt_path, start=gt_folder),
                "pred_file": os.path.relpath(pred_path, start=pred_folder),
                "iou": round(iou, 4),
                "mse": round(mse, 6),
                "n_vertices": len(att_pred)
            })

            print(f"[{nombre}] ✅ IoU: {iou:.3f} | MSE: {mse:.6f}")

        except Exception as e:
            print(f"[{nombre}] ❌ Error procesando: {e}")

    # Guardar CSV
    df = pd.DataFrame(resultados)
    df.sort_values("nombre").to_csv(nombre_salida, index=False)
    print(f"\n✅ Resultados guardados en {nombre_salida}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación de mapas de calor en modelos .obj")
    parser.add_argument("gt_folder", help="Carpeta con los modelos Ground Truth")
    parser.add_argument("pred_folder", help="Carpeta con los modelos generados por IA")
    parser.add_argument("output_csv", help="Nombre del archivo CSV de salida")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Umbral de atención (por defecto 0.5)")
    args = parser.parse_args()

    main(args.gt_folder, args.pred_folder, args.output_csv, args.threshold)
