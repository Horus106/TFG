import trimesh
import numpy as np
import argparse

def compute_iou(values1, values2, threshold=0.5):
    """Calcula IoU (Jaccard) entre dos conjuntos de valores de atención binarizados por umbral."""
    mask1 = values1 >= threshold
    mask2 = values2 >= threshold
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        # Si ninguno de los dos mapas tiene valores sobre el umbral,
        # definimos IoU como 1.0 (ambos vacíos coinciden completamente).
        return 1.0
    return inter / union

def compute_mse(values1, values2):
    """Calcula el Error Cuadrático Medio entre dos arrays de valores de atención."""
    # Convertir a float para evitar problemas con enteros
    diff = values1.astype(float) - values2.astype(float)
    return np.mean(diff ** 2)

def normalizar_vertices(vertices):
    centro = vertices.mean(axis=0)
    vertices_centrados = vertices - centro
    escala = np.linalg.norm(vertices_centrados, axis=1).max()
    return vertices_centrados / escala

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparar mapas de atención en dos modelos .obj con colores por vértice.")
    parser.add_argument("model_pred", help="Ruta al modelo .obj con el mapa de atención predicho (colores por vértice).")
    parser.add_argument("model_ref", help="Ruta al modelo .obj con el mapa de atención de referencia.")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Umbral para binarizar la atención (0 a 1). Por defecto 0.5.")
    args = parser.parse_args()

    # Cargar los dos modelos .obj
    mesh_pred = trimesh.load(args.model_pred, process=False)
    mesh_ref = trimesh.load(args.model_ref, process=False)

    # Si la carga produce escenas con múltiples geometrías, combinarlas en una sola malla (opcional, según el caso)
    if isinstance(mesh_pred, trimesh.Scene):
        # Unir geometrías de la escena en un solo mesh
        mesh_pred = trimesh.util.concatenate([trimesh.Trimesh(vertices=g.vertices, faces=g.faces, 
                                                              vertex_colors=g.visual.vertex_colors) 
                                              for g in mesh_pred.geometry.values()])
    if isinstance(mesh_ref, trimesh.Scene):
        mesh_ref = trimesh.util.concatenate([trimesh.Trimesh(vertices=g.vertices, faces=g.faces, 
                                                             vertex_colors=g.visual.vertex_colors) 
                                             for g in mesh_ref.geometry.values()])
        
    # Comparar vértices y caras
    print(f"[{mesh_pred}] Vértices: {len(mesh_pred.vertices)}, Caras: {len(mesh_pred.faces)}")
    print(f"[{mesh_ref}] Vértices: {len(mesh_ref.vertices)}, Caras: {len(mesh_ref.faces)}")

    v1 = normalizar_vertices(mesh_pred.vertices)
    v2 = normalizar_vertices(mesh_ref.vertices)

    if np.allclose(v1, v2, atol=1e-3):
        print("✅ Los modelos tienen la misma forma y orden, solo cambia la escala.")
    else:
        print("⚠️ Los modelos tienen diferencias más allá de la escala.")

    # Extraer colores de vértice (RGBA) y calcular valores de atención [0,1] usando el canal rojo
    colors_pred = mesh_pred.visual.vertex_colors  # array Nx4
    colors_ref = mesh_ref.visual.vertex_colors    # array Nx4
    att_pred = colors_pred[:, 0].astype(float) / 255.0  # componente R normalizado
    att_ref = colors_ref[:, 0].astype(float) / 255.0    # componente R normalizado

    # Verificar que la cantidad de vértices coincide
    if att_pred.shape != att_ref.shape:
        print("Error: Los modelos tienen distinta cantidad de vértices, no se pueden comparar directamente.")
        exit(1)

    # Calcular métricas de similitud
    iou_value = compute_iou(att_pred, att_ref, threshold=args.threshold)
    mse_value = compute_mse(att_pred, att_ref)

    # Mostrar resultados
    print(f"Umbral de alta atención: {args.threshold}")
    print(f"IoU (Jaccard) de regiones de alta atención: {iou_value:.3f}")
    print(f"Error Cuadrático Medio (MSE) de atención continua: {mse_value:.6f}")
