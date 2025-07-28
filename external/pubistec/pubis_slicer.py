import os
import sys
import argparse
import trimesh as tm
import numpy as np
import gc

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, tm.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = tm.util.concatenate(
                tuple(tm.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, tm.Trimesh))
        mesh = scene_or_mesh
    return mesh

def main(input_folder, output_folder, depth, verbose):

    depths = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".obj"):
                mesh_path = os.path.join(root, file)
                mesh_name = os.path.basename(mesh_path).split(".", 1)[0]
                case_name = mesh_name.split("_")[0]
                print("Processing {}...".format(mesh_path))
                if not os.path.exists(os.path.join(output_folder, case_name, mesh_name) + "_sliced.obj"):
                # if True:
                    scene = tm.load_mesh(mesh_path)


                    mesh = as_mesh(scene)

                    tx = -1*mesh.center_mass[0]
                    ty = -1*mesh.center_mass[1]
                    tz = -1*mesh.center_mass[2]
                    T = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])
                    mesh.apply_transform(T)

                    z_depth = mesh.bounds[1][2] - mesh.bounds[0][2]
                    depths.append(z_depth)
                    print("Depth: {}".format(z_depth))
                    print(mesh.bounds)

                    normal = [0.0, 0.0, 1.0]
                    point = [0.0, 0.0, mesh.bounds[1][2]-depth]
                    print(point)
                    sliced_pubis = mesh.slice_plane(point, normal)

                    if not os.path.exists(os.path.join(output_folder, case_name)):
                        os.makedirs(os.path.join(output_folder, case_name))
                    
                    z_depth = sliced_pubis.bounds[1][2] - sliced_pubis.bounds[0][2]
                    print("Depth: {}".format(z_depth))

                    sliced_pubis.export(os.path.join(output_folder, case_name, mesh_name) + "_sliced.obj", file_type = "obj")
                    print("Mean depth by far: {}".format(np.mean(np.array(depths))))
                    del scene, mesh, sliced_pubis
                    gc.collect()

                else:
                    print("{} already processed".format(mesh_path))

                print()

    depths = np.array(depths)
    print("Mean depth: {}".format(np.mean(depths)))


if __name__ == "__main__":


    parser = argparse.ArgumentParser(
                    prog = 'Pubistec',
                    description = 'App para filetear pubis y convertir a ply')
    
    parser.add_argument('-i', '--input_folder', 
                        help='Directorio a procesar',
                        required=True)
          
    parser.add_argument('-o', '--output_folder', 
                        help='Directorio de almacenamiento',
                        default='output')
             
    parser.add_argument('-d','--depth',
                        help='Profundidad a recortar (defecto:25)',
                        type=int,
                        default=25)
    
    parser.add_argument('-v', '--verbose',
                        help='visualizar cada pubis',
                        action='store_true')
                    

    args = parser.parse_args()

    main(args.input_folder, args.output_folder, args.depth, args.verbose)