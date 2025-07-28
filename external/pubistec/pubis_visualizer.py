import os
import sys
import argparse
import trimesh as tm
import numpy as np
import gc
import pyvista as pv
import cv2 as cv

def project_into_yz_plane(mesh, orientation = "right"):    
    if orientation == "right":
        position = [-1, 0, 0]
    elif orientation == "left":
        position = [1, 0, 0]
    elif orientation == "front":
        position = [0, 0, 1]

    pv.global_theme.camera = {'position': position, 'viewup': [0, 1, 0]}
    mesh2 = pv.wrap(mesh)
    # projection = mesh2.project_points_to_plane(origin = (0, 0, 0), normal = (0, 0, 1))
    pl = pv.Plotter(off_screen = True)
    pl.add_mesh(mesh2)
    pl.set_background('black')
    #pl.show()
    return pl.screenshot()

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

def main(input_folder, output_folder):

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".obj"):
                mesh_path = os.path.join(root, file)
                mesh_name = os.path.basename(mesh_path).split(".", 1)[0]
                case_name = mesh_name.split("_")[0]
                print("Processing {}...".format(mesh_path))
                # if not os.path.exists(os.path.join(output_folder, case_name, mesh_name) + "_right.png"):
                scene = tm.load_mesh(mesh_path)
            
                mesh = as_mesh(scene)
                

                image_right = project_into_yz_plane(mesh, orientation="right")
                image_left = project_into_yz_plane(mesh, orientation="left")
                image_front = project_into_yz_plane(mesh, orientation="front")

                xmin = 205
                xmax = 819
                image_right = image_right[:,xmin:xmax]
                image_left = image_left[:,xmin:xmax]
                image_front = image_front[:,xmin:xmax]

                image = cv.hconcat([image_right, image_front, image_left])

                if not os.path.exists(os.path.join(output_folder, case_name)):
                    os.makedirs(os.path.join(output_folder, case_name))


                # cv.imwrite(os.path.join(output_folder, case_name, mesh_name) + "_1.png", image_right)
                # cv.imwrite(os.path.join(output_folder, case_name, mesh_name) + "_2.png", image_front)
                # cv.imwrite(os.path.join(output_folder, case_name, mesh_name) + "_3.png", image_left)

                cv.imwrite(os.path.join(output_folder, case_name, mesh_name) + ".png", image)
                
                del scene, mesh
                gc.collect()
                # else:
                #     print("{} already processed".format(mesh_path))
                print()


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
                    

    args = parser.parse_args()

    main(args.input_folder, args.output_folder)