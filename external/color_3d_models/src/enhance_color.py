import argparse
import logging
import re
import os
import vtk
import pymeshlab
import tempfile
import multiprocessing

def color_model_heatmap(model_path, v_colors):
    vtk.vtkObject.GlobalWarningDisplayOff()
    temp_dir = tempfile.TemporaryDirectory()
    # print(temp_dir.name)

    reader = vtk.vtkOBJReader()
    reader.SetFileName(model_path)
    reader.Update()

    pubis = reader.GetOutput()

    atention = vtk.vtkFloatArray()
    atention.SetName("Atention")
    # print(v_colors[0])
    for value in v_colors:
        atention.InsertNextValue(value[0])

    pubis.GetPointData().SetScalars(atention)

    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.667, 0.0)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetLookupTable(lut)
    mapper.SetInputData(pubis)
    mapper.SetScalarRange(pubis.GetScalarRange())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    ren1 = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren1)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    ren1.AddActor(actor)
    
    # renWin.SetSize(512, 512)
    # renWin.SetWindowName('Rainbow')

    # iren.Initialize()

    # iren.Start()

    exporter = vtk.vtkX3DExporter()

    exporter.SetActiveRenderer(ren1)
    exporter.SetRenderWindow(renWin)

    name = os.path.basename(os.path.splitext(model_path)[0])
    name = re.search("[0-9]+_[A-Za-z]+_[XYZ]?_?", name)
    name = str(name.group())

    exporter.SetFileName(os.path.join(temp_dir.name, name+"full_colored.x3d"))
    exporter.Write()

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.path.join(temp_dir.name, name+"full_colored.x3d"))
    # ms.save_current_mesh(os.path.join(os.path.dirname(folder), name+"_full_colored.obj"))
    ms.save_current_mesh(os.path.join(os.path.dirname(model_path), name+"full_colored.obj"))
    ms.apply_color_laplacian_smoothing_per_vertex(iteration = 5)
    # ms.save_current_mesh(os.path.join(os.path.dirname(folder), name+"_full_colored_smooth.obj"))
    ms.save_current_mesh(os.path.join(os.path.dirname(model_path), name+"full_colored_smooth.obj"))

    print(model_path)

    temp_dir.cleanup()


def enhance_color(model_path):
    v_colors = []
    with open(model_path, "r") as model_obj:
        obj_lines = model_obj.readlines()

    for i, line in enumerate(obj_lines):
        if line.startswith("v "):
            line = [ float(i) for i in line.split(" ")[4:] ]
            v_colors.append(line)
        
    color_model_heatmap(model_path, v_colors)

def main(folder):
    
    files_to_enhance = []
    for root, folders, files in os.walk(folder):
        for file in files:
            v_colors = []
            if file.endswith(".obj") and not "full" in file:
                model_path = os.path.join(root, file)
                files_to_enhance.append(model_path)

    pool = multiprocessing.Pool(processes=10)
    pool.map(enhance_color, files_to_enhance)
                



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