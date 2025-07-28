#include "mesh3d.h"

/** @file main.cpp
 * 	@author [Alejandro Manzanares Lemus](https://github.com/Alexmnzlms)
 */

int main(int argc, char * argv[]) {

	srand (time(NULL));

	if(argc != 7){
		std::cout << "Wrong parameters" << std::endl;
		std::cout << "bin/panorama_extended [name] [axis] [relive route to 3D model]"
		<< " [relive route to image] [output folder] [threshold]" << std::endl;
		exit(-1);
	}

	std::string name = argv[1];
	std::string axis_param = argv[2];
	std::string path = argv[3];
	std::string path_relevance = argv[4];
	std::string output = argv[5];
	std::string threshold = argv[6];

	Axis axis;

	std::cout << "Loading " << name << "\tPath: " << path << "..." << std::endl;

	Mesh3D malla(name, path, true);
	// Mesh3D malla_dch(name, path, true);

	if (axis_param == "X"){
		axis = X;
	} else if (axis_param == "Y"){
		axis = Y;
	} else if (axis_param == "Z"){
		axis = Z;
	}

	if (malla.num_vertexs() > 0){
		std::cout << "Loaded " << malla.get_name() << std::endl;

		std::cout << "Coloring: " << path_relevance << "..." << std::endl;

		malla.color_3d_model(path_relevance, axis, std::stod(threshold));

		malla.export_obj(output+"/"+name+"_"+axis_param+"_colored.obj", true);

	}

	// if (malla_dch.num_vertexs() > 0){
	// 	std::cout << "Loaded " << malla_dch.get_name() << std::endl;

	// 	std::cout << "Coloring: " << path3 << "..." << std::endl;

	// 	malla_dch.color_3d_model(path3, axis, std::stod(threshold));

	// 	malla_dch.export_obj(output+"/"+name+"_"+axis_param+"_dch_colored.obj", true);

	// }
	
	return 0;
}