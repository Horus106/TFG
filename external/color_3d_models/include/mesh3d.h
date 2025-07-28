#ifndef MALLA3D_H_INCLUDED
#define MALLA3D_H_INCLUDED
#define GLM_ENABLE_EXPERIMENTAL
#include <fstream>
#include <cstdio>
#include <stdio.h>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc.hpp>
#include "aux.h"

/** @file malla3d.h
 *  @brief Implementation of class Mesh3D
 * 	@author [Alejandro Manzanares Lemus](https://github.com/Alexmnzlms)
 * 
 */

/** @brief This class contains the information about the 3D model
 * 
 * This class implements the functions related to the PANORAMA representation 
 * (extended version) and the SYNPAN pose estimation algorithm.
 * 
 * It provides the methods for loading and exporting models, as well as the 
 * necessary transformations on the 3D mesh.
 * 
 * Main functionality is described in @link calculate_panorama() @endlink and
 * @link mesh_pose_norm() @endlink
 * 
 * @note PANORAMA extended method is extended by the use of the NDM gradient. 
 * However, for simplicity in the documentation and implementation we use an 
 * attribute \link panorama_extended \endlink that contains the 
 * PANORAMA representation extended 1.5 times. Any reference to the 
 * PANORAMA or PANORAMA extended representation is understood to mean 
 * PANORAMA extended in the sense of the use of the GNDM feature map
 * regardless of whether the size is extended or not..
 *  
 * 
 * @see Ensemble of PANORAMA-based convolutional neural networks for 3D model 
 * classification and retrieval @cite SFIKAS2018208
 */

class Mesh3D{
public:
	Mesh3D();
	Mesh3D(const std::string name, const std::string path, bool color = false);
	Mesh3D(const Mesh3D& mesh),
	~Mesh3D();
	std::string get_name();
	void set_name(const std::string name);
	int num_vertexs();
	int num_faces();
	void export_obj(const std::string path, bool color = false);
	void rotate_mesh(Axis axis_rot, float angle);
	void calculate_panorama(Map map, Axis axis, float precision, int power = 4);
	void mesh_pose_norm(int angle_pass = 1, float precision = 1, int power = 4);
	void combine_panorama(Axis axis, std::string output, bool resize = true);
	void concat_panorama(Axis axis, std::string output, bool resize = true);
	void concat_panorama(Map map, std::string output,  bool resize = true);
	void color_3d_model(std::string image_path, Axis axis, double threshold);
						
private:
	/** @brief Vertexs of the mesh */
	std::vector<glm::vec3> vertexs; 
	/** @brief Faces of the mesh */
	std::vector<std::vector<int>> facesIndex; 
	/** @brief Mesh faces divided according to height and sector to which they belong */
	std::vector<std::vector<std::vector<int>>> facesIndex_filter; 
	/** @brief Colors of vertexs */
	std::vector<std::vector<double>> vertexColor; 

	/** @brief PANORAMA representation of loaded mesh */
	std::vector<std::vector<float>> panorama; 
	/** @brief Extended PANORAMA representation of loaded mesh */
	std::vector<std::vector<float>> panorama_extended; 
	/** @brief Normals of mesh faces */
	std::vector<glm::vec3> normals; 
	/** @brief Centroid of the mesh */
	glm::vec3 centroid; 
	/** @brief Max distance from centroid to mesh */
	double d_max; 
	/** @brief Height of the cylinder */
	double height; 
	/** @brief Radius of the cylinder */
	double radius; 
	/** @brief Number of divisions of the cylinder height
	 * 
	 * Is set by default to 180 divisions
	 */
	int B = 1800;
	/**
	 * @brief SDM feature map
	 */
	std::vector<cv::Mat> sdm;
	/**
	 * @brief NDM feature map
	 */
	std::vector<cv::Mat> ndm;
	/**
	 * @brief GNDM feature map
	 */
	std::vector<cv::Mat> gndm;
	/**
	 * @brief Model's name
	 */
	std::string name;

	void calc_normals();
	void calc_centroid();
	void calc_distance();
	void scale_to_unit();

	bool load_obj(const std::string path, bool color);

	bool RayIntersectsTriangle(glm::vec3 rayOrigin, 
								glm::vec3 rayVector, 
								glm::vec3 vertex0,
								glm::vec3 vertex1,
								glm::vec3 vertex2,
								glm::vec3& outIntersectionPoint);

	glm::vec3 get_orig(Axis axis, float v, float precision);
	glm::vec3 get_dir(Axis axis, float angle);
	float get_height(Axis axis, float v, float precision);
	int get_sector(glm::vec2 point);
	int get_sector(float angle);

	void filter_faces(Axis axis, float precision);
	float feature_map(Map map, Axis axis, float precision, int power, glm::vec3 origin,
						glm::vec3 direction, std::vector<glm::vec3> &colisiones,
						std::vector<int> &faces_hit);
	void export_panorama(Map map, Axis axis, bool extended = true);
	std::vector<float> compute_panorama_symetry();
	float panorama_symetry_value();
	int panorama_symetry_column();
	float variance_of_panorama();
} ;


#endif
