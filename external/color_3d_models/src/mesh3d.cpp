#include "../include/mesh3d.h"

/** @file malla3d.cpp
 *  @brief Implementation of class Mesh3D
 * 	@author [Alejandro Manzanares Lemus](https://github.com/Alexmnzlms)
 * 
 */

/**
 * @brief Default constructor
 */
Mesh3D::Mesh3D(){
	for(int i = 0; i < 3; i++){
		cv::Mat aux;
		sdm.push_back(aux);
		ndm.push_back(aux);
		gndm.push_back(aux);
	}
	name = "no_name";
}

/**
 * @brief Constructor with two parameters
 * 
 * Constructor calls load_obj() function to load de mesh data and store model's name.
 * @param name string containing model's name
 * @param path string containing the relative path to the 3D model to be loaded
 * @param color TO DO
 * @see load_obj()
 */
Mesh3D::Mesh3D(const std::string name, const std::string path, bool color){
	for(int i = 0; i < 3; i++){
		cv::Mat aux;
		sdm.push_back(aux);
		ndm.push_back(aux);
		gndm.push_back(aux);
	}
	load_obj(path, color);
	this->name = name;
}

/**
 * @brief Copy constructor
 * 
 * @param mesh 3D model original data
 */
Mesh3D::Mesh3D(const Mesh3D& mesh){
	vertexs = mesh.vertexs;
	facesIndex = mesh.facesIndex;
	facesIndex_filter = mesh.facesIndex_filter;
	normals = mesh.normals;
	centroid = mesh.centroid;
	d_max = mesh.d_max;
	height = mesh.height;
	radius = mesh.radius;
	panorama = mesh.panorama;
	panorama_extended = mesh.panorama_extended;
	sdm = mesh.sdm;
	ndm = mesh.ndm;
	gndm = mesh.gndm;
	name = mesh.name;
}

/**
 * @brief Default destructor
 */
Mesh3D::~Mesh3D(){}

/**
 * @brief Get model's name
 * 
 * @return std::string Model's name
 */
std::string Mesh3D::get_name(){
	return name;
}

/**
 * @brief Set model's name
 * 
 * @param name Name for 3D model
 */
void Mesh3D::set_name(const std::string name){
	this->name = name;
}

/**
 * @brief Get number of 3D models' vertex
 * 
 * @return int number of 3D models' vertex
 */
int Mesh3D::num_vertexs(){
	return (int)vertexs.size();
}

/**
 * @brief Get number of 3D models' triangles (faces)
 * 
 * @return int number of 3D models' faces
 */
int Mesh3D::num_faces(){
	return (int)facesIndex.size();
}

/**
 * @brief Calculate centroid of mesh
 */
void Mesh3D::calc_centroid(){
	float sum_x, sum_y, sum_z;
	sum_x = sum_y = sum_z = 0.0;
	
	for(int i = 0; i < vertexs.size(); i++){
		sum_x += vertexs[i].x;
		sum_y += vertexs[i].y;
		sum_z += vertexs[i].z;
	}

	sum_x /= vertexs.size();
	sum_y /= vertexs.size();
	sum_z /= vertexs.size();

	centroid.x = sum_x;
	centroid.y = sum_y;
	centroid.z = sum_z;
}

/**
 * @brief Calculate normals of mesh faces
 */
void Mesh3D::calc_normals(){
	normals.clear();
	glm::vec3 v1, v2, v3;
	for(int i = 0; i < facesIndex.size(); i++){
		v1 = vertexs[facesIndex[i][0]];
		v2 = vertexs[facesIndex[i][1]];
		v3 = vertexs[facesIndex[i][2]];

		glm::vec3 cross_p = glm::cross((v2-v1), (v3-v1));

		cross_p = glm::normalize(cross_p);

		normals.push_back(cross_p);
	}
}

/**
 * @brief Calculate maximun distance from centroid
 */
void Mesh3D::calc_distance(){
	float dist;
	float max = glm::distance(vertexs[0],centroid);
	for(int i = 1; i < vertexs.size(); i++){
		dist = glm::distance(vertexs[i],centroid);
		if (dist > max){
			max = dist;
		}
	}

	d_max = max;
	radius = 2*d_max;
	height = 2*radius;
}

/**
 * @brief Scale the mesh to the unit sphere
 */
void Mesh3D::scale_to_unit(){
	std::vector<glm::vec3> v_unit;

	for(int i = 0; i < vertexs.size(); i++){
		glm::vec3 v_orig = vertexs[i] - centroid;
		glm::vec3 v_scale = v_orig;
		v_scale.x = v_scale.x / d_max;
		v_scale.y = v_scale.y / d_max;
		v_scale.z = v_scale.z / d_max;

		v_unit.push_back(v_scale);
	}

	vertexs = v_unit;
	centroid.x = 0.0;
	centroid.y = 0.0;
	centroid.z = 0.0;
	calc_normals();
	calc_distance();
}

/**
 * @brief Rotate the mesh
 * @param axis_rot Axis of rotation
 * @param angle Angles desired to rotate de mesh
 */
void Mesh3D::rotate_mesh(Axis axis_rot, float angle){
	std::vector<glm::vec3> mesh_rotate;
	glm::vec3 rotate_vertex;
	switch (axis_rot)
	{
	case X:
		for(int i = 0; i < vertexs.size(); i++){
			rotate_vertex.x = vertexs[i].x;
			rotate_vertex.y = vertexs[i].y*cos(angle) - vertexs[i].z*sin(angle);
			rotate_vertex.z = vertexs[i].y*sin(angle) + vertexs[i].z*cos(angle);

			mesh_rotate.push_back(rotate_vertex);
		}
		break;

	case Y:
		for(int i = 0; i < vertexs.size(); i++){
			rotate_vertex.x = vertexs[i].x*cos(angle) - vertexs[i].z*sin(angle);
			rotate_vertex.y = vertexs[i].y;
			rotate_vertex.z = vertexs[i].x*sin(angle) + vertexs[i].z*cos(angle);

			mesh_rotate.push_back(rotate_vertex);
		}
		break;

	case Z:
		for(int i = 0; i < vertexs.size(); i++){
			rotate_vertex.x = vertexs[i].x*cos(angle) - vertexs[i].y*sin(angle);
			rotate_vertex.y = vertexs[i].x*sin(angle) + vertexs[i].y*cos(angle);
			rotate_vertex.z = vertexs[i].z;

			mesh_rotate.push_back(rotate_vertex);
		}
		break;
	}
	
	vertexs.clear();
	vertexs = mesh_rotate;
	calc_normals();
	calc_distance();
}

/**
 * @brief Load data of the mesh from an OBJ Wavefront file
 * @param path string containing the relative path to the 3D model to be loaded
 * @return bool Indicates whether the read was successful (true) or unsuccessful (false)
 * @see [OBJ Wavefront specification](http://paulbourke.net/dataformats/obj/)
 */
bool Mesh3D::load_obj(const std::string path, bool color){
	std::vector< unsigned int > vertexIndices;
	std::vector< glm::vec3 > temp_vertices;
	std::ifstream file;
	std::string line;
	std::string header;
	std::string face_line;
	std::string face;
	std::string delimiter = "/";
	file.open(path);

	if( !file.is_open() ){
		std::cout << "Impossible to open the file !" << std::endl;
		return false;
	}

	while(std::getline(file,line)){
		file >> header;
		if(header == "v"){
			glm::vec3 vertex;
			file >> vertex.x;
			file >> vertex.y;
			file >> vertex.z;
			vertexs.push_back(vertex);
			if(color){
				vertexColor.push_back({0.0, 0.0, 0.0});
			}
		} else if (header == "f"){
			std::string vertex1, vertex2, vertex3;
			std::vector<int> vertexIndex;
			file >> face_line;
			vertex1 = face_line.substr(0, face_line.find(delimiter));
			vertexIndex.push_back(std::stoi(vertex1)-1);

			file >> face_line;
			vertex2 = face_line.substr(0, face_line.find(delimiter));
			vertexIndex.push_back(std::stoi(vertex2)-1);

			file >> face_line;
			vertex3 = face_line.substr(0, face_line.find(delimiter));
			vertexIndex.push_back(std::stoi(vertex3)-1);

			facesIndex.push_back(vertexIndex);
		}
	}

	calc_centroid();
	calc_normals();
	calc_distance();

	std::cout << "Number of vertexs: " << vertexs.size() << std::endl;
	std::cout << "Number of faces: " << facesIndex.size() << std::endl;

	scale_to_unit();

	return true;
}

/**
 * @brief Export mesh data to OBJ Wavefront file
 * @param path string containing the path where you will export the model to
 */
void Mesh3D::export_obj(const std::string path, bool color){
	std::ofstream file;
	file.open(path);

	for(int i = 0; i < vertexs.size(); i++){
		file << "v " << vertexs[i][0] << " " << vertexs[i][1] << " " << vertexs[i][2] << " ";
		if(color){
			double red = vertexColor[i][0];
			if(red > 1.0){
				red = 1.0;
			}
			double green = vertexColor[i][1];
			if(green > 1.0){
				green = 1.0;
			}
			double blue = vertexColor[i][2];
			if(blue > 1.0){
				blue = 1.0;
			}
			file << red << " " << green << " " << blue << std::endl;
		} else {
			file << std::endl;
		}
	}

	for(int j = 0; j < facesIndex.size(); j++){
		file << "f " << facesIndex[j][0]+1 << " " 
		<< facesIndex[j][1]+1 << " " << facesIndex[j][2]+1 
		<< std::endl;
	}
	
	file.close();
}

/**
 * @brief Compute Ray-Triangle Intersection using Möller-Trumbore algorithm
 * 
 * This function compute the Intersection of a ray represented by rayOrigin 
 * and rayVector and a triangle represented by vertex0, vertex1 and vertex2.
 * @note The faces of the 3D mesh are made up of triangles.
 * @note Triangle is represented by three separated params in order to archive better 
 * times of execution.
 * @param rayOrigin Origin of ray
 * @param rayVector Direction of ray
 * @param vertex0 Point of triangle
 * @param vertex1 Point of triangle
 * @param vertex2 Point of triangle
 * @param outIntersectionPoint Hit point of ray in triangle (face)
 * @return bool Represents whether the ray hits the triangle (true) or not (false)
 * @see [Möller–Trumbore intersection algorithm](https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm)
 * @see [Fast, Minimum Storage Ray/Triangle Intersection](https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf)
 * @cite Moller97
 */
bool Mesh3D::RayIntersectsTriangle(glm::vec3 rayOrigin, 
									glm::vec3 rayVector, 
									glm::vec3 vertex0,
									glm::vec3 vertex1,
									glm::vec3 vertex2,
									glm::vec3& outIntersectionPoint){
	const float EPSILON = 0.00000001;
	glm::vec3 edge1, edge2, h, s, q;
	float a,f,u,v;
	edge1 = vertex1 - vertex0;
	edge2 = vertex2 - vertex0;
	h = glm::cross(rayVector,edge2);
	a = glm::dot(edge1,h);

	if (a > -EPSILON && a < EPSILON){
		return false;
	}

	f = 1.0/a;
	s = rayOrigin - vertex0;
	u = f * glm::dot(s,h);

	if (u < 0.0 || u > 1.0){
		return false;
	}

	q = glm::cross(s,edge1);
	v = f * glm::dot(rayVector,q);

	if (v < 0.0 || u + v > 1.0){
		return false;
	}
		
	float t = f * glm::dot(edge2,q);

	if (t > EPSILON){
		outIntersectionPoint = rayOrigin + rayVector * t;
		return true;
	}
	else{
		return false;
	}
		
}

/**
 * @brief Get origin of ray based on axis and height
 * @param axis Axis of ray
 * @param v Height of ray
 * @param precision Precision of heights division
 * @see calculate_panorama()
 * @return vec3 Point of origin
 */
glm::vec3 Mesh3D::get_orig(Axis axis, float v, float precision){
	glm::vec3 orig;

	switch (axis)
	{
	case X:
		orig.x = centroid.x - height/2 + v * height/(B/precision);
		orig.y = centroid.y;
		orig.z = centroid.z;
		break;

	case Y:
		orig.x = centroid.x;
		orig.y = centroid.y - height/2 + v * height/(B/precision);
		orig.z = centroid.z;
		break;

	case Z:
		orig.x = centroid.x;
		orig.y = centroid.y;
		orig.z = centroid.z - height/2 + v * height/(B/precision);
		break;
	}

	return orig;
}

/**
 * @brief Get direction of ray based on axis and angle
 * @param axis Axis of ray
 * @param angle Angle of ray
 * @see calculate_panorama()
 * @return vec3 Vector of ray
 */
glm::vec3 Mesh3D::get_dir(Axis axis, float angle){
	glm::vec3 dir;
	
	switch (axis)
	{
	case X:
		dir.x = 0.0;
		dir.y = radius * cos(angle);
		dir.z = radius * sin(angle);
		break;

	case Y:
		dir.x = radius * cos(angle);
		dir.y = 0.0;
		dir.z = radius * sin(angle);
		break;

	case Z:
		dir.x = radius * cos(angle);
		dir.y = radius * sin(angle);
		dir.z = 0.0;
		break;
	}

	return glm::normalize(dir);
}

/**
 * @brief Get height of a division of the cylinder
 * @param axis Axis of cylinder
 * @param v Height division
 * @param precision Precision of heights division
 * @see calculate_panorama()
 * @return float Height of v division of the cylinder
 */
float Mesh3D::get_height(Axis axis, float v, float precision){
	float h;

	switch (axis)
	{
	case X:
		h = centroid.x - height/2 + v * height/(B/precision);
		break;

	case Y:
		h = centroid.y - height/2 + v * height/(B/precision);
		break;

	case Z:
		h = centroid.z - height/2 + v * height/(B/precision);
		break;
	}

	return h;
}

/**
 * @brief Get sector of a point
 * @param point Point coordinates
 * @return int Sector of point point
 */
int Mesh3D::get_sector(glm::vec2 point){
	int sector = -1;
	float x = point.x;
	float y = point.y;

	if(x >= 0 && y >= 0){
		sector = 1;
	} else if(x < 0 && y >= 0){
		sector = 2;
	} else if(x < 0 && y < 0){
		sector = 3;
	} else if(x >= 0 && y < 0){
		sector = 4;
	}

	return sector;
}

/**
 * @brief Get sector of a vector represented by angle
 * @param angle Angle of vector
 * @return int Sector of vector represented by angle
 */
int Mesh3D::get_sector(float angle){
	int sector = -1;
	float x = radius * cos(angle);
	float y = radius * sin(angle);

	if(x >= 0 && y >= 0){
		sector = 1;
	} else if(x < 0 && y >= 0){
		sector = 2;
	} else if(x < 0 && y < 0){
		sector = 3;
	} else if(x >= 0 && y < 0){
		sector = 4;
	}

	return sector;
}

/**
 * @brief Separates the faces of the mesh according to their height and sector
 * 
 * The aim of this method is to separate the faces of the mesh to avoid checking 
 * all the faces of the model when shooting the ray.
 * 
 * If the ray has a height V and its direction belongs to sector S, 
 * it can only hit the faces that are at height V (or close to it) 
 * and in the sector S.
 * 
 * In this way, the faces are filtered out to reduce the number of checks and 
 * to make the calculation of the ray-triangle collisions easier.
 * 
 * For the estimation of the height, a theshold is established, so that if 
 * at least one point of each face belongs to the range 
 * [v - threshold, v+threshold] then this face belongs to the division v.
 * 
 * @note It takes time to perform this process, but it is only performed once 
 * for each feature map, so the gain in execution time is remarkable
 * @param axis Axis of feature map
 * @param precision Precsion used in the PANORAMA computation
 * @see calculate_panorama()
 */
void Mesh3D::filter_faces(Axis axis, float precision){
	float v_height;

	float threshold = 0.5;

	float v_length;

	bool salir = false;

	float coord_axis;

	float angle;

	glm::vec2 point_sector;
	glm::vec2 point_angle;

	std::vector<int> facesIndex_height_temp;
	std::vector<int> facesIndex_ang_temp;
	std::vector<std::vector<int>> facesIndex_height;

	facesIndex_filter.clear();

	for(float v = 0; v < (B/precision); v++){

		v_height = get_height(axis,v,precision);

		v_length = height/(B/precision);

		facesIndex_height.clear();
		facesIndex_height_temp.clear();
		
		for(int i = 0; i < facesIndex.size(); i++){
			salir = false;
			for(int j = 0; j < facesIndex[i].size() && !salir; j++){
				switch (axis)
				{
				case X:
					coord_axis = vertexs[facesIndex[i][j]].x;
					break;

				case Y:
					coord_axis = vertexs[facesIndex[i][j]].y;
					break;

				case Z:
					coord_axis = vertexs[facesIndex[i][j]].z;
					break;  
				}
				
				if((coord_axis >= (v_height - threshold*v_length)) &&
					(coord_axis <= (v_height + threshold*v_length))){
					facesIndex_height_temp.push_back(i);
					salir = true;
				}
			}
		}

		for(int sector = 1; sector <= 4; sector++){
			facesIndex_ang_temp.clear();

			for(int i = 0; i < facesIndex_height_temp.size(); i++){
				salir = false;
				for(int j = 0; j < facesIndex[facesIndex_height_temp[i]].size() && !salir; j++){
					switch (axis)
					{
					case X:
						point_sector.x = vertexs[facesIndex[facesIndex_height_temp[i]][j]].y;
						point_sector.y = vertexs[facesIndex[facesIndex_height_temp[i]][j]].z;
						break;

					case Y:
						point_sector.x = vertexs[facesIndex[facesIndex_height_temp[i]][j]].x;
						point_sector.y = vertexs[facesIndex[facesIndex_height_temp[i]][j]].z;
						break;

					case Z:
						point_sector.x = vertexs[facesIndex[facesIndex_height_temp[i]][j]].x;
						point_sector.y = vertexs[facesIndex[facesIndex_height_temp[i]][j]].y;
						break;  
					}

					if(get_sector(point_sector) == sector){
						facesIndex_ang_temp.push_back(facesIndex_height_temp[i]);
						salir = true;
					}
				}
			}

			facesIndex_height.push_back(facesIndex_ang_temp);
		}

		facesIndex_filter.push_back(facesIndex_height);
	}
}

/**
 * @brief Compute the feature map specified \cite SFIKAS2018208
 * 
 * There are two posible feature map:
 * - Spatial Distribution Map (SDM)
 * - Normal's Derivation Map (NDM)
 * 
 * For a collision vector, it is calculated which of all intersections 
 * has a maximum distance to the origin (since all rays emerge from the 
 * origin) in order to establish the last collision in case the ray crosses 
 * the mesh in several places.
 * 
 * Once the farthest collision is obtained, the feature maps are calculated.
 * - For the SDM the value is simply the distance from the origin to the 
 * intersection.
 * - For the NDM, the calculated value is the absolute value raised to a 
 * number of the angle between the ray that caused the collision and the 
 * normal vector to the intercepted face.


 * @param map Feature map
 * @param axis Axis of feature map
 * @param precision Precsion used in the PANORAMA computation
 * @param power Number to raise absolute value in NDM computation
 * @param origin Origin of ray
 * @param direction Direction of ray
 * @param colisiones Vector of ray colision with mesh
 * @param faces_hit Vector of faces intercepted by ray
 * @see @link Description @endlink 
 * @see calculate_panorama()
 * @return float Value of feature map
 */
float Mesh3D::feature_map(Map map, Axis axis, float precision, int power, glm::vec3 origin, 
							glm::vec3 direction, std::vector<glm::vec3> &colisiones, 
							std::vector<int> &faces_hit){

	float feature_value = 0.0;

	float dist;
	float dist_max = -1;
	float ind_max = -1;

	glm::vec3 ray;
	glm::vec3 normal;

	for(int c = 0; c < colisiones.size(); c++){ 
		dist = glm::distance(colisiones[c],origin); 
		if((dist > dist_max) && (dist > 0)){
			ind_max = c;
			dist_max = dist;
		}
	}

	assert(ind_max != -1);

	switch (map)
	{
	case SDM:
		assert(dist_max >= 0);
		feature_value = dist_max;
		
		break;

	case NDM:
		ray = origin + direction;
		ray = glm::normalize(ray);
		normal = normals[faces_hit[ind_max]];
		feature_value = glm::angle(ray,normal);
		feature_value = cos(feature_value);
		feature_value = std::abs(feature_value);
		feature_value = std::pow(feature_value,power);
		break;
	}
	

	assert(feature_value >= 0 && feature_value <= 1.0);
	return feature_value;
}

/**
 * @brief Calculate PANORAMA [extended] representation of the 3D mesh
 * 
 * The calculation of the PANORAMA [extended] representation is 
 * based on the projection of the 3D model on a cylinder. 
 * This cylinder has a radius of 2 times the furthest distance from 
 * the mesh centroid to the mesh and a height twice the radius.
 * 
 * Once the cylinder has been defined, based on the axis (X, Y or Z) chosen, 
 * the cylinder shall be divided into 180 sections (along the longitudinal axis). 
 * For each section a ray will be fired in 360 degrees around the origin and 
 * it will be calculated if the ray collides with the mesh. Once the collisions 
 * are calculated the feature maps are calculated based on what is specified in 
 * @link feature_map() @endlink.
 * 
 * @note A precision value is included. 
 * The number of divisions of the cylinder height is divided by the accuracy, 
 * so that if it is less than 0, the accuracy increases 
 * (0.5 means double the accuracy) and if it is greater, 
 * the accuracy decreases. This also applies to the accuracy of the ray angle.
 * 
 * The representation is extended, because the calculated 
 * representation is extended by a factor of 1.5. 
 * The first half of the representation is added at the end 
 * so that the first third of the PANORAMA and the last third are identical.
 * 
 * The final result is a PANORAMA representation of 
 * 540x180 pixels (depending on accuracy).
 * 
 * @param map Feature map
 * @param axis Axis of feature map
 * @param precision Precsion used in the PANORAMA computation
 * @param power Number to raise absolute value in NDM computation
 * @see feature_map()
 * @see Ensemble of PANORAMA-based convolutional neural networks for 3D model 
 * classification and retrieval @cite SFIKAS2018208
 */
void Mesh3D::calculate_panorama(Map map, Axis axis, float precision, int power){
	if(vertexs.size() > 0){
		std::chrono::steady_clock::time_point begin;
		std::chrono::steady_clock::time_point end;
		begin = std::chrono::steady_clock::now();

		panorama.clear();
		panorama_extended.clear();
		
		glm::vec3 direction;
		glm::vec3 origin;

		int n_colisiones = 0;

		std::vector<glm::vec3> colisiones;
		std::vector<int> face_hit;

		filter_faces(axis, precision);

		end = std::chrono::steady_clock::now();
		std::cout << "Face filtering"
		<< "\t Time: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;

		begin = std::chrono::steady_clock::now();

		for(float i = 0; i < (B/precision); i++){
			std::vector<float> row(2*(B/precision), 0.0);
			std::vector<float> row_ext(2*(B/precision)*1.5, 0.0);
			panorama.push_back(row);
			panorama_extended.push_back(row_ext);
		}

		for(float v = 0; v < (B/precision); v++){

			for(float u = 0; u < 2*(B/precision); u++){
				float angle = u*2*M_PI / (2*(B/precision));
				int s = get_sector(angle) - 1;

				origin = get_orig(axis,v,precision);
				direction = get_dir(axis,angle);

				for(int j = 0; j < facesIndex_filter[v][s].size(); j++){
					glm::vec3 hit_point;
					glm::vec2 hit;
					float dist;

					glm::vec3 t1 = vertexs[facesIndex[facesIndex_filter[v][s][j]][0]];
					glm::vec3 t2 = vertexs[facesIndex[facesIndex_filter[v][s][j]][1]];
					glm::vec3 t3 = vertexs[facesIndex[facesIndex_filter[v][s][j]][2]];

					if(RayIntersectsTriangle(origin, direction, t1, t2, t3, hit_point)){
						n_colisiones++;
						colisiones.push_back(hit_point);
						face_hit.push_back(facesIndex_filter[v][s][j]);
					} 
				}
				
				if(n_colisiones > 0){
					panorama[v][u] = feature_map(map, axis, precision, power, origin, direction, colisiones, face_hit);
				}
				
				n_colisiones = 0;
				colisiones.clear();
				face_hit.clear();
			}
			
		}

		for(int i = 0; i < panorama.size(); i++){
			for(int j = 0; j < panorama[i].size(); j++){
				panorama[i][j] = (panorama[i][j] * 255);
				assert(panorama[i][j] <= 255.0 && panorama[i][j] >= 0.0);
			}
		}

		int max = 2*(B/precision);
		for(int i = 0; i < panorama_extended.size(); i++){
			for(int j = 0; j < panorama_extended[i].size(); j++){
				panorama_extended[i][j] = panorama[i][j%max];
			}
		}

		end = std::chrono::steady_clock::now();
		std::cout << "Map: " <<  map_to_string(map)
		<< "\t Time: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;

		export_panorama(map,axis,"");
	}
}

/**
 * @brief Export PANORAMA representation to @link sdm @endlink , @link ndm @endlink or @link gndm @endlink
 * 
 * Exportation also computes GNDM (Gradient NDM). 
 * The Gradient is compute using Laplacian function provided by OpenCV. 
 * 
 * OpenCV functions are used to export the PANORAMA representation.
 * 
 * @param map Feature map
 * @param axis Axis of feature map
 * @param extended Exportation of extended representation or regular PANORAMA
 * 
 * @see [cv::imwrite()](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce)
 * @see [cv::Laplacian()](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gad78703e4c8fe703d479c1860d76429e6)
 */
void Mesh3D::export_panorama(Map map, Axis axis, bool extended){
	cv::Mat panorama_cv;

	if(extended){
		panorama_cv = cv::Mat(0, panorama_extended[0].size(), CV_32F);

		for(int i = panorama_extended.size()-1; i >= 0; i--){
			cv::Mat row(1, panorama_extended[i].size(), CV_32F, panorama_extended[i].data());
			panorama_cv.push_back(row);
		}
	} else {
		panorama_cv = cv::Mat(0, panorama[0].size(), CV_32F);

		for(int i = panorama.size()-1; i >= 0; i--){
			cv::Mat row(1, panorama[i].size(), CV_32F, panorama[i].data());
			panorama_cv.push_back(row);
		}
	}

	if(map == NDM){
		cv::Mat laplace;
		cv::Laplacian(panorama_cv, laplace, CV_32F, 1);
		gndm[(int)axis] = laplace;
		ndm[(int)axis] = panorama_cv;
	} else {
		sdm[(int)axis] = panorama_cv;
	}
}

/**
 * @brief Computes symmetry values for every comlumn w of PANORAMA representation
 * 
 * Symetry of a image column is defined as following \cite sfikas2014pose :
 * 
 * - \f$ Sym(w) = 1 - \frac{1}{2m} \sum_{h=\frac{height}{2}-m}^{h=\frac{height}{2}-+m} SymDiff(w,h) \f$
 * - \f$ SymDiff(w,h) = \frac{1}{n} \sum_{x=1}^n |(w-x,h) - (w+x,h)| \f$
 * 
 * @return std::vector<float> Vector of symmetry values for every comlumn w
 */
std::vector<float> Mesh3D::compute_panorama_symetry(){
	float height_sym = panorama.size();
	float width_sym = panorama[0].size();
	float m = 0.4 * height_sym;
	float n = 0.1 * width_sym;

	float sym = 0;
	float sym_diff = 0;
	int index_up, index_down;
	float up, down;

	std::vector<float> sumas;
	
	for(int w = 0; w < width_sym; w++){
		sym = 0;
		for(int h = (height_sym/2 - m); h <= (height_sym/2 + m); h++){
			sym_diff = 0;
			for(int x = 1; x <= n; x++){
				index_up = w+x;
				index_down = w-x;
				if(index_down < 0){
					index_down = (int)(width_sym + index_down);
				}
				if(index_up > (int)width_sym){
					index_up = index_up % (int)width_sym;
				}

				up = panorama[h][index_up] / 255.0;
				down = panorama[h][index_down] / 255.0;
				sym_diff += abs(down - up);
			}
			sym_diff = sym_diff / n;
			sym += sym_diff;
		}
		
		sym = sym / (2*m);
		sym = 1.0 - sym;
		sumas.push_back(sym);
	}

	return sumas;
}

/**
 * @brief Return max value of symmetry of PANORAMA representation
 * 
 * Symetry of a image is defined as following \cite sfikas2014pose :
 * 
 * - \f$ S(I) = max\{Sym(w) | w \in 1:width\} \f$
 * 
 * @return float max value of symmetry
 */
float Mesh3D::panorama_symetry_value(){
	
	std::vector<float> sumas = compute_panorama_symetry();

	float max = -1000000.0;
	for(int i = 0; i < sumas.size(); i++){
		if(max < sumas[i]){
			max = sumas[i];
		}
	}
	return max;
}

/**
 * @brief Return w column where symmetry value of PANORAMA representation is maximun
 * 
 * This is defined as following \cite sfikas2014pose :
 * 
 * - \f$ S_{index}(I) = argmax\{ Sym(w) | w \in 1:width\} \f$
 * 
 * @return int w column where symmetry value is maximun
 */
int Mesh3D::panorama_symetry_column(){
	
	std::vector<float> sumas = compute_panorama_symetry();

	float max = -1000000.0;
	int pos_max = -1;
	for(int i = 0; i < sumas.size(); i++){
		if(max < sumas[i]){
			max = sumas[i];
			pos_max = i;
		}
	}
	return pos_max;
}

/**
 * @brief Compute variance of PANORAMA representation
 * 
 * Variance of an image is defined as following:
 * 
 * - \f$ Var(I) = \sum_{}^{} (I(i,j) - \mu )^2 \f$ where \f$ \mu \f$ is the mean value of the image.
 * 
 * @see [Variance](https://en.wikipedia.org/wiki/Variance)
 * @return float variance of PANORAMA representation
 */
float Mesh3D::variance_of_panorama(){
	float sum = 0.0;
	float sq_diff = 0.0;

	float size = (panorama.size()*panorama[0].size());

	for(int i = 0; i < panorama.size(); i++){
		for(int j = 0; j < panorama[0].size(); j++){
			sum += panorama[i][j];
		}
	}

	float mean = sum / size;

	for(int i = 0; i < panorama.size(); i++){
		for(int j = 0; j < panorama[0].size(); j++){
			sq_diff += ((panorama[i][j] - mean) * (panorama[i][j] - mean));
		}
	}

	sq_diff = sq_diff / size;

	return sq_diff;
}

/**
 * @brief Compute the normalized pose of 3d mesh
 * 
 * This algorithm is an adaptation of the one presented in \cite sfikas2014pose
 * 
 * @param angle_pass Increment of angle to normalize pose
 * @param precision Precsion used in the PANORAMA computation
 * @param power Number to raise absolute value in NDM computation
 */
void Mesh3D::mesh_pose_norm(int angle_pass, float precision, int power){
	float best_sym = 0;
	float best_rot = 0;
	float min_var = 1000000000.0;
	int pos_sym = 0;
	float pos_var = 0;
	Axis cyl_axis = Z;
	Axis rot_axis = X;
	float angle = angle_pass * (M_PI/180.0);
	int ind = 0;

	std::vector<glm::vec3> vertexs_aux(vertexs);
	for(float i = 0; i <= 180; i+=angle_pass, ind++){
		rotate_mesh(rot_axis,angle);
		calculate_panorama(NDM,cyl_axis,precision,power);
		// export_panorama(NDM,cyl_axis,false);
		// cv::imwrite("NDM_cly_axis.png",ndm[(int)cyl_axis]);
		float sym = panorama_symetry_value();
		std::cout << "Angle: " << ind*angle*(180.0/M_PI) << std::endl;
		std::cout << "Symmetry: " << sym << std::endl;
		if(sym >= best_sym){
			best_rot = ind*angle;
			best_sym = sym;
			pos_sym = panorama_symetry_column();
		}
	}

	std::cout << "Best rotation angle: " << best_rot*(180.0/M_PI) << std::endl;

	vertexs = vertexs_aux;
	rotate_mesh(rot_axis,best_rot);
	rotate_mesh(Y,-M_PI/2); // TRANSFORM BASIS
	vertexs_aux = vertexs;

	ind = 0;
	for(float i = 0; i <= 180; i+=angle_pass, ind++){
		rotate_mesh(cyl_axis,angle);
		calculate_panorama(SDM,rot_axis,precision,power);
		// export_panorama(SDM,rot_axis,false);
		// cv::imwrite("SDM_rot_axis.png",sdm[(int)rot_axis]);
		float var = variance_of_panorama();
		std::cout << "Angle: " << ind*angle*(180.0/M_PI) << std::endl;
		std::cout << "Variance: " << var << std::endl;
		if(var <= min_var){
			min_var = var;
			pos_var = ind*angle;
		}
	}

	vertexs = vertexs_aux;
	rotate_mesh(cyl_axis,pos_var);
	calc_normals();

	std::cout << "Best position angle: " << pos_var*(180.0/M_PI) << std::endl;
}

/**
 * @brief Combine PANORAMA feature maps SDM, NDM and GNDM in PANORAMA extended
 * representation
 * 
 * SDM, NDM and GNDM are combine in to a single image. One feature map 
 * per color channel.
 * 
 * PANORAMA extended output image is exported as original size or 10% of the
 * original size. This reduction is performed by using resize function provided
 * by OpenCV.
 * 
 * @param axis Axis of feature maps
 * @param output Relative path to image export folder
 * @param resize Indicates whether the image is exported in original size or 10%
 * 
 * @see [cv::resize](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d)
 */
void Mesh3D::combine_panorama(Axis axis, std::string output, bool resize){
	if(vertexs.size() > 0){
		std::string extension = ".png";

		std::vector<cv::Mat> channels;

		cv::Mat panorama_extended, panorama_resize;

		channels.push_back(gndm[(int)axis]);
		channels.push_back(sdm[(int)axis]);
		channels.push_back(ndm[(int)axis]);

		cv::merge(channels, panorama_extended);

		int r = (int) (panorama_extended.rows * 0.1);
		int c = (int) (panorama_extended.cols * 0.1);

		if(resize){
			cv::resize(panorama_extended, panorama_resize, cv::Size(c,r), CV_INTER_CUBIC);
			cv::imwrite(output + name + "_panorama_ext_" + axis_to_string(axis) + extension , panorama_resize);
		} else {
			cv::imwrite(output + name + "_panorama_ext_" + axis_to_string(axis) + extension , panorama_extended);
		}
	}
}

/**
 * @brief Concatenation PANORAMA feature maps SDM and NDM
 * 
 * vconcat function is used to compute this concatenation.
 *
 * PANORAMA output image is exported as original size or 10% of the
 * original size. This reduction is performed by using resize function provided
 * by OpenCV.
 * 
 * @param axis Axis of feature maps
 * @param output Relative path to image export folder
 * @param resize Indicates whether the image is exported in original size or 10%
 * 
 * @see [cv::resize](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d)
 * @see [cv::vconcat](https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga744f53b69f6e4f12156cdde4e76aed27)
 */
void Mesh3D::concat_panorama(Axis axis, std::string output, bool resize){
	if(vertexs.size() > 0){
		std::string extension = ".png";

		std::vector<cv::Mat> channels;

		cv::Mat panorama, panorama_resize;

		cv::Mat feature_array[3] = {sdm[(int)axis],ndm[(int)axis],gndm[(int)axis]};

		cv::vconcat(feature_array,3,panorama);

		int r = (int) (panorama.rows * 0.1);
		int c = (int) (panorama.cols * 0.1);
		
		if(resize) {
			cv::resize(panorama, panorama_resize, cv::Size(c,r), CV_INTER_CUBIC);
			cv::imwrite(output + name + "_panorama_" + axis_to_string(axis) + extension , panorama_resize);
		} else {
			cv::imwrite(output + name + "_panorama_" + axis_to_string(axis) + extension , panorama);
		}
	}
}

/**
 * @brief Concatenation PANORAMA feature maps SDM and NDM
 * 
 * vconcat function is used to compute this concatenation.
 *
 * PANORAMA output image is exported as original size or 10% of the
 * original size. This reduction is performed by using resize function provided
 * by OpenCV.
 * 
 * @param map Axis of feature maps
 * @param output Relative path to image export folder
 * @param resize Indicates whether the image is exported in original size or 10%
 * 
 * @see [cv::resize](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d)
 * @see [cv::vconcat](https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga744f53b69f6e4f12156cdde4e76aed27)
 */
void Mesh3D::concat_panorama(Map map, std::string output, bool resize){
	std::string extension = ".png";

	std::vector<cv::Mat> channels;

	cv::Mat map_x, map_y, map_z, panorama, panorama_resize;

	switch (map)
	{
	case SDM:
		map_x = sdm[0];
		map_y = sdm[1];
		map_z = sdm[2];
		break;
	
	case NDM:
		map_x = ndm[0];
		map_y = ndm[1];
		map_z = ndm[2];
		break;

	case GNDM:
		map_x = gndm[0];
		map_y = gndm[1];
		map_z = gndm[2];
		break;
	}

	cv::Mat feature_array[3] = {map_x, map_y, map_z};

	cv::vconcat(feature_array,3,panorama);

	int r = (int) (panorama.rows * 0.1);
	int c = (int) (panorama.cols * 0.1);
	
	if(resize){
		cv::resize(panorama, panorama_resize, cv::Size(c,r), CV_INTER_CUBIC);
		cv::imwrite(output + name + "_panorama_" + map_to_string(map) + extension , panorama_resize);
	} else {
		cv::imwrite(output + name + "_panorama_" + map_to_string(map) + extension , panorama);
	}
	
}
/** @brief
 * 
 * 
*/
void Mesh3D::color_3d_model(std::string image_path, Axis axis, double threshold){
	cv::Mat img_orig = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
	// cv::Mat img = img_orig.colRange(0,72);
	cv::Mat img = img_orig;
	// cv::imshow("Window",img);
	// cv::waitKey();
	// std::vector<cv::Mat> three_channels;
	// cv::split(img,three_channels);

	cv::Mat img_resize;
	cv::resize(img,img_resize,cv::Size(2*B,B));
	// cv::imshow("Window",img_resize);
	// cv::waitKey();
	// cv::destroyAllWindows();

	// Now I can access each channel separately
	// for(int i=0; i<img.rows; i++){
	// 	for(int j=0; j<img.cols; j++){
	// 		int red = static_cast<int>(three_channels[0].at<uint8_t>(i,j));
	// 		int green = static_cast<int>(three_channels[1].at<uint8_t>(i,j));
	// 		int blue = static_cast<int>(three_channels[2].at<uint8_t>(i,j));
	// 		// std::cout << red << " " << green << " " << blue << std::endl;
	// 	}	
		// std::cout << std::endl;
	// }

	// std::cout << three_channels[0] << std::endl;

	if(vertexs.size() > 0){

		std::chrono::steady_clock::time_point begin;
		std::chrono::steady_clock::time_point end;
		begin = std::chrono::steady_clock::now();
		
		glm::vec3 direction;
		glm::vec3 origin;

		int n_colisiones = 0;

		std::vector<glm::vec3> colisiones;
		std::vector<int> face_hit;

		filter_faces(axis, 1);

		end = std::chrono::steady_clock::now();
		// std::cout << "Face filtering"
		// << "\t Time: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;

		begin = std::chrono::steady_clock::now();

		for(float v = 0; v < B; v++){
			// std::cout << B-v << std::endl;

			for(float u = 0; u < 2*B; u++){

				// int red = static_cast<int>(three_channels[2].at<uint8_t>(v,u));
				// int green = static_cast<int>(three_channels[1].at<uint8_t>(v,u));
				// int blue = static_cast<int>(three_channels[1].at<uint8_t>(v,u));
				int gray = static_cast<int>(img_resize.at<uint8_t>(v,u));
				// red = gray;
				// blue = gray;
				// green = gray;
				// std::cout << u << " " << v << " " << red << " " << green << " " << blue << std::endl;
				float angle = u*2*M_PI / (2*B);
				// std::cout << angle << std::endl;
				int s = get_sector(angle) - 1;

				origin = get_orig(axis,v,1);
				direction = get_dir(axis,angle);
				// std::cout << "Origin: " << origin[0] << " " << origin[1] << " " << origin[2] << std::endl;
				// std::cout << "Direction: " << direction[0] << " " << direction[1] << " " << direction[2] << std::endl;

				if(gray != 0){
					for(int j = 0; j < facesIndex_filter[v][s].size(); j++){
						glm::vec3 hit_point;
						glm::vec2 hit;
						float dist;

						// std::cout << facesIndex[j][0] << " " << facesIndex[j][1] << " " << facesIndex[j][2] << std::endl;

						glm::vec3 t1 = vertexs[facesIndex[facesIndex_filter[v][s][j]][0]];
						glm::vec3 t2 = vertexs[facesIndex[facesIndex_filter[v][s][j]][1]];
						glm::vec3 t3 = vertexs[facesIndex[facesIndex_filter[v][s][j]][2]];

						if(RayIntersectsTriangle(origin, direction, t1, t2, t3, hit_point)){
							n_colisiones++;
							colisiones.push_back(hit_point);
							face_hit.push_back(facesIndex_filter[v][s][j]);
						} 
					}
				
					if(n_colisiones > 0){
						// panorama[v][u] = feature_map(map, axis, precision, power, origin, direction, colisiones, face_hit);
						float dist;
						float dist_max = -1;
						float ind_max = -1;

						for(int c = 0; c < colisiones.size(); c++){ 
							dist = glm::distance(colisiones[c],origin); 
							if((dist > dist_max) && (dist > 0)){
								ind_max = c;
								dist_max = dist;
							}
						}

						// std::cout << "COLISION " << face_hit[ind_max] << " " << colisiones[ind_max].x << " " << colisiones[ind_max].y << " " << colisiones[ind_max].y << std::endl;
						// std::cout << ind_max << std::endl;
						// std::cout << face_hit.size() << std::endl;
						// std::cout << face_hit[ind_max] << std::endl;
						auto vertex_index = facesIndex[face_hit[ind_max]];
						// std::cout << vertex_index[0] << " " << vertex_index[1] << " " << vertex_index[2] << std::endl;
						// std::cout << "gray value: " << gray << std::endl;
						// double true_color = (gray - 127.5) / 127.5;
						double true_color = gray / 255.0;
						// std::cout << "true_color: " << true_color << std::endl;
						// double red_value = 0.25;
						// double green_value = 0.25;
						// double blue_value = 0.25;
						double red_value = 0;
						double green_value = 0;
						double blue_value = 0;

						// if (true_color < -0.05){
						// 	blue_value = 1.0; //* true_color;
						// } else if (true_color > 0.05) {
						// 	red_value = 1.0;// * true_color;
						// }
						
						// switch (axis)
						// {
						// case X:
						// 	red_value = 0.75 * true_color + 0.25;
						// 	break;

						// case Y:
						// 	green_value = 0.75 * true_color + 0.25;
						// 	break;

						// case Z:
						// 	blue_value = 0.75 * true_color + 0.25;
						// 	break;
						// }

						if (true_color >= threshold){
							red_value = true_color;
							// red_value = 0.75 * true_color + 0.25;
						}

						vertexColor[vertex_index[0]][0] = red_value;
						vertexColor[vertex_index[0]][1] = green_value;
						vertexColor[vertex_index[0]][2] = blue_value;
						vertexColor[vertex_index[1]][0] = red_value;
						vertexColor[vertex_index[1]][1] = green_value;
						vertexColor[vertex_index[1]][2] = blue_value;
						vertexColor[vertex_index[2]][0] = red_value;
						vertexColor[vertex_index[2]][1] = green_value;
						vertexColor[vertex_index[2]][2] = blue_value;


					} else {
						// std::cout << "NO COLISION" << std::endl;
						// if(red != 0 || blue != 0 || green != 0){
						// 	std::cout << "Origen: " << origin.x << " " << origin.y << " " << origin.z << std::endl;
						// 	std::cout << "Direction: " << direction.x << " " << direction.y << " " << direction.z << std::endl;
						// 	// exit(1);
						// }
						
					}
					
					n_colisiones = 0;
					colisiones.clear();
					face_hit.clear();
				}
			}
		}	
	}

	// cv::imshow("Windows", three_channels[1]);
	// cv::waitKey();
}


