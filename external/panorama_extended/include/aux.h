#ifndef AUX_H_INCLUDED
#define AUX_H_INCLUDED
#include <string>
#include <vector>
#include <sstream>


/** @file aux.h
 *  @brief Auxiliar enums definitions and functions related
 * 	@author [Alejandro Manzanares Lemus](https://github.com/Alexmnzlms)
 * 
 */

/**
 * @brief Represents the three axis of euclidean geometry.
 * 
 */
enum Axis {
    X = 0, /**< Axis X */
    Y = 1, /**< Axis Y */
    Z = 2  /**< Axis Z */
};

/**
 * @brief Represents the two possible feature maps PANORAMA is composed
 * 
 */
enum Map {
    SDM = 0, /**< Spatial Distribution Map */
    NDM = 1,  /**< Normal's Derivation Map */
    GNDM = 2 /**< Gradient Normal's Derivation Map */
};

std::string axis_to_string(Axis axis);
std::string map_to_string(Map map);
#endif