#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_PURE
#pragma once
#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"
#include <memory>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <device_launch_parameters.h>
#include "vector_types.h"
#include "vector_functions.h"
#include "helper_math.h"
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>
#include <vector>
#define PI (float)3.14159265358979323846264338327950288
#define kFloatEpsilon (float)1e-6
#define particle_interval 0.75f
#define _h 0.5f
#define mass 1.0f
#define rho_0_recpr 1.0f
#define epsilon 200.0f

#define sh_cell_size 1.5f
#define num_iterations 10
#define corr_delta_q_coeff 0.3
#define corr_k 0.001f
#define corr_n 4
#define vorticity_epsilon 1.0f
#define xsph_c 0.001f

struct tempSolver {
	std::vector<float4> positions;
	std::vector<float3> velocities;
	std::vector<int> phases;

	std::vector<float4> diffusePos;
	std::vector<float3> diffuseVelocities;

	std::vector<int> clothIndices;
	std::vector<float> restLengths;
	std::vector<float> stiffness;
	std::vector<int> triangles;
};

struct solverParams {
public:
	int maxNeighbors = 50;
	int maxParticles = 100;
	int maxContacts = 10;
	int gridWidth = 64, gridHeight = 64, gridDepth = 64;
	int gridSize = 64 * 64 * 64;

	int numParticles = 16384;
	int numDiffuse = 1024 * 2048;

	int numCloth = 0;
	int numConstraints = 0;

	float3 gravity = make_float3(0, -98.0f, 0);
	float3 bounds = make_float3(64.0f, 64.0f, 64.0f)*radius;

	int numIterations = 10;
	float radius = 0.05f;
	float restDistance = 0.05f;
	//float sor;
	//float vorticity;

	float KPOLY = 315.0f / (64.0f * PI * pow(radius, 9));
	float SPIKY = 45.0f / (PI * pow(radius, 6));;
	float restDensity = 6378.0f;
	float lambdaEps = 600.0f;
	float vorticityEps = 0.0001f;
	float C = 0.01f;
	float K = 0.00001f;
	float dqMag = 0.2f * radius;
	float wQH = KPOLY * pow((radius * radius - dqMag * dqMag), 3);
};
struct TsolverParams {
public:
	int maxNeighbors;
	int maxParticles;
	int maxContacts;
	int gridWidth, gridHeight, gridDepth;
	int gridSize;

	int numParticles;
	int numDiffuse;

	int numCloth;
	int numConstraints;

	float3 gravity;
	float3 bounds;

	int numIterations;
	float radius;
	float restDistance;
	//float sor;
	//float vorticity;

	float KPOLY;
	float SPIKY;;
	float restDensity;
	float lambdaEps;
	float vorticityEps;
	float C;
	float K;
	float dqMag;
	float wQH;
};
struct solver {
	float4* oldPos;
	float4* newPos;
	float3* velocities;
	int* phases;
	float* densities;

	float4* diffusePos;
	float3* diffuseVelocities;

	int* clothIndices;
	float* restLengths;
	float* stiffness;

	int* neighbors;
	int* numNeighbors;
	int* gridCells;
	int* gridCounters;
	int* contacts;
	int* numContacts;

	float3* deltaPs;

	float* buffer0;
};