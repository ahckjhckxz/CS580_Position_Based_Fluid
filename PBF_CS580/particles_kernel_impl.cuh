/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_
#include <stdio.h>
#include <math.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"
#include "Sover.h"
#if USE_TEX
// textures for particle position and velocity
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldVelTex;

texture<uint, 1, cudaReadModeElementType> gridParticleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;
#endif

// simulation parameters in constant memory
__constant__ SimParams params;
#define CUDA_CALLABLE __host__ __device__
#define PI_FLT (float)3.14159265358979323846264338327950288
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

CUDA_CALLABLE constexpr float kPoly6Factor() {
	return (315.0f / 64.0f / PI_FLT);
}
CUDA_CALLABLE inline float Poly6Value(const float s, const float h) {
	if (s < 0.0f || s >= h)
		return 0.0f;

	float x = (h * h - s * s) / (h * h * h);
	float result = kPoly6Factor() * x * x * x;
	return result;
}
CUDA_CALLABLE inline float Poly6Value(const float3 r, const float h) {
	float r_len = length(r);
	return Poly6Value(r_len, h);
}
__device__ float ComputeScorr(const float3 pos_ji, const float h,
	const float corr_delta_q_coeff_,
	const float corr_k_, const float corr_n_) {
	// Eq (13)
	float x = Poly6Value(pos_ji, h) / Poly6Value(corr_delta_q_coeff_ * h, h);
	float result = (-corr_k_) * pow(x, corr_n_);
	return result;
}
CUDA_CALLABLE constexpr float kSpikyGradFactor() { return (-45.0f / PI_FLT); }

CUDA_CALLABLE inline float3 SpikyGradient(const float3 r, const float h) {
	float r_len = length(r);
	if (r_len <= 0.0f || r_len >= h)
		return make_float3(0.0f);

	float x = (h - r_len) / (h * h * h);
	float g_factor = kSpikyGradFactor() * x * x;
	float3 result = normalize(r) * g_factor;
	return result;
}
struct integrate_functor
{
    float deltaTime;

    __host__ __device__
    integrate_functor(float delta_time) : deltaTime(delta_time) {}

    template <typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        volatile float4 posData = thrust::get<0>(t);
        volatile float4 velData = thrust::get<1>(t);
        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);

        vel += params.gravity * deltaTime;
        //vel *= params.globalDamping;

        // new position = old position + velocity * deltaTime
        pos += vel * deltaTime;

        // set this to zero to disable collisions with cube sides

        if (pos.x > 1.0f - params.particleRadius)
        {
            pos.x = 1.0f - params.particleRadius;
            vel.x = 0;
        }

        if (pos.x < -1.0f + params.particleRadius)
        {
            pos.x = -1.0f + params.particleRadius;
            vel.x = 0;
        }

        if (pos.y > 1.0f - params.particleRadius)
        {
            pos.y = 1.0f - params.particleRadius;
            vel.y = -vel.y;
        }

        if (pos.z > 1.0f - params.particleRadius)
        {
            pos.z = 1.0f - params.particleRadius;
            vel.z = 0;
        }

        if (pos.z < -1.0f + params.particleRadius)
        {
            pos.z = -1.0f + params.particleRadius;
            vel.z = 0;
        }

        if (pos.y < -1.0f + params.particleRadius)
        {
            pos.y = -1.0f + params.particleRadius;
            vel.y = 0;
        }

        // store new position and velocity
        thrust::get<0>(t) = make_float4(pos, posData.w);
        thrust::get<1>(t) = make_float4(vel, velData.w);
    }
};

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y-1);
    gridPos.z = gridPos.z & (params.gridSize.z-1);
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint   *gridParticleHash,  // output
               uint   *gridParticleIndex, // output
               float4 *pos,               // input: positions
               uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint   *cellStart,        // output: cell start index
                                  uint   *cellEnd,          // output: cell end index
                                  float4 *sortedPos,        // output: sorted positions
                                  float4 *sortedVel,        // output: sorted velocities
                                  uint   *gridParticleHash, // input: sorted grid hashes
                                  uint   *gridParticleIndex,// input: sorted particle indices
                                  float4 *oldPos,           // input: sorted position array
                                  float4 *oldVel,           // input: sorted velocity array
                                  uint    numParticles)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index-1];
        }
    }

    cg::sync(cta);

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos and vel data
        uint sortedIndex = gridParticleIndex[index];
        float4 pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch
        float4 vel = FETCH(oldVel, sortedIndex);       // see particles_kernel.cuh

        sortedPos[index] = pos;
        sortedVel[index] = vel;
    }


}

// collide two spheres using DEM method
__device__
float3 collideSpheres(float3 posA, float3 posB,
                      float3 velA, float3 velB,
                      float radiusA, float radiusB,
                      float attraction)
{
    // calculate relative position
    float3 relPos = posB - posA;

    float dist = length(relPos);
    float collideDist = radiusA + radiusB;

    float3 force = make_float3(0.0f);

    if (dist < collideDist)
    {
        float3 norm = relPos / dist;

        // relative velocity
        float3 relVel = velB - velA;

        // relative tangential velocity
        float3 tanVel = relVel - (dot(relVel, norm) * norm);

        // spring force
        force = -params.spring*(collideDist - dist) * norm;
        // dashpot (damping) force
        force += params.damping*relVel;
        // tangential shear force
        force += params.shear*tanVel;
        // attraction
        force += attraction*relPos;
    }

    return force;
}



// collide a particle against all other particles in a given cell
__device__
void CalLamda(int3    gridPos,
                   uint    index,
                   float3  pos,
                   float4 *oldPos,
                   uint   *cellStart,
                   uint   *cellEnd,
				   float  *gradient,
				   float  *density,
				   float3 *gradient_i)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH(cellStart, gridHash);

    float3 force = make_float3(0.0f);

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = FETCH(cellEnd, gridHash);

        for (uint j=startIndex; j<endIndex; j++)
        {
            if (j != index)                // check not colliding with self
            {
                float3 pos2 = make_float3(FETCH(oldPos, j));
				const float3 pos_ji = pos - pos2;
				const float3 gradient_j = SpikyGradient(pos_ji, _h);
				*gradient += dot(gradient_j, gradient_j);
				*gradient_i += gradient_j;
				*density += mass * Poly6Value(pos_ji, _h);
                // collide two spheres
                //force += collideSpheres(pos, pos2, vel, vel2, params.particleRadius, params.particleRadius, params.attraction);
            }
        }
    }
}
__device__
void CalPos(int3    gridPos,
	uint    index,
	float3  pos,
	float4 *oldPos,
	uint   *cellStart,
	uint   *cellEnd,
	float3 *delta_pos_i,
	float *lamdas)
{
	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = FETCH(cellStart, gridHash);

	float3 force = make_float3(0.0f);
	const float lambda_i = lamdas[index];
	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index)                // check not colliding with self
			{
				float3 pos2 = make_float3(FETCH(oldPos, j));
				const float lambda_j = lamdas[j];
				const float3 pos_ji = pos - pos2;
				const float scorr_ij =
					ComputeScorr(pos_ji, _h, corr_delta_q_coeff, corr_k, corr_n);
				*delta_pos_i += (lambda_i + lambda_j + scorr_ij) * SpikyGradient(pos_ji, _h);
			}
		}
	}
}
__device__ void boundary(float3 &pos)
{
	if (pos.x > 1.0f)
	{
		pos.x = 1.0f;
	}

	if (pos.x < -1.0f)
	{
		pos.x = -1.0f;
	}

	if (pos.y > 1.0f)
	{
		pos.y = 1.0f;
	}

	if (pos.z > 1.0f)
	{
		pos.z = 1.0f;
	}

	if (pos.z < -1.0f)
	{
		pos.z = -1.0f;
	}

	if (pos.y < -1.0f)
	{
		pos.y = -1.0f;
	}
}
__global__
void UpdateVelocitys(
	float4 *oldPos,
	float4 *writePos,
	float4 *newPos,
	float deltaTime,
	float4 *newVel,
	uint    numParticles,
	uint  *gridParticleIndex
	)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;
	uint originalIndex = gridParticleIndex[index];
	float3 old_pos_i = make_float3(oldPos[originalIndex].x, oldPos[originalIndex].y, oldPos[originalIndex].z);
	boundary(old_pos_i);
	float3 new_pos_i = make_float3(newPos[index].x, newPos[index].y, newPos[index].z);
	boundary(new_pos_i);
	float3 new_vel_i = (new_pos_i - old_pos_i) / deltaTime;
	newVel[index] = make_float4(new_vel_i.x, new_vel_i.y, new_vel_i.z, 0.0f);
	writePos[originalIndex] = make_float4(new_pos_i, 1.0f);
}
__global__
void CorrectVelocitys(
	float3 *vorticity_corr_forces,
	float3 *xsphs,
	float deltaTime,
	float4 *sortedVel,
	uint    numParticles,
	float4 *newVel,
	uint  *gridParticleIndex
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;
	float3 vel_i = make_float3(sortedVel[index]);
	vel_i += vorticity_corr_forces[index] * deltaTime;
	vel_i += xsphs[index];
	uint originalIndex = gridParticleIndex[index];
	newVel[originalIndex] = make_float4(vel_i, 0.0f);
}
__global__ 
void AddPos(
	float4 *oldPos,
	float3 *delta_positions,
	uint    numParticles
)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;
	float4 pos = make_float4(0.0f);
	pos.x = delta_positions[index].x;
	pos.y = delta_positions[index].y;
	pos.z = delta_positions[index].z;
	oldPos[index] += pos;
}
__device__
void CalVor(int3    gridPos,
	uint    index,
	float3  pos,
	float3  vel,
	float4 *oldPos,
	float4 *oldVel,
	uint   *cellStart,
	uint   *cellEnd,
	float3 *vorticity
	)
{
	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = FETCH(cellStart, gridHash);

	float3 force = make_float3(0.0f);
	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index)                // check not colliding with self
			{
				float3 pos2 = make_float3(FETCH(oldPos, j));
				float3 vel2 = make_float3(FETCH(oldVel, j));
				const float3 pos_ji = pos - pos2;
				const float3 vel_ij = vel2 - vel;
				const float3 gradient = SpikyGradient(pos_ji, _h);
				*vorticity += cross(vel_ij, gradient);
			}
		}
	}
}
__device__
void CalVorCorrForces(int3    gridPos,
	uint    index,
	float3  pos,
	float4 *oldPos,
	uint   *cellStart,
	uint   *cellEnd,
	float3 *vorticities,
	float3 *eta
)
{
	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = FETCH(cellStart, gridHash);

	float3 force = make_float3(0.0f);
	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index)                // check not colliding with self
			{
				float3 pos2 = make_float3(FETCH(oldPos, j));
				const float3 pos_ji = pos - pos2;
				const float omega_j_len = length(vorticities[j]);
				const float3 gradient = SpikyGradient(pos_ji, _h);
				*eta += (omega_j_len * gradient);
			}
		}
	}
}
__global__
void VorticityCorrForces(
	float4 *oldPos,               // input: sorted positions
	uint   *gridParticleIndex,    // input: sorted particle indices
	uint   *cellStart,
	uint   *cellEnd,
	uint    numParticles,
	float3  *vorticities,
	float3  *vorticity_corr_forces
)// output: new pos
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	// read particle data from sorted arrays
	float3 pos = make_float3(FETCH(oldPos, index));

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	// examine neighbouring cells
	float3 eta = make_float3(0.0f);
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				CalVorCorrForces(neighbourPos, index, pos, oldPos, cellStart, cellEnd, vorticities, &eta);
			}
		}
	}
	const float eta_len = length(eta);
	float3 vort_corr_force = make_float3(0.0f);
	if (eta_len > 1e-6) {
		eta = normalize(eta);
		const float3 omega_i = vorticities[index];
		vort_corr_force = vorticity_epsilon * cross(eta, omega_i);
	}
	vorticity_corr_forces[index] = vort_corr_force;

	// collide with cursor sphere
   // force += collideSpheres(pos, params.colliderPos, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius, 0.0f);

	// write new velocity back to original unsorted location
	//uint originalIndex = gridParticleIndex[index];
	//newVel[originalIndex] = make_float4(vel + force, 0.0f);
}
__global__
void Vorticities(
	float4 *oldPos,               // input: sorted positions
	float4 *newVel,               // input: sorted velocities
	uint   *gridParticleIndex,    // input: sorted particle indices
	uint   *cellStart,
	uint   *cellEnd,
	uint    numParticles,
	float3  *vorticities
	)// output: new pos
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	// read particle data from sorted arrays
	float3 pos = make_float3(FETCH(oldPos, index));
	float3 vel = make_float3(FETCH(newVel, index));

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	// examine neighbouring cells
	float3 vorticity = make_float3(0.0f);
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				CalVor(neighbourPos, index, pos, vel, oldPos, newVel, cellStart, cellEnd, &vorticity);
			}
		}
	}
	vorticities[index] = vorticity;

	// collide with cursor sphere
   // force += collideSpheres(pos, params.colliderPos, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius, 0.0f);

	// write new velocity back to original unsorted location
	//uint originalIndex = gridParticleIndex[index];
	//newVel[originalIndex] = make_float4(vel + force, 0.0f);
}

__global__
void Pos(              
	float4 *oldPos,               // input: sorted positions
	uint   *gridParticleIndex,    // input: sorted particle indices
	uint   *cellStart,
	uint   *cellEnd,
	uint    numParticles,
	float  *lambdas,
	float3 *delta_positions)// output: new pos
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	// read particle data from sorted arrays
	float3 pos = make_float3(FETCH(oldPos, index));

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	// examine neighbouring cells
	float3 delta_pos_i = make_float3(0.0f);
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				CalPos(neighbourPos, index, pos, oldPos, cellStart, cellEnd, &delta_pos_i, lambdas);
			}
		}
	}
	delta_pos_i *= rho_0_recpr;
	delta_positions[index] = delta_pos_i;


	// collide with cursor sphere
   // force += collideSpheres(pos, params.colliderPos, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius, 0.0f);

	// write new velocity back to original unsorted location
	//uint originalIndex = gridParticleIndex[index];
	//newVel[originalIndex] = make_float4(vel + force, 0.0f);
}
__device__
void CalXsph(int3    gridPos,
	uint    index,
	float3  pos,
	float3  vel,
	float4 *oldPos,
	float4 *oldVel,
	uint   *cellStart,
	uint   *cellEnd,
	float3 *xsph
)
{
	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = FETCH(cellStart, gridHash);

	float3 force = make_float3(0.0f);
	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index)                // check not colliding with self
			{
				float3 pos2 = make_float3(FETCH(oldPos, j));
				float3 vel2 = make_float3(FETCH(oldVel, j));
				const float3 vel_ij = vel2 - vel;
				const float w = Poly6Value(pos - pos2, _h);
				*xsph += (w * vel_ij);
			}
		}
	}
}
__global__
void Xsph(
	float4 *oldPos,               // input: sorted positions
	float4 *oldVel,               // input: sorted velocities
	uint   *gridParticleIndex,    // input: sorted particle indices
	uint   *cellStart,
	uint   *cellEnd,
	uint    numParticles,
	float3 *Xsph)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	// read particle data from sorted arrays
	float3 pos = make_float3(FETCH(oldPos, index));
	float3 vel = make_float3(FETCH(oldVel, index));

	// get address in grid
	int3 gridPos = calcGridPos(pos);
	float3 xsph = make_float3(0.0f);
	
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				CalXsph(neighbourPos, index, pos, vel, oldPos, oldVel, cellStart, cellEnd, &xsph);
			}
		}
	}
	xsph *= xsph_c;
	Xsph[index] = xsph;


	// collide with cursor sphere
   // force += collideSpheres(pos, params.colliderPos, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius, 0.0f);

	// write new velocity back to original unsorted location
	//uint originalIndex = gridParticleIndex[index];
	//newVel[originalIndex] = make_float4(vel + force, 0.0f);
}

__global__
void Lamda(
              float4 *oldPos,               // input: sorted positions
              uint   *gridParticleIndex,    // input: sorted particle indices
              uint   *cellStart,
              uint   *cellEnd,
              uint    numParticles,
			  float *lambdas)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    // read particle data from sorted arrays
    float3 pos = make_float3(FETCH(oldPos, index));

    // get address in grid
    int3 gridPos = calcGridPos(pos);

    // examine neighbouring cells
    float3 force = make_float3(0.0f);
	float sum_gradient = 0.0f;
	float density_constraint = 0.0f;
	float3 gradient_i = make_float3(0.0f);
    for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
				CalLamda(neighbourPos, index, pos, oldPos, cellStart, cellEnd, &sum_gradient, &density_constraint, &gradient_i);
            }
        }
    }
	sum_gradient += dot(gradient_i, gradient_i);
	density_constraint = (density_constraint * rho_0_recpr) - 1.0f;
	const float lambda_i = (-density_constraint) / (sum_gradient + epsilon);
	lambdas[index] = lambda_i;
	

    // collide with cursor sphere
   // force += collideSpheres(pos, params.colliderPos, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius, 0.0f);

    // write new velocity back to original unsorted location
    //uint originalIndex = gridParticleIndex[index];
    //newVel[originalIndex] = make_float4(vel + force, 0.0f);
}


#define cudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %d in %s at %s:%d\n", err, #x, __FILE__, __LINE__); assert(0); } }

__constant__ float deltaT = 0.0083f;
__device__ int foamCount = 0;
__constant__ float distr[] =
{
	-0.34828757091811f, -0.64246175794046f, -0.15712936555833f, -0.28922267225069f, 0.70090742209037f,
	0.54293139350737f, 0.86755128105523f, 0.68346917800767f, -0.74589352018474f, 0.39762042062246f,
	-0.70243115988673f, -0.85088539675385f, -0.25780126697281f, 0.61167922970451f, -0.8751634423971f,
	-0.12334015086449f, 0.10898816916579f, -0.97167591190509f, 0.89839695948101f, -0.71134930649369f,
	-0.33928178406287f, -0.27579196788175f, -0.5057460942798f, 0.2341509513716f, 0.97802030852904f,
	0.49743173248015f, -0.92212845381448f, 0.088328595779989f, -0.70214782175708f, -0.67050553191011f
};

__device__ float WPoly6(float3 const &pi, float3 const &pj, TsolverParams &sp) {
	float3 r = pi - pj;
	float rLen = length(r);
	if (rLen > sp.radius || rLen == 0) {
		return 0;
	}

	return sp.KPOLY * pow((sp.radius * sp.radius - pow(length(r), 2)), 3);
}

__device__ float3 gradWPoly6(float3 const &pi, float3 const &pj, TsolverParams &sp) {
	float3 r = pi - pj;
	float rLen = length(r);
	if (rLen > sp.radius || rLen == 0) {
		return make_float3(0.0f);
	}

	float coeff = glm::pow((sp.radius * sp.radius) - (rLen * rLen), 2);
	coeff *= -6 * sp.KPOLY;
	return r * coeff;
}

__device__ float3 WSpiky(float3 const &pi, float3 const &pj, TsolverParams &sp) {
	float3 r = pi - pj;
	float rLen = length(r);
	if (rLen > sp.radius || rLen == 0) {
		return make_float3(0.0f);
	}

	float coeff = (sp.radius - rLen) * (sp.radius - rLen);
	coeff *= sp.SPIKY;
	coeff /= rLen;
	return r * -coeff;
}

__device__ float WAirPotential(float3 const &pi, float3 const &pj, TsolverParams &sp) {
	float3 r = pi - pj;
	float rLen = length(r);
	if (rLen > sp.radius || rLen == 0) {
		return 0.0f;
	}

	return 1 - (rLen / sp.radius);
}

//Returns the eta vector that points in the direction of the corrective force
__device__ float3 eta(float4* newPos, int* phases, int* neighbors, int* numNeighbors, int &index, float &vorticityMag, TsolverParams &sp) {
	float3 eta = make_float3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.maxNeighbors) + i]] == 0)
			eta += WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]]), sp) * vorticityMag;
	}

	return eta;
}

__device__ float3 vorticityForce(float4* newPos, float3* velocities, int* phases, int* neighbors, int* numNeighbors, int index, TsolverParams &sp) {
	//Calculate omega_i
	float3 omega = make_float3(0.0f);
	float3 velocityDiff;
	float3 gradient;

	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.maxNeighbors) + i]] == 0) {
			velocityDiff = velocities[neighbors[(index * sp.maxNeighbors) + i]] - velocities[index];
			gradient = WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]]), sp);
			omega += cross(velocityDiff, gradient);
		}
	}

	float omegaLength = length(omega);
	if (omegaLength == 0.0f) {
		//No direction for eta
		return make_float3(0.0f);
	}

	float3 etaVal = eta(newPos, phases, neighbors, numNeighbors, index, omegaLength, sp);
	if (etaVal.x == 0 && etaVal.y == 0 && etaVal.z == 0) {
		//Particle is isolated or net force is 0
		return make_float3(0.0f);
	}

	float3 n = normalize(etaVal);

	return (cross(n, omega) * sp.vorticityEps);
}

__device__ float sCorrCalc(float4 &pi, float4 &pj, TsolverParams &sp) {
	//Get Density from WPoly6
	float corr = WPoly6(make_float3(pi), make_float3(pj), sp) / sp.wQH;
	corr *= corr * corr * corr;
	return -sp.K * corr;
}

__device__ float3 xsphViscosity(float4* newPos, float3* velocities, int* phases, int* neighbors, int* numNeighbors, int index, TsolverParams &sp) {
	float3 visc = make_float3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.maxNeighbors) + i]] == 0) {
			float3 velocityDiff = velocities[neighbors[(index * sp.maxNeighbors) + i]] - velocities[index];
			velocityDiff *= WPoly6(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]]), sp);
			visc += velocityDiff;
		}
	}

	return visc * sp.C;
}

__device__ void confineToBox(float4 &pos, float3 &vel) {
	if (pos.x < -1.0) {
		vel.x = -vel.x * 0.2;
		pos.x = -1.0f + 0.001f;
	}
	else if (pos.x > 1.0) {
		vel.x = -vel.x * 0.2;
		pos.x = 1.0 - 0.001f;
	}

	if (pos.y < -1.0) {
		vel.y = -vel.y * 0.2;
		pos.y = -1.0f + 0.001f;
	}
	else if (pos.y > 1.0) {
		vel.y = -vel.y * 0.2;
		pos.y = 1.0 - 0.001f;
	}
	if (pos.z < -1.0) {
		vel.z = -vel.z * 0.2;
		pos.z = -1.0f + 0.001f;
	}
	else if (pos.z > 1.0) {
		vel.z = -vel.z * 0.2;
		pos.z = 1.0 - 0.001f;
	}
}

__device__ int3 getGridPos(float4 pos) {
	return make_int3((pos.x)*32.0f + 32, pos.y*32.0 + 32, pos.z*32.0 + 32);
}

__device__ int getGridIndex(int3 pos, TsolverParams &sp) {
	return (pos.z * sp.gridHeight * sp.gridWidth) + (pos.y * sp.gridWidth) + pos.x;
}

__global__ void predictPositions(float4* newPos, float3* velocities, TsolverParams &sp) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;
	//float3 pos = make_float3(newPos[index].x, newPos[index].y, newPos[index].z);
	//float3 vel = make_float3(velocities[index].x, velocities[index].y, velocities[index].z);
	//float3 force = 100*collideSpheres(pos, params.colliderPos, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius, 0.0f);
	//update velocity vi = vi + dt * fExt
	velocities[index] += ((newPos[index].w > 0) ? 1 : 0) * (sp.gravity) * deltaT;
	if (velocities[index].y > 0.1)  
		velocities[index].y = 0.1;
	newPos[index] += make_float4(velocities[index] * deltaT, 0);

	confineToBox(newPos[index], velocities[index]);
}

__global__ void clearNeighbors(int* numNeighbors, int* numContacts, TsolverParams &sp) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	numNeighbors[index] = 0;
	numContacts[index] = 0;
}

__global__ void clearGrid(int* gridCounters, TsolverParams &sp) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.gridSize) return;

	gridCounters[index] = 0;
}

__global__ void updateGrid(float4* newPos, int* gridCells, int* gridCounters, TsolverParams &sp) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	int3 pos = getGridPos(newPos[index]);
	int gIndex = getGridIndex(pos, sp);
	int i = atomicAdd(&gridCounters[gIndex], 1);
	i = min(i, sp.maxParticles - 1);
	gridCells[gIndex * sp.maxParticles + i] = index;
}

__global__ void updateNeighbors(float4* newPos, int* phases, int* gridCells, int* gridCounters, int* neighbors, int* numNeighbors, int* contacts, int* numContacts, TsolverParams &sp) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	int3 pos = getGridPos(newPos[index]);
	int pIndex;

	for (int z = -1; z < 2; z++) {
		for (int y = -1; y < 2; y++) {
			for (int x = -1; x < 2; x++) {
				int3 n = make_int3(pos.x + x, pos.y + y, pos.z + z);
				if (n.x >= 0 && n.x < sp.gridWidth && n.y >= 0 && n.y < sp.gridHeight && n.z >= 0 && n.z < sp.gridDepth) {
					int gIndex = getGridIndex(n, sp);
					int cellParticles = min(gridCounters[gIndex], sp.maxParticles - 1);
					for (int i = 0; i < cellParticles; i++) {
						if (numNeighbors[index] >= sp.maxNeighbors) return;

						pIndex = gridCells[gIndex * sp.maxParticles + i];
						if (length(make_float3(newPos[index]) - make_float3(newPos[pIndex])) <= sp.radius) {
							neighbors[(index * sp.maxNeighbors) + numNeighbors[index]] = pIndex;
							numNeighbors[index]++;
							if (phases[index] == 0 && phases[pIndex] == 1 && numContacts[index] < sp.maxContacts) {
								contacts[index * sp.maxContacts + numContacts[index]] = pIndex;
								numContacts[index]++;
							}
						}
					}
				}
			}
		}
	}
}

__global__ void particleCollisions(float4* newPos, int* contacts, int* numContacts, float3* deltaPs, float* buffer0, TsolverParams &sp) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	for (int i = 0; i < numContacts[index]; i++) {
		int nIndex = contacts[index * sp.maxContacts + i];
		if (newPos[nIndex].w == 0) continue;
		float3 dir = make_float3(newPos[index] - newPos[nIndex]);
		float len = length(dir);
		float invMass = newPos[index].w + newPos[nIndex].w;
		float3 dp;
		if (len > sp.radius || len == 0.0f || invMass == 0.0f) dp = make_float3(0);
		else dp = (1 / invMass) * (len - sp.radius) * (dir / len);
		deltaPs[index] -= dp * newPos[index].w;
		buffer0[index]++;

		atomicAdd(&deltaPs[nIndex].x, dp.x * newPos[nIndex].w);
		atomicAdd(&deltaPs[nIndex].y, dp.y * newPos[nIndex].w);
		atomicAdd(&deltaPs[nIndex].z, dp.z * newPos[nIndex].w);
		atomicAdd(&buffer0[nIndex], 1);
	}
}

__global__ void calcDensities(float4* newPos, int* phases, int* neighbors, int* numNeighbors, float* densities, TsolverParams &sp) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;

	float rhoSum = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.maxNeighbors) + i]] == 0)
			rhoSum += WPoly6(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]]), sp);
	}

	densities[index] = rhoSum;
}

__global__ void calcLambda(float4* newPos, int* phases, int* neighbors, int* numNeighbors, float* densities, float* buffer0, TsolverParams &sp) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;

	float densityConstraint = (densities[index] / sp.restDensity) - 1;
	float3 gradientI = make_float3(0.0f);
	float sumGradients = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.maxNeighbors) + i]] == 0) {
			//Calculate gradient with respect to j
			float3 gradientJ = WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]]), sp) / sp.restDensity;

			//Add magnitude squared to sum
			sumGradients += pow(length(gradientJ), 2);
			gradientI += gradientJ;
		}
	}

	//Add the particle i gradient magnitude squared to sum
	sumGradients += pow(length(gradientI), 2);
	buffer0[index] = (-1 * densityConstraint) / (sumGradients + sp.lambdaEps);
}

__global__ void calcDeltaP(float4* newPos, int* phases, int* neighbors, int* numNeighbors, float3* deltaPs, float* buffer0, TsolverParams &sp) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;
	deltaPs[index] = make_float3(0);

	float3 deltaP = make_float3(0.0f);
	for (int i = 0; i < numNeighbors[index]; i++) {
		if (phases[neighbors[(index * sp.maxNeighbors) + i]] == 0) {
			float lambdaSum = buffer0[index] + buffer0[neighbors[(index * sp.maxNeighbors) + i]];
			float sCorr = sCorrCalc(newPos[index], newPos[neighbors[(index * sp.maxNeighbors) + i]], sp);
			deltaP += WSpiky(make_float3(newPos[index]), make_float3(newPos[neighbors[(index * sp.maxNeighbors) + i]]), sp) * (lambdaSum + sCorr);

		}
	}

	deltaPs[index] = deltaP / sp.restDensity;
}

__global__ void applyDeltaP(float4* newPos, float3* deltaPs, float* buffer0, int flag, TsolverParams &sp) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	if (buffer0[index] > 0 && flag == 1) newPos[index] += make_float4(deltaPs[index] / buffer0[index], 0);
	else if (flag == 0) newPos[index] += make_float4(deltaPs[index], 0);
	//newPos[index] += make_float4(deltaPs[index], 0);
}

__global__ void updateVelocities(float4* oldPos, float4* newPos, float3* velocities, int* phases, int* neighbors, int* numNeighbors, float3* deltaPs, TsolverParams &sp) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;


	//set new velocity vi = (x*i - xi) / dt
	velocities[index] = (make_float3(newPos[index]) - make_float3(oldPos[index])) / deltaT;

	//apply vorticity confinement
	velocities[index] += vorticityForce(newPos, velocities, phases, neighbors, numNeighbors, index, sp) * deltaT;

	//apply XSPH viscosity
	deltaPs[index] = xsphViscosity(newPos, velocities, phases, neighbors, numNeighbors, index, sp);
	confineToBox(newPos[index], velocities[index]);
	//update position xi = x*i
	oldPos[index] = newPos[index];
}

__global__ void updateXSPHVelocities(float4* newPos, float3* velocities, int* phases, float3* deltaPs, TsolverParams &sp) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0) return;

	velocities[index] += deltaPs[index] * deltaT;
}

__global__ void generateFoam(float4* newPos, float3* velocities, int* phases, float4* diffusePos, float3* diffuseVelocities, int* neighbors, int* numNeighbors, float* densities, TsolverParams &sp) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles || phases[index] != 0 || foamCount >= sp.numDiffuse) return;

	float velocityDiff = 0.0f;
	for (int i = 0; i < numNeighbors[index]; i++) {
		int nIndex = neighbors[(index * sp.maxNeighbors) + i];
		if (index != nIndex) {
			float wAir = WAirPotential(make_float3(newPos[index]), make_float3(newPos[nIndex]), sp);
			float3 xij = normalize(make_float3(newPos[index] - newPos[nIndex]));
			float3 vijHat = normalize(velocities[index] - velocities[nIndex]);
			velocityDiff += length(velocities[index] - velocities[nIndex]) * (1 - dot(vijHat, xij)) * wAir;
		}
	}

	float ek = 0.5f * pow(length(velocities[index]), 2);
	float potential = velocityDiff * ek * max(1.0f - (1.0f * densities[index] / sp.restDensity), 0.0f);
	int nd = 0;
	if (potential > 0.5f) nd = min(20, (sp.numDiffuse - 1 - foamCount));
	if (nd <= 0) return;

	int count = atomicAdd(&foamCount, nd);
	count = min(count, sp.numDiffuse - 1);
	int cap = min(count + nd, sp.numDiffuse - 1);

	for (int i = count; i < cap; i++) {
		float rx = distr[i % 30] * sp.restDistance;
		float ry = distr[(i + 1) % 30] * sp.restDistance;
		float rz = distr[(i + 2) % 30] * sp.restDistance;
		int rd = distr[index % 30] > 0.5f ? 1 : -1;

		float3 xd = make_float3(newPos[index]) + make_float3(rx * rd, ry * rd, rz * rd);

		diffusePos[i] = make_float4(xd, 1);
		diffuseVelocities[i] = velocities[index];
	}

	if (foamCount >= sp.numDiffuse) atomicExch(&foamCount, sp.numDiffuse - 1);
}

__global__ void updateFoam(float4* newPos, float3* velocities, float4* diffusePos, float3* diffuseVelocities, int* gridCells, int* gridCounters, TsolverParams &sp) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numDiffuse || diffusePos[index].w <= 0) return;

	confineToBox(diffusePos[index], diffuseVelocities[index]);

	int3 pos = getGridPos(diffusePos[index]);
	int pIndex;
	int fluidNeighbors = 0;
	float3 vfSum = make_float3(0.0f);
	float kSum = 0;

	for (int z = -1; z < 2; z++) {
		for (int y = -1; y < 2; y++) {
			for (int x = -1; x < 2; x++) {
				int3 n = make_int3(pos.x + x, pos.y + y, pos.z + z);
				if (n.x >= 0 && n.x < sp.gridWidth && n.y >= 0 && n.y < sp.gridHeight && n.z >= 0 && n.z < sp.gridDepth) {
					int gIndex = getGridIndex(n, sp);
					int cellParticles = min(gridCounters[gIndex], sp.maxParticles - 1);
					for (int i = 0; i < cellParticles; i++) {
						pIndex = gridCells[gIndex * sp.maxParticles + i];
						if (length(make_float3(diffusePos[index] - newPos[pIndex])) <= sp.radius) {
							fluidNeighbors++;
							float k = WPoly6(make_float3(diffusePos[index]), make_float3(newPos[pIndex]), sp);
							vfSum += velocities[pIndex] * k;
							kSum += k;
						}
					}
				}
			}
		}
	}

	if (fluidNeighbors < 8) {
		//Spray
		//diffuseVelocities[index].x *= 0.8f;
		//diffuseVelocities[index].z *= 0.8f;
		diffuseVelocities[index] += sp.gravity * deltaT;
		diffusePos[index] += make_float4(diffuseVelocities[index] * deltaT, 0);
	}
	else {
		//Foam
		diffusePos[index] += make_float4((1.0f * (vfSum / kSum)) * deltaT, 0);
	}

	diffusePos[index].w -= deltaT;
	if (diffusePos[index].w <= 0.0f) {
		atomicSub(&foamCount, 1);
		diffusePos[index] = make_float4(0);
		diffuseVelocities[index] = make_float3(0);
	}
}

__global__ void clearDeltaP(float3* deltaPs, float* buffer0, TsolverParams &sp) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	deltaPs[index] = make_float3(0);
	buffer0[index] = 0;
}

__global__ void solveDistance(float4* newPos, int* clothIndices, float* restLengths, float* stiffness, float3* deltaPs, float* buffer0, TsolverParams &sp) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numConstraints) return;

	int p1 = clothIndices[2 * index];
	int p2 = clothIndices[2 * index + 1];

	float3 dir = make_float3(newPos[p1] - newPos[p2]);
	float len = length(dir);
	float invMass = newPos[p1].w + newPos[p2].w;
	float3 dp;
	if (len == 0.0f || invMass == 0.0f) dp = make_float3(0);
	else {
		if (stiffness[index] > 0) dp = (1 / invMass) * (len - restLengths[index]) * (dir / len) * (1.0f - pow(1.0f - stiffness[index], 1.0f / sp.numIterations));
		else if (len > restLengths[index]) {
			dp = (1 / invMass) * (len - restLengths[index]) * (dir / len) * (1.0f - pow(1.0f + stiffness[index], 1.0f / sp.numIterations));
		}
	}

	if (newPos[p1].w > 0) {
		atomicAdd(&deltaPs[p1].x, -dp.x * newPos[p1].w);
		atomicAdd(&deltaPs[p1].y, -dp.y * newPos[p1].w);
		atomicAdd(&deltaPs[p1].z, -dp.z * newPos[p1].w);
		atomicAdd(&buffer0[p1], 1);
	}

	if (newPos[p2].w > 0) {
		atomicAdd(&deltaPs[p2].x, dp.x * newPos[p2].w);
		atomicAdd(&deltaPs[p2].y, dp.y * newPos[p2].w);
		atomicAdd(&deltaPs[p2].z, dp.z * newPos[p2].w);
		atomicAdd(&buffer0[p2], 1);
	}
}

__global__ void updateClothVelocity(float4* oldPos, float4* newPos, float3* velocities, int* phases, TsolverParams &sp) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numCloth || phases[index] != 1) return;

	velocities[index] = make_float3(newPos[index] - oldPos[index]) / deltaT;
	oldPos[index] = newPos[index];
}

struct OBCmp {
	__host__ __device__
		bool operator()(const float4& a, const float4& b) const {
		return a.w > b.w;
	}
};
uint __iDivUp(uint a, uint b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}
void __computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = __iDivUp(n, numThreads);
}
void updateWater(solver* s, int numIterations, TsolverParams &sp) {
	uint numThreads, numBlocks;
	__computeGridSize(sp.numParticles, 256, numBlocks, numThreads);
	//------------------WATER-----------------
	

	//generateFoam<<<numBlocks, numThreads>>>(s->newPos, s->velocities, s->phases, s->diffusePos, s->diffuseVelocities, s->neighbors, s->numNeighbors, s->densities);
	//updateFoam<<<diffusenumBlocks, numThreads>>>(s->newPos, s->velocities, s->diffusePos, s->diffuseVelocities, s->gridCells, s->gridCounters);
}


#endif
