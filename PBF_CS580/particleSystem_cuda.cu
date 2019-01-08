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

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>

#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include "particleSystem.h"
#include "particles_kernel_impl.cuh"

extern "C"
{

    void cudaInit(int argc, char **argv)
    {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);

        if (devID < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    void allocateArray(void **devPtr, size_t size)
    {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }

    void freeArray(void *devPtr)
    {
        checkCudaErrors(cudaFree(devPtr));
    }

    void threadSync()
    {
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void copyArrayToDevice(void *device, const void *host, int offset, int size)
    {
        checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
    }

    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,
                                                     cudaGraphicsMapFlagsNone));
    }

    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
    }

    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
    {
        void *ptr;
        checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
                                                             *cuda_vbo_resource));
        return ptr;
    }

    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
    }

    void copyArrayFromDevice(void *host, const void *device,
                             struct cudaGraphicsResource **cuda_vbo_resource, int size)
    {
        if (cuda_vbo_resource)
        {
            device = mapGLBufferObject(cuda_vbo_resource);
        }

        checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

        if (cuda_vbo_resource)
        {
            unmapGLBufferObject(*cuda_vbo_resource);
        }
    }

    void setParameters(SimParams *hostParams)
    {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
    }

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    void integrateSystem(float *pos,
                         float *vel,
                         float deltaTime,
                         uint numParticles)
    {
        thrust::device_ptr<float4> d_pos4((float4 *)pos);
        thrust::device_ptr<float4> d_vel4((float4 *)vel);

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4)),
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4+numParticles, d_vel4+numParticles)),
            integrate_functor(deltaTime));
    }

    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  int    numParticles)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
                                               gridParticleIndex,
                                               (float4 *) pos,
                                               numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *sortedPos,
                                     float *sortedVel,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     float *oldPos,
                                     float *oldVel,
                                     uint   numParticles,
                                     uint   numCells)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // set all cells to empty
        checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

#if USE_TEX
        checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4)));
        checkCudaErrors(cudaBindTexture(0, oldVelTex, oldVel, numParticles*sizeof(float4)));
#endif

        uint smemSize = sizeof(uint)*(numThreads+1);
        reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
            cellStart,
            cellEnd,
            (float4 *) sortedPos,
            (float4 *) sortedVel,
            gridParticleHash,
            gridParticleIndex,
            (float4 *) oldPos,
            (float4 *) oldVel,
            numParticles);
        getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

#if USE_TEX
        checkCudaErrors(cudaUnbindTexture(oldPosTex));
        checkCudaErrors(cudaUnbindTexture(oldVelTex));
#endif
    }
	void setParams(solverParams *tempParams, TsolverParams *dtmp) {
		cudaMemcpy(dtmp, tempParams, sizeof(solverParams), cudaMemcpyHostToDevice);
	}
	void PSupdate(solver* s, solverParams* sp, ParticleSystem *ps, TsolverParams *dtmp) {
		//Predict positions and update velocity
		static int cnt = 0;
		setParams(sp, dtmp);
		//std::ofstream sb("sbczx" + std::to_string(cnt++) + ".txt", std::ios::out);
		//ps->dumpParticles(0, ps->getNumParticles(), sb, (float *)dtmp);
		//sb.close();
		uint numThreads, numBlocks;
		computeGridSize(sp->numParticles, 256, numBlocks, numThreads);
		predictPositions <<< numBlocks, numThreads >>> (s->newPos, s->velocities, *dtmp);
		
		
		//Update neighbors
		clearNeighbors <<< numBlocks, numThreads >>> (s->numNeighbors, s->numContacts, *dtmp);
		clearGrid <<<64*64*64 / numThreads + 1, numThreads >>> (s->gridCounters, *dtmp);
		updateGrid <<< numBlocks, numThreads >>> (s->newPos, s->gridCells, s->gridCounters, *dtmp);
		updateNeighbors <<< numBlocks, numThreads >>> (s->newPos, s->phases, s->gridCells, s->gridCounters, s->neighbors, s->numNeighbors, s->contacts, s->numContacts, *dtmp);

		/*for (int i = 0; i < sp->numIterations; i++) {
			clearDeltaP <<< numBlocks, numThreads >>> (s->deltaPs, s->buffer0, *dtmp);
			particleCollisions <<< numBlocks, numThreads >>> (s->newPos, s->contacts, s->numContacts, s->deltaPs, s->buffer0, *dtmp);
			applyDeltaP <<< numBlocks, numThreads >>> (s->newPos, s->deltaPs, s->buffer0, 1, *dtmp);
		}*/

		//Solve constraints
		for (int i = 0; i < sp->numIterations; i++) {
			//Calculate fluid densities and store in densities
			calcDensities <<<numBlocks, numThreads >>> (s->newPos, s->phases, s->neighbors, s->numNeighbors, s->densities, *dtmp);

			//Calculate all lambdas and store in buffer0
			calcLambda <<<numBlocks, numThreads >>> (s->newPos, s->phases, s->neighbors, s->numNeighbors, s->densities, s->buffer0, *dtmp);

			//calculate deltaP
			calcDeltaP <<<numBlocks, numThreads >>> (s->newPos, s->phases, s->neighbors, s->numNeighbors, s->deltaPs, s->buffer0, *dtmp);
			particleCollisions << < numBlocks, numThreads >> > (s->newPos, s->contacts, s->numContacts, s->deltaPs, s->buffer0, *dtmp);
			//update position x*i = x*i + deltaPi
			applyDeltaP <<<numBlocks, numThreads >>> (s->newPos, s->deltaPs, s->buffer0, 0, *dtmp);
		}

		//Update velocity, apply vorticity confinement, apply xsph viscosity, update position
		updateVelocities <<<numBlocks, numThreads >>> (s->oldPos, s->newPos, s->velocities, s->phases, s->neighbors, s->numNeighbors, s->deltaPs, *dtmp);

		//Set new velocity
		updateXSPHVelocities <<<numBlocks, numThreads >>> (s->newPos, s->velocities, s->phases, s->deltaPs, *dtmp);

		/*thrust::device_ptr<float4> devPtr = thrust::device_pointer_cast(s->diffusePos);
		thrust::sort(devPtr, devPtr + sp->numDiffuse, OBCmp());
		updateCloth(s, sp->numIterations);*/
	}

    void collide(float *newVel,
				 float *newPos,
                 float *sortedPos,
				 float *sortedVel,
				 float *old_position,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   numParticles,
                 uint   numCells,
				 float *m_lamda,
				 float *delta_positions,
				 float *vorticities,
				 float *vorticity_corr_forces,
				 float *xsph)
    {
#if USE_TEX
        checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
        checkCudaErrors(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));
        checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
        checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));
#endif

        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

		
        // execute the kernel
		for (int i = 0; i < 10; i++)
		{
			Lamda<<< numBlocks, numThreads >>>(
				(float4 *)sortedPos,
				gridParticleIndex,
				cellStart,
				cellEnd,
				numParticles,
				m_lamda);
			cudaDeviceSynchronize();
			Pos<<< numBlocks, numThreads >>>(
				(float4 *)sortedPos,
				gridParticleIndex,
				cellStart,
				cellEnd,
				numParticles,
				m_lamda,
				(float3 *)delta_positions
				);
			cudaDeviceSynchronize();
			AddPos<<< numBlocks, numThreads >>>(
				(float4 *)sortedPos,
				(float3 *)delta_positions,
				numParticles
				);
			cudaDeviceSynchronize();
		}
		UpdateVelocitys <<<numBlocks, numThreads >>> (
			(float4 *)old_position,
			(float4 *)newPos,
			(float4 *)sortedPos,
			0.5f,
			(float4 *)sortedVel,
			numParticles,
			gridParticleIndex
			);
		cudaDeviceSynchronize();
		Vorticities <<<numBlocks, numThreads >>>(
			(float4 *)sortedPos,
			(float4 *)sortedVel,
			gridParticleIndex,
			cellStart,
			cellEnd,
			numParticles,
			(float3 *)vorticities
			);
		cudaDeviceSynchronize();
		VorticityCorrForces <<<numBlocks, numThreads >>> (
			(float4 *)sortedPos,
			gridParticleIndex,
			cellStart,
			cellEnd,
			numParticles,
			(float3 *)vorticities,
			(float3 *)vorticity_corr_forces
		);
		cudaDeviceSynchronize();
		Xsph <<<numBlocks, numThreads >>> (
			(float4 *)sortedPos,
			(float4 *)sortedVel,
			gridParticleIndex,
			cellStart,
			cellEnd,
			numParticles,
			(float3 *)xsph
		);
		cudaDeviceSynchronize();
		CorrectVelocitys <<<numBlocks, numThreads >>> (
			(float3 *)vorticity_corr_forces,
			(float3 *)xsph,
			0.5f,
			(float4 *)sortedVel,
			numParticles,
			(float4 *)newVel,
			gridParticleIndex
			);
		cudaDeviceSynchronize();
        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");

#if USE_TEX
        checkCudaErrors(cudaUnbindTexture(oldPosTex));
        checkCudaErrors(cudaUnbindTexture(oldVelTex));
        checkCudaErrors(cudaUnbindTexture(cellStartTex));
        checkCudaErrors(cudaUnbindTexture(cellEndTex));
#endif
    }


    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
    {
        thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
                            thrust::device_ptr<uint>(dGridParticleHash + numParticles),
                            thrust::device_ptr<uint>(dGridParticleIndex));
    }

}   // extern "C"
