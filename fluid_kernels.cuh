#pragma once
#include <cuda_runtime.h>
#include <vector_functions.h>
#include <vector_types.h>

// kernels
void simulateFluid(float* u, float* v, float* uTmp, float* vTmp,
    float* pres, float* presTmp, float* div,
    uchar4* dye, uchar4* dyeTmp,
    int Nx, int Ny, size_t pitchF, size_t pitchC, float dt, float viscosity);

void launchAddDye(uchar4* dye, int Nx, int Ny, size_t pitchBytesC, int x, int y);
void launchAddForce(float* velX, float* velY, int Nx, int Ny, size_t pitchBytesF, int x, int y, float fx, float fy);
