#include "fluid_simulator.h"
#include "fluid_kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

//Construction & Destruction
FluidSimulator::FluidSimulator(int w, int h) : Nx(w), Ny(h),
d_velX(nullptr), d_velY(nullptr), d_velX_tmp(nullptr), d_velY_tmp(nullptr),
d_pressure(nullptr), d_pressure_tmp(nullptr), d_divergence(nullptr),
d_dye(nullptr), d_dye2(nullptr), dyeHost(nullptr)
{
    //Nothing to do here: allocations happen in init().
}

FluidSimulator::~FluidSimulator() {
    //cudaFree/cudaFreeHost accept nullptr, so we can free unconditionally.
    cudaFree(d_velX);
    cudaFree(d_velY);
    cudaFree(d_velX_tmp);
    cudaFree(d_velY_tmp);
    cudaFree(d_pressure);
    cudaFree(d_pressure_tmp);
    cudaFree(d_divergence);
    cudaFree(d_dye);
    cudaFree(d_dye2);
    if (dyeHost) cudaFreeHost(dyeHost);
}

//Initialization
bool FluidSimulator::init() {
    size_t pitch; //To capture the pitch returned by MALLOC2D
    //All pitches should be the same for each allocation
    cudaError_t err;
#define MALLOC2D(ptr) err = cudaMallocPitch((void**)&ptr,&pitch,Nx*sizeof(float),Ny);
    MALLOC2D(d_velX);
    MALLOC2D(d_velY);
    MALLOC2D(d_velX_tmp);
    MALLOC2D(d_velY_tmp);
    MALLOC2D(d_pressure);
    MALLOC2D(d_pressure_tmp);
    MALLOC2D(d_divergence);
#undef MALLOC2D
    pitchBytesF = pitch;

    //Allocation for dye buffers
    err = cudaMallocPitch((void**)&d_dye, &pitch, Nx * sizeof(uchar4), Ny);
    if (err != cudaSuccess) { std::cerr << cudaGetErrorString(err) << "\n"; return false; }
    err = cudaMallocPitch((void**)&d_dye2, &pitch, Nx * sizeof(uchar4), Ny);
    if (err != cudaSuccess) { std::cerr << cudaGetErrorString(err) << "\n"; return false; }
    pitchBytesC = pitch;

    //Allocate a pinned host buffer for fast device->host copies each frame
    cudaHostAlloc((void**)&dyeHost, Nx*Ny*sizeof(uchar4), cudaHostAllocMapped);
    
    //Clears fields before first frame
    reset();
    return true;
}

//State management
void FluidSimulator::reset() {
    //Zero all buffers
    cudaMemset2D(d_velX, pitchBytesF, 0, Nx * sizeof(float), Ny);
    cudaMemset2D(d_velX_tmp, pitchBytesF, 0, Nx * sizeof(float), Ny);
    cudaMemset2D(d_velY, pitchBytesF, 0, Nx * sizeof(float), Ny);
    cudaMemset2D(d_velY_tmp, pitchBytesF, 0, Nx * sizeof(float), Ny);
    cudaMemset2D(d_dye, pitchBytesC, 0, Nx * sizeof(uchar4), Ny);
    cudaMemset2D(d_pressure, pitchBytesF, 0, Nx * sizeof(float), Ny);
}

//Interaction
void FluidSimulator::injectDye(int x, int y) {
    //Launches a kernel that adds dye at (x, y)
    launchAddDye(d_dye, Nx, Ny, pitchBytesC, x, y);
}

void FluidSimulator::applyForce(int x, int y, int dx, int dy) {
    //Convert mouse motion into a velocity impulse and add it to the field
    launchAddForce(d_velX, d_velY, Nx, Ny, pitchBytesF, x, y, dx * forceScale, dy * forceScale);
}

//Simulation step
void FluidSimulator::update(float dt) {
    dt = std::min(dt, 0.033f);
    simulateFluid(d_velX, d_velY, d_velX_tmp, d_velY_tmp, d_pressure, d_pressure_tmp, d_divergence, d_dye, d_dye2,
        Nx, Ny, pitchBytesF, pitchBytesC, dt, viscosity);
}

//Rendering helpers
void FluidSimulator::copyDyeToHost() {
    cudaMemcpy2D(dyeHost, Nx * sizeof(uchar4), d_dye, pitchBytesC, Nx * sizeof(uchar4), Ny, cudaMemcpyDeviceToHost);
}
