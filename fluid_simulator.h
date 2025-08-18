#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <glm/glm.hpp>

class FluidSimulator {
public:
    FluidSimulator(int w, int h); //Construct simulator for  grid of size w x h
    ~FluidSimulator(); //Release GPU and host resources

    bool init(); //Allocate GPU/CPU buffers and clear simulation
    void reset(); //Reset dynamic fields to zero
    void update(float dt);

    //Interaction
    void injectDye(int x, int y); //Dye
    void applyForce(int x, int y, int dx, int dy); //Force

    //Render helpers
    void copyDyeToHost(); //Copies dye from device to host
    const uchar4* getDyeHost() const { return dyeHost; }

    int width() const { return Nx; }
    int height() const { return Ny; }

private:
    int Nx, Ny; //Grid size
    size_t pitchBytesF, pitchBytesC; //Row pitch for floats and uchar4

    // device fields
    float* d_velX;
    float* d_velY;
    float* d_velX_tmp;
    float* d_velY_tmp;
    float* d_pressure;
    float* d_pressure_tmp;
    float* d_divergence;
    uchar4* d_dye;
    uchar4* d_dye2;

    // host pinned buffer
    uchar4* dyeHost;

    // simulation params
    float viscosity = 0.f;
    float forceScale = 1.0f;
    float dtAccum = 0.f;
};
