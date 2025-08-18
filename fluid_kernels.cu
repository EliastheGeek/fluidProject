#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <cstdio>
#include <algorithm>
#include <cmath>

//Tunables (adjust in one place)
#define BLOCK_SIZE          16
#define JACOBI_ITERS        30
#define DYE_JACOBI_ITERS    30 
#define DYE_DIFFUSION       5.f         

//Utility helpers (host & device)
__device__ __host__ inline int divUp(int a, int b) { return (a + b - 1) / b; }
__device__ inline float lerpf(float a, float b, float t) { return a + t * (b - a); }
__device__ inline float u8_to_f(unsigned char v) { return v * (1.0f / 255.0f); }
__device__ inline unsigned char f_to_u8(float f) {
    f = fminf(1.0f, fmaxf(0.0f, f)); return (unsigned char)lrintf(255.0f * f);
}

//1. Semi‑Lagrangian advection (velocity component – float)
__global__ void k_advectFloat(const float* __restrict__ velX,
    const float* __restrict__ velY,
    const float* __restrict__ src,
    float* dst,
    int Nx, int Ny, int pitchF, float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;
    
    float xf = x - dt * velX[y * pitchF + x];
    float yf = y - dt * velY[y * pitchF + x];

    xf = fminf(fmaxf(xf, 0.f), Nx - 1.f);
    yf = fminf(fmaxf(yf, 0.f), Ny - 1.f);

    float maxStep = 0.9f;
    float step = hypotf(x - xf, y - yf);
    if (step > maxStep) {
        float s = maxStep / (step + 1e-6f);
        xf = x - (x - xf) * s; yf = y - (y - yf) * s;
    }

    int x0 = (int)floorf(xf), y0 = (int)floorf(yf);
    int x1 = min(x0 + 1, Nx - 1), y1 = min(y0 + 1, Ny - 1);
    float sx = xf - x0, sy = yf - y0;

    float v00 = src[y0 * pitchF + x0];
    float v10 = src[y0 * pitchF + x1];
    float v01 = src[y1 * pitchF + x0];
    float v11 = src[y1 * pitchF + x1];

    dst[y * pitchF + x] = lerpf(lerpf(v00, v10, sx), lerpf(v01, v11, sx), sy);
}

//1b. Semi‑Lagrangian advection (dye – uchar4 red channel)
__global__ void k_advectDye(const float* __restrict__ velX,
    const float* __restrict__ velY,
    const uchar4* __restrict__ src,
    uchar4* dst,
    int Nx, int Ny, int pitchF, int pitchC, float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    float xf = x - dt * velX[y * pitchF + x];
    float yf = y - dt * velY[y * pitchF + x];

    xf = fminf(fmaxf(xf, 0.f), Nx - 1.f);
    yf = fminf(fmaxf(yf, 0.f), Ny - 1.f);

    float maxStep = 0.9f;
    float step = hypotf(x - xf, y - yf);
    if (step > maxStep) {
        float s = maxStep / (step + 1e-6f);
        xf = x - (x - xf) * s; yf = y - (y - yf) * s;
    }

    int x0 = (int)floorf(xf), y0 = (int)floorf(yf);
    int x1 = min(x0 + 1, Nx - 1), y1 = min(y0 + 1, Ny - 1);
    float sx = xf - x0, sy = yf - y0;

    float d00 = u8_to_f(src[y0 * pitchC + x0].x);
    float d10 = u8_to_f(src[y0 * pitchC + x1].x);
    float d01 = u8_to_f(src[y1 * pitchC + x0].x);
    float d11 = u8_to_f(src[y1 * pitchC + x1].x);

    float dens = lerpf(lerpf(d00, d10, sx), lerpf(d01, d11, sx), sy);
    dst[y * pitchC + x] = make_uchar4(f_to_u8(dens), 0, 0, 255);
}

//2. Jacobi iteration
__global__ void k_jacobiDiffuse(const float* __restrict__ xv, const float* __restrict__ x0, float* xOut,
    int Nx, int Ny, int pitch,
    float alpha, float invBeta)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;
    int idx = y * pitch + x;

    float l = xv[y * pitch + max(x - 1, 0)];
    float r = xv[y * pitch + min(x + 1, Nx - 1)];
    float d = xv[max(y - 1, 0) * pitch + x];
    float u = xv[min(y + 1, Ny - 1) * pitch + x];

    xOut[idx] = (x0[idx] + alpha * (l + r + u + d)) * invBeta;
}

__global__ void k_jacobiPressure(const float* __restrict__ pIn, const float* __restrict__ div, float* pOut,
    int Nx, int Ny, int pitchF)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;
    int idx = y * pitchF + x;
    int idxL = idx - 1, idxR = idx + 1;
    int idxU = idx - pitchF, idxD = idx + pitchF;

    float sumN = pIn[idxL] + pIn[idxR] + pIn[idxU] + pIn[idxD];
    pOut[idx] = (sumN - div[idx]) * 0.25f;
}

//2b. Jacobi iteration for dye
__global__ void prepareDyeRHS(uchar4* dye, int Nx, int Ny, int pitchC) 
//Stores the RHS in an unused field of the dye variable
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int idx = y * pitchC + x;
    uchar4 c = dye[idx];
    c.y = c.x;
    c.w = 255;
    dye[idx] = c;
}

__global__ void jacobiDye(const uchar4* in, uchar4* out,
    int Nx, int Ny, int pitchC, float alpha)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || y <= 0 || x >= Nx - 1 || y >= Ny - 1) return;

    int idx = y * pitchC + x;
    int idxL = idx - 1, idxR = idx + 1;
    int idxU = idx - pitchC, idxD = idx + pitchC;

    uchar4 c = in[idx];
    float rhs = u8_to_f(c.y);

    float l = u8_to_f(in[idxL].x);
    float r = u8_to_f(in[idxR].x);
    float u = u8_to_f(in[idxU].x);
    float d = u8_to_f(in[idxD].x);

    float xnew = (rhs + alpha * (l + r + u + d)) / (1.0f + 4.0f * alpha);

    uchar4 o = c;  o.x = f_to_u8(xnew);
    out[idx] = o;
}

//3. Compute divergence
__global__ void k_divergence(const float* __restrict__ u, const float* __restrict__ v,
    float* div, int Nx, int Ny, int pitchF)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;
    int idx = y * pitchF + x;
    int idxL = idx - 1, idxR = idx + 1;
    int idxU = idx - pitchF, idxD = idx + pitchF;

    float dudx = u[idxR] - u[idxL];
    float dvdy = v[idxD] - v[idxU];

    div[idx] = 0.5f * (dudx + dvdy);
}

//4. Projection – subtract pressure gradient from velocity
__global__ void k_subtractGradient(float* u, float* v, const float* __restrict__ p,
    int Nx, int Ny, int pitchF)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;
    int idx   = y * pitchF + x;
    int idxL  = idx - 1, idxR = idx + 1;
    int idxU  = idx - pitchF, idxD = idx + pitchF;

    float dpdx = 0.5f * (p[idxR] - p[idxL]);
    float dpdy = 0.5f * (p[idxD] - p[idxU]);

    u[idx] -= dpdx;
    v[idx] -= dpdy;
}

//5. Boundary conditions – solid walls
__global__ void k_setBoundaryVel(float* u, float* v, int Nx, int Ny, int pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < Nx) {
        u[x] = u[pitch + x];
        v[x] = -v[pitch + x];
        int b = (Ny - 1) * pitch + x;
        u[b] = u[(Ny - 2) * pitch + x];
        v[b] = -v[(Ny - 2) * pitch + x];;
    }
    if (y < Ny) {
        int l = y * pitch, r = l + (Nx - 1);
        u[l] = -u[l + 1];
        v[l] = v[l + 1];
        u[r] = -u[r - 1];
        v[r] = v[r - 1];
    }
}

__global__ void k_setBoundaryPressure(float* p, int Nx, int Ny, int pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < Nx) { p[x] = p[pitch + x]; int b = (Ny - 1) * pitch + x; p[b] = p[(Ny - 2) * pitch + x]; }
    if (y < Ny) { int l = y * pitch, r = l + (Nx - 1); p[l] = p[l + 1]; p[r] = p[r - 1]; }
}

__global__ void k_setBoundaryDye(uchar4* dye, int Nx, int Ny, int pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < Nx) { dye[x] = dye[pitch + x]; int b = (Ny - 1) * pitch + x; dye[b] = dye[(Ny - 2) * pitch + x]; }
    if (y < Ny) { int l = y * pitch, r = l + (Nx - 1); dye[l] = dye[l + 1]; dye[r] = dye[r - 1]; }
}

//6. Interaction kernels – add dye / force from mouse
__global__ void k_addDye(uchar4* dye, int Nx, int Ny, int pitch, int cx, int cy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    const int R = 15;
    int dx = x - cx;
    int dy = y - cy;
    if (dx * dx + dy * dy <= R * R)
        dye[y * pitch + x] = make_uchar4(255, 0, 0, 255);
}

__global__ void k_addForce(float* u, float* v, int Nx, int Ny, int pitch,
    int cx, int cy, float fx, float fy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int sigma = 20;
    int dx = x - cx;
    int dy = y - cy;

    float falloff = expf(-(dx * dx + dy * dy) / (2 * sigma * sigma));
    int idx = y * pitch + x;
    u[idx] += fx * falloff;
    v[idx] += fy * falloff;
}

//7. Wrapper around Jacobi sweeps
static void jacobiDiffuseSolve(const float* d_uRHS, float* d_u, float* uTmp,
    const float* d_vRHS, float* d_v, float* vTmp,
    int Nx, int Ny, int pitch, float alpha, int iters)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(divUp(Nx, BLOCK_SIZE), divUp(Ny, BLOCK_SIZE));
    float invBeta = 1.f / (1.f + 4.f * alpha);

    for (int i = 0; i < iters; ++i) {
        k_jacobiDiffuse << <grid, block >> > (d_u, d_uRHS, uTmp, Nx, Ny, pitch, alpha, invBeta);
        k_setBoundaryVel << <grid, block >> > (uTmp, d_v, Nx, Ny, pitch);
        std::swap(d_u, uTmp);

        k_jacobiDiffuse << <grid, block >> > (d_v, d_vRHS, vTmp, Nx, Ny, pitch, alpha, invBeta);
        k_setBoundaryVel << <grid, block >> > (d_u, vTmp, Nx, Ny, pitch);
        std::swap(d_v, vTmp);
    }
    cudaDeviceSynchronize();
}

static void jacobiPressureSolve(float* p, float* pTmp, const float* div, int Nx, int Ny, int pitch, int iters)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(divUp(Nx, BLOCK_SIZE), divUp(Ny, BLOCK_SIZE));

    for (int i = 0; i < iters; ++i) {
        k_jacobiPressure << <grid, block >> > (p, div, pTmp, Nx, Ny, pitch);
        k_setBoundaryPressure << <grid, block >> > (pTmp, Nx, Ny, pitch);
        std::swap(p, pTmp);
    }
    cudaDeviceSynchronize();
}

//8. Public host API functions (called from C++)
void launchAddDye(uchar4* dye, int Nx, int Ny, size_t pitchBytes,
    int cx, int cy)
{
    int pitch = int(pitchBytes / sizeof(uchar4));
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(divUp(Nx, BLOCK_SIZE), divUp(Ny, BLOCK_SIZE));
    k_addDye << <grid, block >> > (dye, Nx, Ny, pitch, cx, cy);
    cudaDeviceSynchronize();
}

void launchAddForce(float* u, float* v,
    int Nx, int Ny, size_t pitchBytes,
    int cx, int cy, float fx, float fy)
{
    int pitch = int(pitchBytes / sizeof(float));
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(divUp(Nx, BLOCK_SIZE), divUp(Ny, BLOCK_SIZE));
    k_addForce << <grid, block >> > (u, v, Nx, Ny, pitch, cx, cy, fx, fy);
    cudaDeviceSynchronize();
}

//9. Main simulation step
void simulateFluid(float* d_u, float* d_v,
    float* d_uTmp, float* d_vTmp,
    float* d_p, float* d_pTmp,
    float* d_div,
    uchar4* d_dye, uchar4* d_dyeTmp,
    int Nx, int Ny, size_t pitchBytesF, size_t pitchBytesC,
    float dt, float viscosity)
{
    float* u_orig = d_u;
    float* v_orig = d_v;
    uchar4* dye_orig = d_dye;

    int pitchF = int(pitchBytesF / sizeof(float));
    int pitchC = int(pitchBytesC / sizeof(uchar4));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(divUp(Nx, BLOCK_SIZE), divUp(Ny, BLOCK_SIZE));

    //1) Advect velocity
    k_advectFloat << <grid, block >> > (d_u, d_v, d_u, d_uTmp, Nx, Ny, pitchF, dt);
    k_advectFloat << <grid, block >> > (d_u, d_v, d_v, d_vTmp, Nx, Ny, pitchF, dt);
    std::swap(d_u, d_uTmp); std::swap(d_v, d_vTmp);
    k_setBoundaryVel << <grid, block >> > (d_u, d_v, Nx, Ny, pitchF);

    //2) Diffuse velocity
    float alphaV = viscosity * dt;
    cudaMemcpy2D(d_pTmp, pitchBytesF, d_u, pitchBytesF, Nx * sizeof(float), Ny, cudaMemcpyDeviceToDevice);
    cudaMemcpy2D(d_div, pitchBytesF, d_v, pitchBytesF, Nx * sizeof(float), Ny, cudaMemcpyDeviceToDevice);
    jacobiDiffuseSolve(d_pTmp, d_u, d_uTmp, d_div, d_v, d_vTmp, Nx, Ny, pitchF, alphaV, JACOBI_ITERS);
    k_setBoundaryVel << <grid, block >> > (d_u, d_v, Nx, Ny, pitchF);

    //3) Projection: build divergence, pressure solve, subtract gradient
    k_divergence << <grid, block >> > (d_u, d_v, d_div, Nx, Ny, pitchF);
    cudaMemset2D(d_p, pitchBytesF, 0, Nx * sizeof(float), Ny); // initial p = 0
    jacobiPressureSolve(d_p, d_pTmp, d_div, Nx, Ny, pitchF, JACOBI_ITERS);
    k_subtractGradient << <grid, block >> > (d_u, d_v, d_p, Nx, Ny, pitchF);
    k_setBoundaryVel << <grid, block >> > (d_u, d_v, Nx, Ny, pitchF);

    //4) Advect dye
    k_advectDye << <grid, block >> > (d_u, d_v, d_dye, d_dyeTmp, Nx, Ny, pitchF, pitchC, dt);
    std::swap(d_dye, d_dyeTmp);
    k_setBoundaryDye << <grid, block >> > (d_dye, Nx, Ny, pitchC);

    //5) Diffuse dye
    float alphaD = DYE_DIFFUSION * dt;
    prepareDyeRHS << <grid, block >> > (d_dye, Nx, Ny, pitchC);
    cudaMemcpy2D(d_dyeTmp, pitchBytesC, d_dye, pitchBytesC, Nx * sizeof(uchar4), Ny, cudaMemcpyDeviceToDevice);

    for (int it = 0; it < DYE_JACOBI_ITERS; ++it) {
        jacobiDye << <grid, block >> > (d_dye, d_dyeTmp, Nx, Ny, pitchC, alphaD);
        std::swap(d_dye, d_dyeTmp);
        k_setBoundaryDye << <grid, block >> > (d_dye, Nx, Ny, pitchC);
    }

    k_setBoundaryPressure << <grid, block >> > (d_p, Nx, Ny, pitchF);
    k_setBoundaryVel << <grid, block >> > (d_u, d_v, Nx, Ny, pitchF);

    //6) Ensure caller‑owned pointers receive the up‑to‑date data
    if (d_u != u_orig)
        cudaMemcpy2D(u_orig, pitchBytesF, d_u, pitchBytesF, Nx * sizeof(float), Ny, cudaMemcpyDeviceToDevice);
    if (d_v != v_orig)
        cudaMemcpy2D(v_orig, pitchBytesF, d_v, pitchBytesF, Nx * sizeof(float), Ny, cudaMemcpyDeviceToDevice);
    if (d_dye != dye_orig)
        cudaMemcpy2D(dye_orig, pitchBytesC, d_dye, pitchBytesC, Nx * sizeof(uchar4), Ny, cudaMemcpyDeviceToDevice);

    cudaDeviceSynchronize();
}