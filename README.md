# fluidProject

Ever wondered how video games create realistic smoke and water in real-time? I’m building a GPU-accelerated fluid simulation from scratch in C++. This project aims to use Navier-Stokes for fast fluid dynamics simulation on the GPU inspired by the GPU Gems chapter 38 article.

## Background

Fluid dynamics simulation is a prominent topic in computer graphics and games, as fluids like smoke, water, and fire are ubiquitous in the natural world (Harris, 2004). The Navier-Stokes equations for incompressible flow, which describe how fluid velocity and pressure evolve over time. However, simulating fluids is computationally intensive due to the need to solve partial differential equations across many points in space and time.

In 1999, Jos Stam introduced the “Stable Fluids” method that made real-time fluid simulation feasible by using an unconditionally stable solver for the Navier-Stokes equations (Stam, 1999). His approach allows larger time steps without numerical instability, trading some physical accuracy for stability and speed. His solver is the same one presented in the GPU Gems chapter 38 that this project will utilize. It work by splitting the fluid update into steps: advection (moving quantities with the flow), diffusion, adding external forces, and projecting the velocity field to be divergence-free (incompressible).

Modern GPUs offer massively parallel computing power, which can be leveraged for fluid simulations. In fact, GPUs are highly suited to grid-based computations like fluid simulation because they excel at data-parallel tasks on arrays (textures) of values. Mark Harris’s Chapter 38 in GPU Gems (2004) demonstrated a GPU-based fluid solver achieving several times speedup over a CPU version (Harris, 2004). NVIDIA’s CUDA, introduced in 2007, allows general-purpose programming on GPUs using C/C++ (Wikipedia, 2025). This project will use CUDA to implement the fluid solver so that each GPU thread handles a cell of the fluid grid simultaneously, thereby updating the entire fluid field each frame much faster than a single CPU could. To visualize the fluid in real-time, the project will use SDL2 for graphics display and user interaction. It is lightweight and works with C++, making it suitable for our needs.

## Implementation

The simulation will solve the 2D incompressible Navier-Stokes equations on a uniform grid. We focus on a rectangular 2D domain (no full 3D fluids) to keep the problem tractable and interactive (Harris, 2004). The fluid is assumed incompressible and of uniform density (no density variation), meaning we enforce mass conservation (zero divergence in the velocity field). We will simulate a single fluid (like air or water) with no phase transitions, interfaces like water-air free surfaces are beyond the scope for this project (Harris, 2004).

The solver will likely implement Stam’s stable fluids algorithm:
*	**Advection**: move quantities (velocity and optionally dye) through the velocity field
*	**Diffusion/Viscosity**: simulate viscosity by diffusing velocity (this involves solving a linear system, presumably via Jacobi iteration)
*	**External Forces**: apply user input forces (e.g. stirring motions) or gravity if needed
*	**Pressure Projection**: solve a Poisson equation for pressure to enforce incompressibility (making the velocity field divergence-free)

Each major step of the solver will be parallelized. CUDA kernels will update all cells in parallel for operations like advection and diffusion. For the pressure solve, Jacobi iteration will run on the GPU, possibly in a loop until convergence each frame. Memory transfers between CPU and GPU will be minimized by keeping the simulation data on the GPU and only copying over what’s needed for display. 

SDL2 will handle creating the window and drawing the visualization. It will also capture user input events (keyboard, mouse). For example, pressing the mouse to add colored dye or smoke into the fluid at a location or dragging the mouse to add a force (like stirring the fluid).

**Technical Constraints**
*	The simulation is 2D only
*	Visualization will likely be a simple 2D smoke or dye field drawn with color.
*	I won’t simulate temperature or buoyancy in depth
*	The project is limited to environments where an NVIDIA GPU is available (since CUDA is NVIDIA-specific)
*	We assume adequate GPU memory for the simulation

The final system will be a real-time interactive fluid simulation application with visual output. When the program is run, it will open an SDL2 window displaying a representation of the fluid (for example, a field of smoke or colored dye within a black background). The fluid will initially be still, but the user can interact to create interesting motion, In essence, the final program is like a sandbox tool where one can "paint" with fluids. From a technical perspective, behind the scenes each frame the application will call the CUDA kernels to update the fluid and then render the updated state via SDL. But from the user’s perspective, it’s a seamless interactive fluid simulation toy that is fascinating to watch and play with. By the end of the project, the system should be robust (running for extended periods without instability), and real-time (at least 30 FPS, ideally 60 FPS) for a reasonable grid size (to be determined by performance tests).

![image](https://gits-15.sys.kth.se/eliasnys/fluidProject/assets/19526/808e857a-edb7-4388-bc75-9729cbde3b2f)

Figure 1 Final result with colored “dye” (Harris, 2004)

## Results
See video folder
## References
Harris, M. J. (2004). Chapter 38. Fast Fluid Dynamics Simulation on the GPU. Retrieved from Nvidia Developer: https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu

Stam, J. (1999). Stable Fluids. Madison: University of Wisconsin-Madison.

Wikipedia. (2025, May 6). WikiPedia. Retrieved from CUDA: https://en.wikipedia.org/wiki/CUDA#


