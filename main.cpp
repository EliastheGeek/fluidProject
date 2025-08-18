#include <iostream>
#include <algorithm>
#include <glm/glm.hpp>
#include "SDL2auxiliary.h"
#include "palette.h"
#include "fluid_simulator.h"

using glm::vec3;

//Application settings
const int SCREEN_WIDTH = 512;
const int SCREEN_HEIGHT = 512;

SDL2Aux* sdlAux;
FluidSimulator sim(SCREEN_WIDTH, SCREEN_HEIGHT);

void Draw();

int main(int argc, char* argv[])
{
    //Set up window/framebuffer and a basic timer
    sdlAux = new SDL2Aux(SCREEN_WIDTH, SCREEN_HEIGHT);
    int oldTime = SDL_GetTicks();
    float prevMouseX = 0, prevMouseY = 0;

    //Allocate GPU resources and clear the sim state
    sim.init();

    //Main loop
    while (!sdlAux->quitEvent()) {
        int newTime = SDL_GetTicks();
        float dt = (newTime - oldTime) / 1000.0f;//seconds since last frame

        //Input handling
        int mx, my;
        Uint32 mouseState = SDL_GetMouseState(&mx, &my);
        bool lMouseDown = mouseState & SDL_BUTTON(SDL_BUTTON_LEFT);

        //LMB: inject dye where the cursor is
        if (lMouseDown) {
            sim.injectDye(mx, my);
        }

        bool rMouseDown = mouseState & SDL_BUTTON(SDL_BUTTON_RIGHT);

        //RMB: apply a force based on mouse motion delta
        if (rMouseDown) {
            sim.applyForce(mx, my, mx - prevMouseX, my - prevMouseY);
        }
        
        prevMouseX = mx;
        prevMouseY = my;

        //Step the simulation and render the frame
        sim.update(dt);
        Draw();
        std::cout << 1000.f / (newTime - oldTime) << std::endl;

        oldTime = newTime;
    }
    return 0;
}

//Rendering
void Draw()
{
    //Clear the screen
    sdlAux->clearPixels();

    //Grab the latest dye buffer from the GPU and project convert its 
    // content to a RGB-value with the "palette"-function
    sim.copyDyeToHost();
    const uchar4* pixels = sim.getDyeHost();
    for (int y = 0; y < SCREEN_HEIGHT; ++y) {
        for (int x = 0; x < SCREEN_WIDTH; ++x) {
            uchar4 p = pixels[y * SCREEN_WIDTH + x];
            vec3 col = palette(p.x);
            sdlAux->putPixel(x, y, col);
        }
    }
    //Render to screen
    sdlAux->render();
}
