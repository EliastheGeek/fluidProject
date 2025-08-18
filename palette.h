#pragma once
#include <glm/glm.hpp>

using glm::vec3;
// Convert scalar [0,1] to HSV-based RGB
inline vec3 hsv2rgb(float h, float s, float v)
{
    h = fmodf(h, 1.0f) * 6.0f;
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h, 2.0f) - 1.0f));
    vec3 rgb;
    if (h < 1.0f) rgb = vec3(c, x, 0);
    else if (h < 2.0f) rgb = vec3(x, c, 0);
    else if (h < 3.0f) rgb = vec3(0, c, x);
    else if (h < 4.0f) rgb = vec3(0, x, c);
    else if (h < 5.0f) rgb = vec3(x, 0, c);
    else rgb = vec3(c, 0, x);
    float m = v - c;
    return rgb + vec3(m);
}

inline vec3 palette(uint8_t val)
{
    float t = val / 255.0f;
    float h = 0.95f - 0.95f * t;
    return hsv2rgb(h, 1.0f, sqrtf(t));
}
