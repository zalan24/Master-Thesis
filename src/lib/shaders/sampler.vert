#version 330 core

#define M_PI 3.1415926535897932384626433832795

uniform mat4 PVM;
uniform float interpolation;
in vec3 vPos;
in vec2 vGridPos;
in vec3 vPolyPos;
in vec3 vSpherePos;
in vec3 vSide;

out vec2 tex;
out vec3 pos;
out vec3 side;

void main()
{
    vec3 gridPos = vec3(vGridPos - 0.5, 0);
    vec3 p;
    float interp = interpolation*3;
    if (interp <= 1) {
        p = mix(gridPos, vPolyPos, interp);
    } else if (interp <= 2) {
        p = mix(vPolyPos, vSpherePos, interp-1);
    } else {
        p = mix(vSpherePos, vPos, interp-2);
    }
    gl_Position = PVM * vec4(p, 1.0);
    tex = vGridPos;
    pos = vPos;
    side = vSide;
}
