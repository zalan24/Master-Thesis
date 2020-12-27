#version 330 core

uniform mat4 PVM;
uniform mat4 model;
uniform float interpolation;
in vec3 vPos;
in vec3 vNorm;
in vec3 vCol;
in vec3 vNewPos;

out vec3 color;
out vec3 normal;

void main()
{
    gl_Position = PVM * vec4(mix(vPos, vNewPos, interpolation), 1.0);
    normal = normalize((model * vec4(vNorm, 0.0)).xyz);
    color = vCol;
}
