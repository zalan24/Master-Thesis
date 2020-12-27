#version 330 core

uniform mat4 PVM;
uniform float interpolation;

in vec3 vPos;
in vec3 vNewPos;

void main()
{
    gl_Position = PVM * vec4(mix(vPos, vNewPos, interpolation), 1.0);
}
