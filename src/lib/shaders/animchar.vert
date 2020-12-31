#version 330 core

uniform mat4 PVM;
uniform mat4 model;
in vec3 vPos;
in vec3 vNorm;
in vec2 vTex;

out vec2 texcoord;
out vec3 normal;

void main()
{
    gl_Position = PVM * vec4(vPos, 1.0);
    normal = normalize((model * vec4(vNorm, 0.0)).xyz);
    texcoord = vTex;
}
