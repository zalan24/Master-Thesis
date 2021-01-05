#version 330 core

uniform vec3 lightDir;
uniform vec3 lightColor;
uniform vec3 ambientColor;

in vec2 texcoord;
in vec3 normal;
in vec3 color;

void main()
{
    float cosTerm = clamp(dot(normal, -lightDir), 0, 1);
    vec3 albedo = vec3(texcoord, 0) + color;
    vec3 diffuse = albedo * (cosTerm * lightColor + ambientColor);
    gl_FragColor = vec4(diffuse, 1);
}
