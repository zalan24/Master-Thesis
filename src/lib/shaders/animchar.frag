#version 330 core

uniform vec3 lightDir;
uniform vec3 lightColor;
uniform vec3 ambientColor;

in vec2 texcoord;
in vec3 normal;

void main()
{
    float cosTerm = clamp(dot(normal, -lightDir), 0, 1);
    vec3 diffuse = vec3(texcoord, 0) * (cosTerm * lightColor + ambientColor);
    gl_FragColor = vec4(diffuse, 1);
}
