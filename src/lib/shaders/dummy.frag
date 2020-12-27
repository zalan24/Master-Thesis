#version 330 core

uniform vec3 lightDir;
uniform vec3 lightColor;
uniform vec3 ambientColor;

in vec3 color;
in vec3 normal;

void main()
{
    float cosTerm = clamp(dot(normal, -lightDir), 0, 1);
    vec3 diffuse = color * (cosTerm * lightColor + ambientColor);
    gl_FragColor = vec4(diffuse, 1);
}
