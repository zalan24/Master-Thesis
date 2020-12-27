#version 330 core

uniform vec3 lightDir;
uniform vec3 lightColor;
uniform vec3 ambientColor;
uniform int numData;
uniform int colorMode;

in vec2 tex;
in vec3 pos;
in vec3 side;

void main()
{
    vec3 normal = vec3(0, 1, 0);
    // vec3 color = vec3(floor((tex.x + tex.y)*numData));
    vec3 color = vec3(1, 1, 1);
    if (colorMode == 1) {
        color = pos/2 + 0.5;
    } else {
        vec2 t = tex*numData;
        t = fract(t);
        if (t.x+t.y <= 1)
            color = side;
    }
    float cosTerm = clamp(dot(normal, -lightDir), 0, 1);
    vec3 diffuse = color * (cosTerm * lightColor + ambientColor);
    gl_FragColor = vec4(diffuse, 1);
}
