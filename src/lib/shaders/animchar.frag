#version 330 core

uniform vec3 lightDir;
uniform vec3 lightColor;
uniform vec3 ambientColor;
uniform float alphaClipping;

in vec2 texcoord;
in vec3 normal;
in vec3 color;

uniform sampler2D diffuse_tex;

void main()
{
    vec4 diffTex = texture( diffuse_tex, texcoord );

    vec3 albedo = clamp(diffTex.rgb + color, vec3(0), vec3(1));
    float alpha = diffTex.a;

    if (alpha < alphaClipping)
        discard;

    float cosTerm = clamp(dot(normal, -lightDir), 0, 1);
    vec3 diffuse = albedo * (cosTerm * lightColor + ambientColor);
    gl_FragColor = vec4(diffuse, 1);
}
