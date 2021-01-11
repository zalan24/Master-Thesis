#version 430 core

uniform mat4 PVM;
uniform mat4 model;
in vec3 vPos;
in vec3 vNorm;
in vec3 vColor;
in vec2 vTex;
in ivec4 vBoneIds;
in vec4 vBoneWeights;

out vec2 texcoord;
out vec3 normal;
out vec3 color;

buffer BonesBlock
{
  mat4 bones[];
};

void main()
{
    mat4 boneTm =
        BonesBlock.bones[vBoneIds.x] * vBoneWeights.x +
        BonesBlock.bones[vBoneIds.y] * vBoneWeights.y +
        BonesBlock.bones[vBoneIds.z] * vBoneWeights.z +
        BonesBlock.bones[vBoneIds.w] * vBoneWeights.w;
    gl_Position = PVM * (boneTm * vec4(vPos, 1.0));
    normal = normalize((model * (boneTm * vec4(vNorm, 0.0))).xyz);
    texcoord = vTex;
    color = vColor;
}
