#version 430 core

// TODO put in UBO
uniform mat4 PVM;
uniform mat4 model;
in vec3 vPos;
in vec3 vNorm;
in vec3 vColor;
in vec2 vTex;
in vec4 vBoneIds;
in vec4 vBoneWeights;

out vec2 texcoord;
out vec3 normal;
out vec3 color;

layout (std430, binding=0) readonly restrict buffer BonesBlock
{
  mat4 bones[];
};

void main()
{
  ivec4 boneIds = ivec4(vBoneIds);
  mat4 boneTm =
    bones[boneIds.x] * vBoneWeights.x +
    bones[boneIds.y] * vBoneWeights.y +
    bones[boneIds.z] * vBoneWeights.z +
    bones[boneIds.w] * vBoneWeights.w;
  gl_Position = PVM * (boneTm * vec4(vPos, 1.0));
  normal = normalize((model * (boneTm * vec4(vNorm, 0.0))).xyz);
  texcoord = vTex;
  color = vColor;
}
