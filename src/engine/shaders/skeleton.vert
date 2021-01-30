#version 430 core

// TODO put in UBO
uniform mat4 PVM;

uniform vec3 colors[2];

in vec3 vPos;
in vec2 vId_Depth;

out vec3 color;

layout (std430, binding=0) readonly restrict buffer BonesBlock
{
  mat4 bones[];
};

void main()
{
  int boneId = int(vId_Depth.x);
  int depth = int(vId_Depth.y);
  mat4 boneTm = bones[boneId];
  gl_Position = PVM * (boneTm * vec4(vPos, 1.0));
  color = colors[depth % 2];
}
