descriptor {
  variants {
  // testing stuff
  renderPass:
    color, depth, shadow;
  someStuff:
    stuff1, stuff2, stuff3;
  }
  resources {
    mat44 viewProj;
    vec3 cameraPos;
    vec3 ambientLight;
    vec3 sunLight;
    vec3 sunDir;
  }
}
