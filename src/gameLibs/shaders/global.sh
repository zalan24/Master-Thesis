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

stages {
  ps {
#if renderPass == color
    use ambientLight;
    use sunLight;
    use sunDir;
#endif
  }
  vs {
    use viewProj;
#if someStuff == stuff1
    use cameraPos;
#endif
  }
  attachments {
    outColor {
      type = output;
      channels = rgba;
      location = 0;
    }
  }
}
