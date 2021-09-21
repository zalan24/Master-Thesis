descriptor {
  variants {
  }
  resources {
  }
}

stages {
  attachments {
    outColor {
      type = output;
      channels = rgba;
      location = 0;
    }
  }
}

global {
  vec4 lightToColor(vec3 light) {
    return vec4(light, 1); // TODO
  }
}
