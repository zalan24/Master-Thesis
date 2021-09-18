include {
    global;
}

descriptor {
  variants {
  renderPass:
    color_pass, depth_pass, shadow_pass;
  }
  resources {
    vec3 ambientLight;
    vec3 sunLight;
    vec3 sunDir;
  }
}

stages {
  ps {
#if renderPass == color_pass
    use ambientLight;
    use sunLight;
    use sunDir;
#elif renderPass == shadow_pass
    use sunDir;
#endif
  }
  vs {
  }
}

ps {
#if renderPass == color_pass
  vec3 shadePixel(vec3 worldPos, vec3 normal, vec3 albedo) {
    vec3 luminance = PushConstants.ambientLight;
    float NdotL = abs(-dot(PushConstants.sunDir, normal));
    luminance += NdotL * sunLight;
    return luminance * albedo;
  }
#endif
}