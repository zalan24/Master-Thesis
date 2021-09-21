include {
    aglobal;
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
  states {
    depthTest = true;
    depthWrite = true;
    depthCompare = less;
  }
  ps {
#if renderPass == color_pass
    use ambientLight;
    use sunLight;
    use sunDir;
#elif renderPass == shadow_pass
    use sunDir;
#endif
  }
}

ps {
#if renderPass == color_pass
  vec3 shadePixel(vec3 worldPos, vec3 normal, vec3 albedo) {
    vec3 luminance = PushConstants.ambientLight;
    float NdotL = clamp(-dot(PushConstants.sunDir, normal), 0.0, 1.0);
    luminance += NdotL * PushConstants.sunLight;
    return luminance * albedo;
  }
#endif
}