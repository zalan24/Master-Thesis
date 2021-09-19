include {
    aglobal;
}

descriptor {
  variants {
  }
  resources {
    mat4 viewProj;
    mat4 modelTm;
    vec3 eyePos;
  }
}

stages {
  vs {
    use viewProj;
    use modelTm;
    use eyePos;
  }
}

vs {
    vec4 getWorldSpace(vec4 modelSpace) {
      return PushConstants.modelTm * modelSpace;
    }
    vec4 getWorldSpacePoint(vec3 modelSpace) {
      return getWorldSpace(vec4(modelSpace, 1.0));
    }
    vec4 getWorldSpaceDirection(vec3 modelSpace) {
      return getWorldSpace(vec4(modelSpace, 0.0));
    }
    vec4 worldToScreenSpace(vec4 worldSpace) {
      return PushConstants.viewProj * worldSpace;
    }
    vec4 worldToScreenSpacePoint(vec3 worldSpace) {
      return worldToScreenSpace(vec4(worldSpace, 1.0));
    }
    vec4 modelToScreenSpace(vec4 modelSpace) {
      return worldToScreenSpace(getWorldSpace(modelSpace));
    }
    vec4 modelToScreenSpacePoint(vec3 modelSpace) {
      return modelToScreenSpace(vec4(modelSpace, 1.0));
    }
    vec3 getWorldNormal(vec3 modelNorm) {
      return normalize(getWorldSpaceDirection(modelNorm).xyz);
    }
}
