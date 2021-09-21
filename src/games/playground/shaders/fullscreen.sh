include {
    aglobal;
}

descriptor {
    variants {
    }
    resources {
        vec3 cameraUp;
        vec3 cameraDir;
        vec3 topleftViewVec;
    }
}

stages {
    states {
        cull = none;
    }
    vs {
        entry = main;
        use cameraUp;
        use cameraDir;
        use topleftViewVec;
    }
}

vs {
    vec2 positions[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2(-1.0, 3.0),
        vec2(3.0, -1.0)
    );

    layout(location = 0) out vec2 screenpos; // [-1,1]
    layout(location = 1) out vec3 viewVec;

    void main() {
        gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
        screenpos = gl_Position.xy;

        vec3 sideDir = cross(PushConstants.cameraUp, PushConstants.cameraDir);

        vec3 viewVec00 = PushConstants.topleftViewVec;
        vec3 viewVec01 = reflect(viewVec00, sideDir);
        vec3 viewVec10 = reflect(viewVec00, -PushConstants.cameraUp);
        vec3 viewVec11 = reflect(-viewVec00, PushConstants.cameraDir);

        vec2 tc = screenpos * 0.5 + 0.5;

        viewVec = mix(
            mix(viewVec00, viewVec01, tc.x),
            mix(viewVec10, viewVec11, tc.x),
            tc.y
        );
    }
}

ps {
    layout(location = 0) in vec2 screenpos;
    layout(location = 1) in vec3 viewVec;

    // [0, 1]
    vec2 getTexcoord() {
        return screenpos*0.5 + 0.5;
    }

    vec3 getViewVec() {
        return normalize(viewVec);
    }
}
