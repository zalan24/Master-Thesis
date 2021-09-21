include {
    aglobal;
}

descriptor {
    variants {
    }
    resources {
    }
}

stages {
    states {
        cull = none;
    }
    vs {
        entry = main;
    }
}

vs {
    vec2 positions[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2(-1.0, 3.0),
        vec2(3.0, -1.0)
    );

    layout(location = 0) out vec2 screenpos; // [-1,1]

    void main() {
        gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
        screenpos = gl_Position.xy;
    }
}

ps {
    layout(location = 0) in vec2 screenpos;

    // [0, 1]
    vec2 getTexcoord() {
        return screenpos*0.5 + 0.5;
    }
}
