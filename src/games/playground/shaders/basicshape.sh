include {
    aglobal;
}

descriptor {
    variants {
        Shape: shape_plane, shape_box, shape_sphere;
    }
    resources {
        uint resolution;
    }
}

stages {
    states {
#if Shape == shape_plane
        cull = none;
#elif Shape == shape_box
        cull = back;
#elif Shape == shape_sphere
        cull = back;
#endif
    }
    vs {
        use resolution;
    }
}

vs {
    struct VertexData {
        vec3 pos;
        vec3 normal;
        vec2 tc;
    };
#if Shape == shape_plane
    vec2 positions[6] = vec2[](
        vec2(-1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, 1.0),
        vec2(1.0, -1.0)
    );

    VertexData getVertexData(uint vertexId) {
        uint quadInd = vertexId / 6;
        vertexId = vertexId % 6;

        uvec2 quadPos = uvec2(quadInd % PushConstants.resolution, quadInd / PushConstants.resolution);
        VertexData ret;
        ret.pos.y = 0;
        ret.pos.xz = positions[vertexId];
        ret.pos.xz /= float(PushConstants.resolution);
        vec2 quadOffset = vec2(float(quadPos.x) + 0.5, float(quadPos.y) + 0.5) / float(PushConstants.resolution) * 2.0 - 1.0;
        ret.pos.xz += quadOffset;
        ret.normal = vec3(0, 1, 0);
        ret.tc = ret.pos.xz * 0.5 + 0.5;
        return ret;
    }
#elif Shape == shape_box
    vec2 positions[6] = vec2[](
        vec2(-1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, 1.0),
        vec2(1.0, -1.0)
    );

    VertexData getVertexData(uint vertexId) {
        uint quadsPerSide = PushConstants.resolution*PushConstants.resolution;
        uint quadInd = vertexId / 6;
        vertexId = vertexId % 6;
        uint sideId = quadInd / quadsPerSide;
        quadInd = quadInd % quadsPerSide;

        uvec2 quadPos = uvec2(quadInd % PushConstants.resolution, quadInd / PushConstants.resolution);
        VertexData ret;
        ret.pos.y = (sideId % 2) == 0 ? -1 : 1;
        vec2 basePos = positions[vertexId];
        if (abs(dot(basePos, vec2(1, 1))) > 0.5)
            basePos *= ret.pos.y;
        ret.pos.xz = basePos;
        ret.pos.xz /= float(PushConstants.resolution);
        vec2 quadOffset = vec2(float(quadPos.x) + 0.5, float(quadPos.y) + 0.5) / float(PushConstants.resolution) * 2.0 - 1.0;
        ret.pos.xz += quadOffset;
        ret.tc = ret.pos.xz * 0.5 + 0.5;
        ret.normal = vec3(0, ret.pos.y, 0);

        if ((sideId / 2) == 0) {
            // Along X
            ret.pos = ret.pos.yzx;
            ret.normal = ret.normal.yzx;
        } else if ((sideId / 2) == 1) {
            // Aling Y
            // nothing to do
        } else {
            // Aling Z
            ret.pos = ret.pos.zxy;
            ret.normal = ret.normal.zxy;
        }
        return ret;
    }
#elif Shape == shape_sphere
    vec2 positions[6] = vec2[](
        vec2(-1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, 1.0),
        vec2(1.0, -1.0)
    );

    VertexData getVertexData(uint vertexId) {
        uint quadInd = vertexId / 6;
        vertexId = vertexId % 6;

        uvec2 quadPos = uvec2(quadInd % PushConstants.resolution, quadInd / PushConstants.resolution);
        VertexData ret;
        ret.tc = positions[vertexId];
        ret.tc /= float(PushConstants.resolution);
        vec2 quadOffset = vec2(float(quadPos.x) + 0.5, float(quadPos.y) + 0.5) / float(PushConstants.resolution) * 2.0 - 1.0;
        ret.tc += quadOffset;

        vec3 dir;
        dir.xz = ret.tc;
        vec2 absolute = abs(dir.xz);
        dir.y = 1 - absolute.x - absolute.y;
        if (dir.y < 0)
            dir.xz = sign(dir.xz) * vec2(1.0f - absolute.y, 1.0f - absolute.x);

        ret.pos = normalize(dir);
        ret.normal = ret.pos;
        ret.tc = ret.tc * 0.5 + 0.5;
        return ret;
    }
#endif
}
