#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(binding = 2) uniform sampler2D HeightMap;

layout(location = 0) in int inType;
layout(location = 1) in vec3 inColor;
layout(location = 2) in int inWidth;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragTexCoord;

void HeightMapGetPosAndUv(uint VertexIndex, inout vec3 Pos, inout vec2 Uv)
{
    if(inType == 0) {
        uvec2 VertexId;
        VertexId.y = VertexIndex / inWidth;
        VertexId.x = VertexIndex - VertexId.y * inWidth;

        Uv = vec2(VertexId) / vec2((inWidth-1), (inWidth-1));
        vec2 thisUv = Uv / 10.f;
        Pos.x = VertexId.y - (inWidth / 2.f);
	    Pos.z = VertexId.x - (inWidth / 2.f);
        float thisY = ((-texture(HeightMap, thisUv).r) * 10000.f) + 100.f;
        Pos.y = thisY;
    }
    else if(inType == 1){
        Pos = vec3(-512.f, -512.f, -512.f);
        fragTexCoord = vec3(-1.f, 1.f, -1.f);
    }
    else if(inType == 2){
        Pos = vec3(-512.f, -512.f, 512.f);
        fragTexCoord = vec3(-1.f, 1.f, 1.f);
    }
    else if(inType == 3){
        Pos = vec3(512.f, -512.f, 512.f);
        fragTexCoord = vec3(1.f, 1.f, 1.f);
    }
    else if(inType == 4){
        Pos = vec3(512.f, -512.f, -512.f);
        fragTexCoord = vec3(1.f, 1.f, -1.f);
    }
    else if(inType == 5){
        Pos = vec3(-512.f, 512.f, -512.f);
        fragTexCoord = vec3(-1.f, -1.f, -1.f);
    }
    else if(inType == 6){
        Pos = vec3(-512.f, 512.f, 512.f);
        fragTexCoord = vec3(-1.f, -1.f, 1.f);
    }
    else if(inType == 7){
        Pos = vec3(512.f, 512.f, 512.f);
        fragTexCoord = vec3(1.f, -1.f, 1.f);
    }
    else if(inType == 8){
        Pos = vec3(512.f, 512.f, -512.f);
        fragTexCoord = vec3(1.f, -1.f, -1.f);
    }
}

void main() {
    vec3 Pos;
    vec2 Uv;
    HeightMapGetPosAndUv(gl_VertexIndex, Pos, Uv);

    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(Pos, 1.0);
    //fragColor = inColor;
    if(inType == 0){
        fragTexCoord = vec3(Uv, 0.f);
    }
}