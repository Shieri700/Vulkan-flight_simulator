#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 3) uniform samplerCube skyboxSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    if(fragTexCoord.z == 0)
        outColor = texture(texSampler, fragTexCoord.xy);
    else
        outColor = texture(skyboxSampler, fragTexCoord);
}