#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;

layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
    mat4 mvp;
    mat4 model;
    vec4 cameraPosition;
    vec4 lightDirection;
    vec4 lightColorIntensity;
    vec4 ambientColorIntensity;
    vec4 albedo;
    vec4 specularShininess;
    vec4 directionToCamera;
};

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    worldPosition = vec3(model * vec4(position, 1.0));
    worldNormal = transpose(inverse(mat3(model))) * normal;
}
