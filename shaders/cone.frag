#version 450

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 0) out vec4 outColor;

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
    vec3 N = normalize(worldNormal);
    // lightDirection follows the rays; L points from the surface toward the light.
    vec3 L = normalize(-lightDirection.xyz);
    vec3 toCamera = cameraPosition.xyz - worldPosition;
    vec3 V = cameraPosition.w > 0.5 ?
        toCamera / max(length(toCamera), 0.00001) : normalize(directionToCamera.xyz);
    float diffuse = max(dot(N, L), 0.0);

    float specular = 0.0;
    vec3 halfVector = L + V;
    if (diffuse > 0.0 && dot(halfVector, halfVector) > 0.000001) {
        vec3 H = normalize(halfVector);
        specular = pow(max(dot(N, H), 0.0), specularShininess.w);
    }

    vec3 ambient = ambientColorIntensity.rgb * ambientColorIntensity.w * albedo.rgb;
    vec3 direct = lightColorIntensity.rgb * lightColorIntensity.w *
        (albedo.rgb * diffuse + specularShininess.rgb * specular);
    outColor = vec4(ambient + direct, 1.0);
}
