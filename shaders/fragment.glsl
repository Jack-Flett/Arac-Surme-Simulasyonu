#version 330 core

in  vec3 vFragPos;
in  vec3 vNormal;
in  vec2 vUV;
out vec4 FragColor;

uniform vec3      uColor;
uniform sampler2D uTexture;
uniform int       uUseTexture;
uniform vec3      uEmissive;      // additive glow — (0,0,0) for normal geometry

// Directional light (dim evening sun)
uniform vec3 uLightDir;           // direction light travels (FROM source, into scene)
uniform vec3 uLightColor;
uniform vec3 uAmbient;
uniform vec3 uViewPos;

// Street-lamp point lights
#define MAX_LAMPS 36
uniform vec3 uLampPos  [MAX_LAMPS];
uniform vec3 uLampColor[MAX_LAMPS];
uniform int  uNumLamps;

void main()
{
    vec3 base;
    if (uUseTexture == 1)
        base = texture(uTexture, vUV).rgb;
    else
        base = uColor;

    vec3 N = normalize(vNormal);
    vec3 V = normalize(uViewPos - vFragPos);

    // ── Directional (sun) ────────────────────────────────────────────
    vec3  L  = normalize(-uLightDir);
    vec3  H  = normalize(L + V);
    float dS = max(dot(N, L), 0.0);
    float sS = pow(max(dot(N, H), 0.0), 32.0);

    vec3 colour = uAmbient * base
                + dS * uLightColor * base
                + sS * uLightColor * 0.2;

    // ── Street lamps (point lights) ──────────────────────────────────
    for (int i = 0; i < MAX_LAMPS; i++) {
        if (i >= uNumLamps) break;

        vec3  lv   = uLampPos[i] - vFragPos;
        float dist = length(lv);
        vec3  Lp   = lv / dist;
        // quadratic attenuation — meaningful range ≈ 18 units
        float att  = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist);
        vec3  Hp   = normalize(Lp + V);
        float dP   = max(dot(N, Lp), 0.0);
        float sP   = pow(max(dot(N, Hp), 0.0), 32.0);
        colour += att * (dP * uLampColor[i] * base + sP * uLampColor[i] * 0.5);
    }

    FragColor = vec4(colour + uEmissive, 1.0);
}
