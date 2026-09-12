#include <metal_stdlib>
using namespace metal;

struct WorldMapUniforms {
  uint viewCount;
  uint pad0;
  float verticalOffset;
  uint pad1;
  float4 cameraScene;
  // x = fog start distance, y = fog end distance
  float4 fogParams;
  float4x4 navigationInverse;
};

struct WorldMapVertex {
  float3 position;
  float3 normal;
  float2 uv;
  float2 pad;
};

struct WorldMapVertexOut {
  float4 clipPosition [[position]];
  float3 normal;
  float2 uv;
  float terrainElevation;
  float3 scenePosition;
  uint viewIndex [[flat]];
};

struct WorldMapSkyOut {
  float4 clipPosition [[position]];
  float3 direction;
};

static float3 worldMapSceneToWorld(
  float3 scenePosition,
  constant WorldMapUniforms &uniforms)
{
  return (uniforms.navigationInverse * float4(scenePosition, 1.0f)).xyz;
}

vertex WorldMapVertexOut worldMapVertex(
  ushort amplificationID [[amplification_id]],
  const device WorldMapVertex *vertices [[buffer(0)]],
  constant WorldMapUniforms &uniforms [[buffer(1)]],
  constant float4x4 *viewProjectionMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  WorldMapVertex input = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, max(uniforms.viewCount, 1u) - 1u);
  float3 scenePosition = float3(
    input.position.x,
    input.position.y + uniforms.verticalOffset,
    input.position.z);
  float3 worldPosition = worldMapSceneToWorld(scenePosition, uniforms);

  WorldMapVertexOut out;
  out.clipPosition = viewProjectionMatrices[viewIndex] * float4(worldPosition, 1.0f);
  out.normal = normalize((uniforms.navigationInverse * float4(input.normal, 0.0f)).xyz);
  out.uv = input.uv;
  out.terrainElevation = input.position.y;
  out.scenePosition = scenePosition;
  out.viewIndex = viewIndex;
  return out;
}

fragment float4 worldMapFragment(
  WorldMapVertexOut in [[stage_in]],
  constant uint &hasTexture [[buffer(0)]],
  constant WorldMapUniforms &uniforms [[buffer(1)]],
  texture2d<float> satellite [[texture(0)]])
{
  constexpr sampler satelliteSampler(
    coord::normalized,
    address::clamp_to_edge,
    filter::linear,
    mip_filter::linear,
    max_anisotropy(4));
  const float3 lightDirection = normalize(float3(-0.42f, 0.80f, 0.36f));
  float3 normal = normalize(in.normal);
  float3 baseColor;

  if (hasTexture != 0u) {
    baseColor = satellite.sample(satelliteSampler, in.uv).rgb;
  } else {
    float lowland = saturate(in.terrainElevation / 5200.0f);
    baseColor = mix(float3(0.14f, 0.32f, 0.13f), float3(0.74f, 0.71f, 0.66f), lowland);
    baseColor = mix(
      baseColor,
      float3(0.92f, 0.95f, 0.97f),
      smoothstep(4600.0f, 5600.0f, in.terrainElevation));
  }

  float diffuse = max(dot(normal, lightDirection), 0.0f);
  float lighting = 0.58f + 0.52f * diffuse;
  float3 color = baseColor * lighting;
  // Aerial perspective: blend the far terrain into the sky horizon so the
  // 90 km streaming edge never reads as a hard cut.
  const float3 fogColor = float3(0.72f, 0.80f, 0.88f);
  float cameraDistance = length(in.scenePosition - uniforms.cameraScene.xyz);
  float fogAmount = smoothstep(uniforms.fogParams.x, uniforms.fogParams.y, cameraDistance);
  color = mix(color, fogColor, fogAmount);
  return float4(color, 1.0f);
}

vertex WorldMapSkyOut worldMapSkyVertex(
  ushort amplificationID [[amplification_id]],
  const device float3 *vertices [[buffer(0)]],
  constant WorldMapUniforms &uniforms [[buffer(1)]],
  constant float4x4 *viewProjectionMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  float3 direction = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, max(uniforms.viewCount, 1u) - 1u);
  // Keep a large dome centred on the camera so it never runs out as the player
  // travels. navigationInverse restores the pattern-rotation applied to the
  // scene while inv(nav) * cameraScene stays pinned to the camera in world.
  float3 skyScene = uniforms.cameraScene.xyz + direction * 100000.0f;
  float3 worldPosition = worldMapSceneToWorld(skyScene, uniforms);

  WorldMapSkyOut out;
  out.clipPosition = viewProjectionMatrices[viewIndex] * float4(worldPosition, 1.0f);
  out.direction = direction;
  return out;
}

fragment float4 worldMapSkyFragment(WorldMapSkyOut in [[stage_in]]) {
  float3 direction = normalize(in.direction);
  float vertical = saturate(direction.y * 0.5f + 0.5f);
  float3 horizon = float3(0.72f, 0.80f, 0.88f);
  float3 zenith = float3(0.24f, 0.45f, 0.78f);
  float3 color = mix(horizon, zenith, smoothstep(0.5f, 0.96f, vertical));
  return float4(color, 1.0f);
}

struct WorldMapOverlayOut {
  float4 clipPosition [[position]];
  float2 uv;
};

// Screen-space quad. Positions are supplied directly in normalized device
// coordinates so the attribution overlay ignores the scene camera entirely.
vertex WorldMapOverlayOut worldMapOverlayVertex(
  const device float4 *vertices [[buffer(0)]],
  uint vertexID [[vertex_id]])
{
  float4 input = vertices[vertexID];
  WorldMapOverlayOut out;
  out.clipPosition = float4(input.x, input.y, 0.0f, 1.0f);
  out.uv = float2(input.z, input.w);
  return out;
}

fragment float4 worldMapOverlayFragment(
  WorldMapOverlayOut in [[stage_in]],
  texture2d<float> overlay [[texture(0)]])
{
  constexpr sampler overlaySampler(
    coord::normalized,
    address::clamp_to_edge,
    filter::linear);
  return overlay.sample(overlaySampler, in.uv);
}
