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

struct WorldMapCityLabelVertex {
  float4 anchor;
  float4 cornerUV;
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
  // Bend the local tangent map onto an Earth-radius sphere centred beneath
  // the viewer. At walking altitude this is indistinguishable from the old
  // metre-scale plane; from high altitude it exposes the curved horizon.
  constexpr float earthRadius = 6378137.0f;
  float2 horizontalDelta = input.position.xz - uniforms.cameraScene.xz;
  float arcLength = length(horizontalDelta);
  float2 radialDirection = arcLength > 0.01f
    ? horizontalDelta / arcLength
    : float2(1.0f, 0.0f);
  float angle = min(arcLength / earthRadius, M_PI_F * 0.95f);
  float sinAngle = sin(angle);
  float cosAngle = cos(angle);
  float horizontalScale = arcLength > 0.01f
    ? sinAngle * earthRadius / arcLength
    : 1.0f;
  float3 scenePosition = float3(
    uniforms.cameraScene.x + horizontalDelta.x * horizontalScale,
    input.position.y + uniforms.verticalOffset + earthRadius * (cosAngle - 1.0f),
    uniforms.cameraScene.z + horizontalDelta.y * horizontalScale);
  float3 worldPosition = worldMapSceneToWorld(scenePosition, uniforms);

  float3 radialUp = float3(
    sinAngle * radialDirection.x,
    cosAngle,
    sinAngle * radialDirection.y);
  float3 radialTangent = float3(
    cosAngle * radialDirection.x,
    -sinAngle,
    cosAngle * radialDirection.y);
  float3 crossTangent = float3(-radialDirection.y, 0.0f, radialDirection.x);
  float radialSlope = dot(input.normal.xz, radialDirection);
  float crossSlope = dot(input.normal.xz, crossTangent.xz);
  float3 curvedNormal = normalize(
    radialSlope * radialTangent + crossSlope * crossTangent + input.normal.y * radialUp);

  WorldMapVertexOut out;
  out.clipPosition = viewProjectionMatrices[viewIndex] * float4(worldPosition, 1.0f);
  out.normal = normalize((uniforms.navigationInverse * float4(curvedNormal, 0.0f)).xyz);
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
  // Aerial perspective blends the planetary streaming edge into the horizon.
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
  float3 skyScene = uniforms.cameraScene.xyz + direction * 12000000.0f;
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

struct WorldMapCompassOut {
  float4 clipPosition [[position]];
  float2 uv;
};

// A camera-centred horizontal compass in map space. Its centre follows the
// player, while its axes remain locked to geographic north/east.
vertex WorldMapCompassOut worldMapCompassVertex(
  ushort amplificationID [[amplification_id]],
  const device float4 *vertices [[buffer(0)]],
  constant WorldMapUniforms &uniforms [[buffer(1)]],
  constant float4x4 *viewProjectionMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  float4 input = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, max(uniforms.viewCount, 1u) - 1u);
  // input.xy are north/east offsets. Scene +x is east and -z is north, so a
  // viewer initially facing forward sees north ahead and east on the right.
  float3 scenePosition = uniforms.cameraScene.xyz + float3(input.y, 70.0f, -input.x);

  WorldMapCompassOut out;
  out.clipPosition = viewProjectionMatrices[viewIndex]
    * float4(worldMapSceneToWorld(scenePosition, uniforms), 1.0f);
  out.uv = input.zw;
  return out;
}

fragment float4 worldMapCompassFragment(
  WorldMapCompassOut in [[stage_in]],
  bool frontFacing [[front_facing]],
  texture2d<float> compass [[texture(0)]])
{
  constexpr sampler compassSampler(
    coord::normalized,
    address::clamp_to_edge,
    filter::linear);
  // The compass is normally viewed from below. The quad's vertex/UV winding
  // already gives that back face the required orientation; flipping both axes
  // here rotated every cardinal direction by 180 degrees.
  float2 uv = frontFacing ? float2(1.0f - in.uv.x, 1.0f - in.uv.y) : in.uv;
  return compass.sample(compassSampler, uv);
}

struct WorldMapCityLabelOut {
  float4 clipPosition [[position]];
  float2 uv;
  float opacity;
};

vertex WorldMapCityLabelOut worldMapCityLabelVertex(
  ushort amplificationID [[amplification_id]],
  const device WorldMapCityLabelVertex *vertices [[buffer(0)]],
  constant WorldMapUniforms &uniforms [[buffer(1)]],
  constant float4x4 *viewProjectionMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  WorldMapCityLabelVertex input = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, max(uniforms.viewCount, 1u) - 1u);
  constexpr float earthRadius = 6378137.0f;
  float2 horizontalDelta = input.anchor.xz - uniforms.cameraScene.xz;
  float arcLength = length(horizontalDelta);
  float angle = min(arcLength / earthRadius, M_PI_F * 0.95f);
  float horizontalScale = arcLength > 0.01f
    ? sin(angle) * earthRadius / arcLength
    : 1.0f;
  float3 scenePosition = float3(
    uniforms.cameraScene.x + horizontalDelta.x * horizontalScale,
    input.anchor.y + uniforms.verticalOffset + earthRadius * (cos(angle) - 1.0f),
    uniforms.cameraScene.z + horizontalDelta.y * horizontalScale);
  float3 worldPosition = worldMapSceneToWorld(scenePosition, uniforms);
  float4 clipPosition = viewProjectionMatrices[viewIndex] * float4(worldPosition, 1.0f);
  clipPosition.xy += input.cornerUV.xy * clipPosition.w;

  WorldMapCityLabelOut out;
  out.clipPosition = clipPosition;
  out.uv = input.cornerUV.zw;
  out.opacity = 1.0f - smoothstep(
    uniforms.fogParams.x * 0.88f,
    uniforms.fogParams.y,
    length(scenePosition - uniforms.cameraScene.xyz));
  return out;
}

fragment float4 worldMapCityLabelFragment(
  WorldMapCityLabelOut in [[stage_in]],
  texture2d<float> labels [[texture(0)]])
{
  constexpr sampler labelSampler(
    coord::normalized,
    address::clamp_to_edge,
    filter::linear);
  return labels.sample(labelSampler, in.uv) * in.opacity;
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
