import Foundation
import simd

func check(_ condition: @autoclosure () -> Bool, _ message: String) {
  guard condition() else { fatalError(message) }
}

let locations: [(Double, Double)] = [
  (28.281051, 85.545404), (0, 0), (0, 1.40625), (45, -120),
  (80, 30), (-80, 30), (85, 179.8), (-85, -179.8),
]
var cases = 0
var maxLeaves = 0
var maxBytes = 0
let began = Date()
for (latitude, longitude) in locations {
  let reference = MapSceneReference(latitude: latitude, longitude: longitude, geometryZoom: 8)
  for clearance: Float in [30, 1600, 20_000, 500_000, 2_000_000] {
    let root = MapStreamingPolicy.rootZoom(reference: reference, clearance: clearance)
    for offset: Float in [0, 12_345] {
      let camera = SIMD3<Float>(offset, 0, -offset)
      let leaves = MapStreamingPolicy.select(
        camera: camera, clearance: clearance, reference: reference)
      check(!leaves.isEmpty, "No coverage at \(latitude),\(longitude)")
      check(
        leaves.count <= MapStreamingPolicy.maximumLeaves,
        "Leaf budget exceeded at \(latitude),\(longitude), clearance=\(clearance), offset=\(offset): \(leaves.count)")
      let bytes = MapStreamingPolicy.estimatedBytes(leaves: leaves, rootZoom: root)
      check(bytes <= MapStreamingPolicy.workingSetBudget, "Working set exceeds upload headroom")
      maxLeaves = max(maxLeaves, leaves.count)
      maxBytes = max(maxBytes, bytes)
      let sorted = leaves.sorted(by: MapStreamingPolicy.ordered)
      for (i, a) in sorted.enumerated() {
        for b in sorted.dropFirst(i + 1) {
          check(!MapStreamingPolicy.contains(a, b), "Overlapping parent/child")
          if MapStreamingPolicy.sharesEdge(a, b) { check(abs(a.z - b.z) <= 1, "Unbalanced edge") }
        }
      }
      // Sample the entire streaming disk, including tile boundaries. Exclude
      // coordinates outside the Mercator dataset rather than silently wrapping.
      for dx in stride(from: Float(-80_000), through: 80_000, by: 20_000) {
        for dz in stride(from: Float(-80_000), through: 80_000, by: 20_000) {
          guard hypot(dx, dz) < MapStreamingPolicy.viewDistance(clearance: clearance) else {
            continue
          }
          let x = camera.x + dx
          let z = camera.z + dz
          let coordinate = reference.coordinate(sceneX: x, sceneZ: z)
          guard abs(coordinate.latitude) < 85.05112878, abs(coordinate.longitude) < 180 else {
            continue
          }
          let pointTile = reference.tile(atSceneX: x, sceneZ: z, zoom: 15)
          check(
            leaves.contains { MapStreamingPolicy.contains($0, pointTile) },
            "Hole inside streaming disk")
        }
      }
      let wideRadius = MapStreamingPolicy.viewDistance(clearance: clearance) * 0.85
      for direction in [
        SIMD2<Float>(1, 0), SIMD2<Float>(-1, 0), SIMD2<Float>(0, 1), SIMD2<Float>(0, -1),
        SIMD2<Float>(0.707, 0.707), SIMD2<Float>(-0.707, 0.707),
        SIMD2<Float>(0.707, -0.707), SIMD2<Float>(-0.707, -0.707),
      ] {
        let x = camera.x + direction.x * wideRadius
        let z = camera.z + direction.y * wideRadius
        let coordinate = reference.coordinate(sceneX: x, sceneZ: z)
        guard abs(coordinate.latitude) < 85.05112878, abs(coordinate.longitude) < 180 else {
          continue
        }
        let pointTile = reference.tile(atSceneX: x, sceneZ: z, zoom: 15)
        check(
          leaves.contains { MapStreamingPolicy.contains($0, pointTile) },
          "Hole near high-altitude horizon")
      }
      if latitude == 28.281051 && clearance == 1600 && offset == 0 {
        print(
          "Default: \(leaves.count) leaves, \(MapStreamingPolicy.wanted(leaves: leaves, rootZoom: root).count) tiles including ancestors, conservative \(Double(bytes) / 1_048_576) MiB"
        )
      }
      cases += 1
    }
  }
}
let reference = MapSceneReference(latitude: 28.281051, longitude: 85.545404, geometryZoom: 8)
let east = reference.scenePosition(latitude: reference.latitude, longitude: reference.longitude + 0.1)
let west = reference.scenePosition(latitude: reference.latitude, longitude: reference.longitude - 0.1)
let north = reference.scenePosition(latitude: reference.latitude + 0.1, longitude: reference.longitude)
let south = reference.scenePosition(latitude: reference.latitude - 0.1, longitude: reference.longitude)
check(east.x > 0 && abs(east.y) < 1, "East is not on scene +x / viewer right")
check(west.x < 0 && abs(west.y) < 1, "West is not on scene -x / viewer left")
check(north.y < 0 && abs(north.x) < 1, "North is not on scene -z / viewer forward")
check(south.y > 0 && abs(south.x) < 1, "South is not on scene +z / viewer rear")
for scene in [east, west, north, south] {
  let coordinate = reference.coordinate(sceneX: scene.x, sceneZ: scene.y)
  let reconstructed = reference.scenePosition(
    latitude: coordinate.latitude, longitude: coordinate.longitude)
  check(simd_distance(scene, reconstructed) < 0.1, "Scene/geographic direction round trip failed")
}
check(!MapCityCatalog.labels.isEmpty, "City label catalog is empty")
check(
  MapCityCatalog.labels.allSatisfy { $0.name.unicodeScalars.allSatisfy(\.isASCII) },
  "City label catalog contains non-pinyin glyphs")
for (index, city) in MapCityCatalog.labels.enumerated() {
  for other in MapCityCatalog.labels.dropFirst(index + 1) {
    check(
      MapCityCatalog.distance(city, other) >= MapCityCatalog.minimumSpacingMeters,
      "City labels violate 10 km spacing")
  }
}
let low = MapStreamingPolicy.select(camera: .zero, clearance: 30, reference: reference)
let high = MapStreamingPolicy.select(camera: .zero, clearance: 20_000, reference: reference)
check((low.map(\.z).max() ?? 0) > (high.map(\.z).max() ?? 0), "Clearance does not control LOD")
check(
  low == MapStreamingPolicy.select(camera: .zero, clearance: 30, reference: reference),
  "Nondeterministic LOD")
check(
  MapStreamingPolicy.viewDistance(clearance: 30) == MapStreamingPolicy.minimumViewDistance,
  "Low-altitude view distance changed")
check(
  MapStreamingPolicy.viewDistance(clearance: 2_000_000)
    == MapStreamingPolicy.maximumViewDistance,
  "High-altitude horizon does not reach the planetary cap")

// Descending must never reduce the detail directly beneath the camera.
for camera in [
  SIMD3<Float>(0, 0, 0), SIMD3<Float>(12_345, 0, -8_765),
  SIMD3<Float>(-24_680, 0, 17_530),
] {
  var previousZoom = 0
  for clearance: Float in [20_000, 5_000, 1_600, 800, 200, 30] {
    let leaves = MapStreamingPolicy.select(
      camera: camera, clearance: clearance, reference: reference)
    let localZoom = leaves
      .filter { MapStreamingPolicy.distance(to: $0, camera: camera, reference: reference) == 0 }
      .map(\.z).max() ?? 0
    check(
      localZoom >= previousZoom,
      "Descending reduced camera-centred terrain detail at camera=\(camera), clearance=\(clearance): \(previousZoom) -> \(localZoom), leaves=\(leaves.count), bytes=\(MapStreamingPolicy.estimatedBytes(leaves: leaves, rootZoom: MapStreamingPolicy.rootZoom(reference: reference, clearance: clearance)))")
    previousZoom = localZoom
  }
  check(previousZoom == MapStreamingPolicy.maximumZoom, "Low-altitude centre is not maximum LOD")
}

// Keep a textured parent visible until all four children have imagery, then
// replace the region in one frame without exposing terrain-only green tiles.
let parent = MapTileID(z: 10, x: 100, y: 200)
let children = Set([
  MapTileID(z: 11, x: 200, y: 400), MapTileID(z: 11, x: 201, y: 400),
  MapTileID(z: 11, x: 200, y: 401), MapTileID(z: 11, x: 201, y: 401),
])
let loaded = children.union([parent])
var textured = Set([parent])
for child in children.dropFirst() { textured.insert(child) }
let waitingCoverage = MapStreamingPolicy.presentationCoverage(
  leaves: children, imageryRequired: true,
  isLoaded: { loaded.contains($0) }, hasImagery: { textured.contains($0) })
check(waitingCoverage == [parent], "Incomplete imagery replaced its textured parent")
textured.formUnion(children)
let readyCoverage = MapStreamingPolicy.presentationCoverage(
  leaves: children, imageryRequired: true,
  isLoaded: { loaded.contains($0) }, hasImagery: { textured.contains($0) })
check(readyCoverage == children, "Complete child imagery did not replace its parent atomically")

let tangentCamera = SIMD3<Float>(120, 400, -80)
let directlyBelow = MapGlobeProjection.curvedPosition(
  SIMD3<Float>(120, 25, -80), camera: tangentCamera)
check(directlyBelow == SIMD3<Float>(120, 25, -80), "Globe projection moved its tangent point")
let quarterArc = MapGlobeProjection.curvedPosition(
  SIMD3<Float>(120 + MapGlobeProjection.radius * .pi / 2, 0, -80), camera: tangentCamera)
check(
  abs(quarterArc.x - (120 + MapGlobeProjection.radius)) < 2
    && abs(quarterArc.y + MapGlobeProjection.radius) < 2,
  "Globe projection does not follow Earth-radius geometry")

var state = MapNavigationState()
var raw = matrix_identity_float4x4
raw.columns.3 = SIMD4(4, 2, -8, 1)
let first = state.transform(raw, speed: 250)
let switched = state.transform(raw, speed: 2000)
check(first.columns.3 == switched.columns.3, "Speed switch teleported")
raw.columns.3.x += 1
let moved = state.transform(raw, speed: 2000)
check(moved.columns.3.x - switched.columns.3.x == 2000, "New movement did not use new speed")
let worldCamera = SIMD4<Float>(3, 4, 5, 1)
let sceneCamera = moved * worldCamera
check(
  simd_distance(simd_inverse(moved) * sceneCamera, worldCamera) < 0.01,
  "Camera/shader transforms disagree")

let identity = matrix_identity_float4x4
check(
  MapFrustum.isVisible(
    minimum: SIMD3(-0.5, -0.5, 0.1), maximum: SIMD3(0.5, 0.5, 0.9), clipFromScene: [identity]),
  "Visible box culled")
check(
  !MapFrustum.isVisible(
    minimum: SIMD3(2, -0.5, 0.1), maximum: SIMD3(3, 0.5, 0.9), clipFromScene: [identity]),
  "Offscreen box not culled")
var rightEye = identity
rightEye.columns.3.x = -2
check(
  MapFrustum.isVisible(
    minimum: SIMD3(2, -0.5, 0.1), maximum: SIMD3(3, 0.5, 0.9), clipFromScene: [identity, rightEye]),
  "Right-eye-only tile culled")
check(
  MapFrustum.isVisible(
    minimum: SIMD3(-0.5, -0.5, -0.5), maximum: SIMD3(0.5, 0.5, 0.5), clipFromScene: [identity]),
  "Near-plane crossing culled")

var retry = MapRetryState()
for attempt in 1...10 {
  let completion = Double(attempt * 40)
  retry.failed(at: completion)
  check(
    retry.nextAttempt > completion && retry.nextAttempt <= completion + 60,
    "Invalid completion-based backoff")
}
check(retry.nextAttempt.isFinite, "Failures permanently disabled retries")
print(
  "PASS: \(cases) geographic/altitude cases, max \(maxLeaves) leaves, max \(Double(maxBytes) / 1_048_576) MiB; navigation, stereo culling, retries. \(Date().timeIntervalSince(began)) s"
)
testNetworkAndTextures()
