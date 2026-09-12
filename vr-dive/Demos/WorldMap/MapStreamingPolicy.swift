import Foundation
import simd

/// Pure selection and resource accounting, shared by the renderer and regression tests.
nonisolated enum MapStreamingPolicy {
  static let maximumZoom = 15
  // A small reserve above the old 140 cap prevents budget balancing from
  // collapsing the camera-centred branch at tile boundaries and high latitudes.
  static let maximumLeaves = 160
  static let minimumViewDistance: Float = 90_000
  // Covers the horizon from roughly 2,000 km above the surface.
  static let maximumViewDistance: Float = 5_500_000
  static let gpuBudget = 256 * 1024 * 1024
  // Leave room for replacement textures and six concurrent uploads.
  static let workingSetBudget = gpuBudget - 24 * 1024 * 1024

  static func span(zoom: Int, reference: MapSceneReference) -> Float {
    Float(2 * Double.pi * MapProjection.earthRadius * reference.cosLatitude)
      / Float(1 << zoom)
  }

  static func viewDistance(clearance: Float) -> Float {
    let altitude = max(clearance, 0)
    let horizon = sqrt(altitude * (2 * Float(MapProjection.earthRadius) + altitude))
    return min(max(minimumViewDistance, horizon * 1.08), maximumViewDistance)
  }

  static func rootZoom(reference: MapSceneReference, clearance: Float) -> Int {
    let distance = viewDistance(clearance: clearance)
    var zoom = 8
    while zoom > 0 && span(zoom: zoom, reference: reference) < distance { zoom -= 1 }
    return zoom
  }

  static func meshQuads(zoom: Int) -> Int {
    switch zoom {
    case 15: return 64
    case 14: return 48
    case 13: return 32
    case 12: return 24
    case 11: return 16
    case 10: return 12
    case 9: return 16
    default: return 32
    }
  }

  static func meshBytes(zoom: Int) -> Int {
    let q = meshQuads(zoom: zoom)
    return ((q + 1) * (q + 1) + 4 * q + 1) * 48 + (6 * q * q + 24 * q) * 2
  }

  static func textureBytes(bias: Int) -> Int {
    let side = 256 << bias
    return (side * side * 4 - 1) / 3 * 4
  }

  static func ordered(_ a: MapTileID, _ b: MapTileID) -> Bool {
    if a.z != b.z { return a.z < b.z }
    if a.y != b.y { return a.y < b.y }
    return a.x < b.x
  }

  static func contains(_ ancestor: MapTileID, _ tile: MapTileID) -> Bool {
    guard tile.z >= ancestor.z else { return false }
    let shift = tile.z - ancestor.z
    return tile.x >> shift == ancestor.x && tile.y >> shift == ancestor.y
  }

  static func wanted(leaves: Set<MapTileID>, rootZoom: Int) -> Set<MapTileID> {
    var result = leaves
    for leaf in leaves {
      var node = leaf
      while node.z > rootZoom {
        node = node.parent
        result.insert(node)
      }
    }
    return result
  }

  /// Chooses a gap-free presentation set while finer tiles are loading.
  /// When imagery is requested, a terrain-only child does not replace a
  /// textured ancestor. If any sibling is still incomplete, the ancestor wins
  /// for the whole region so the transition happens as one visual swap.
  static func presentationCoverage(
    leaves: Set<MapTileID>,
    imageryRequired: Bool,
    isLoaded: (MapTileID) -> Bool,
    hasImagery: (MapTileID) -> Bool
  ) -> Set<MapTileID> {
    var result = Set<MapTileID>()
    for leaf in leaves {
      var node = leaf
      var terrainFallback: MapTileID?
      while true {
        if isLoaded(node) {
          terrainFallback = terrainFallback ?? node
          if !imageryRequired || hasImagery(node) {
            result.insert(node)
            break
          }
        }
        guard node.z > 0 else {
          if let terrainFallback { result.insert(terrainFallback) }
          break
        }
        node = node.parent
      }
    }
    // A textured ancestor selected for one incomplete branch covers its ready
    // siblings as well. Remove those descendants to avoid overlap and z-fighting.
    for id in Array(result) {
      var ancestor = id
      while ancestor.z > 0 {
        ancestor = ancestor.parent
        if result.contains(ancestor) {
          result.remove(id)
          break
        }
      }
    }
    return result
  }

  static func estimatedBytes(leaves: Set<MapTileID>, rootZoom: Int) -> Int {
    wanted(leaves: leaves, rootZoom: rootZoom).reduce(0) {
      // Conservative: every fine leaf may acquire a 512px texture after a quality change.
      $0 + meshBytes(zoom: $1.z)
        + textureBytes(bias: leaves.contains($1) && $1.z >= 14 ? 1 : 0)
    }
  }

  static func distance(to id: MapTileID, camera: SIMD3<Float>, reference: MapSceneReference)
    -> Float
  {
    let center = MapProjection.tileCenterScenePosition(id: id, reference: reference)
    let half = span(zoom: id.z, reference: reference) / 2
    return hypot(
      max(abs(center.x - camera.x) - half, 0),
      max(abs(center.y - camera.z) - half, 0))
  }

  static func sharesEdge(_ a: MapTileID, _ b: MapTileID) -> Bool {
    let zoom = max(a.z, b.z)
    let ax = a.x << (zoom - a.z)
    let ay = a.y << (zoom - a.z)
    let bx = b.x << (zoom - b.z)
    let by = b.y << (zoom - b.z)
    let asize = 1 << (zoom - a.z)
    let bsize = 1 << (zoom - b.z)
    return ((ax + asize == bx || bx + bsize == ax) && max(ay, by) < min(ay + asize, by + bsize))
      || ((ay + asize == by || by + bsize == ay) && max(ax, bx) < min(ax + asize, bx + bsize))
  }

  private static func merge(_ parent: MapTileID, into leaves: inout Set<MapTileID>) {
    leaves = leaves.filter { !contains(parent, $0) }
    leaves.insert(parent)
  }

  /// Initially refine only edges with a gap greater than one level. After a
  /// budget merge, coarsen the fine side so balancing cannot undo the budget.
  static func balance(_ leaves: inout Set<MapTileID>, refining: Bool = false) {
    while true {
      let sorted = leaves.sorted(by: ordered)
      var mergeTarget: MapTileID?
      outer: for coarse in sorted {
        for fine in sorted where fine.z > coarse.z + 1 && sharesEdge(coarse, fine) {
          var parent = fine
          while parent.z > coarse.z + 1 { parent = parent.parent }
          mergeTarget = refining ? coarse : parent
          break outer
        }
      }
      guard let parent = mergeTarget else { return }
      if refining {
        leaves.remove(parent)
        for y in 0...1 {
          for x in 0...1 {
            leaves.insert(MapTileID(z: parent.z + 1, x: parent.x * 2 + x, y: parent.y * 2 + y))
          }
        }
      } else {
        merge(parent, into: &leaves)
      }
    }
  }

  static func select(camera: SIMD3<Float>, clearance: Float, reference: MapSceneReference) -> Set<
    MapTileID
  > {
    let distanceLimit = viewDistance(clearance: clearance)
    let root = rootZoom(reference: reference, clearance: clearance)
    let center = reference.tile(atSceneX: camera.x, sceneZ: camera.z, zoom: root)
    var leaves = Set<MapTileID>()
    func visit(_ id: MapTileID) {
      guard distance(to: id, camera: camera, reference: reference) <= distanceLimit else { return }
      let size = span(zoom: id.z, reference: reference)
      let middle = MapProjection.tileCenterScenePosition(id: id, reference: reference)
      if id.z < maximumZoom, size > max(80, clearance),
        hypot(middle.x - camera.x, middle.y - camera.z) < size * 1.1
      {
        for y in 0...1 {
          for x in 0...1 { visit(MapTileID(z: id.z + 1, x: id.x * 2 + x, y: id.y * 2 + y)) }
        }
      } else {
        leaves.insert(id)
      }
    }
    // A root spans at least the view radius, including at high latitudes.
    for y in (center.y - 1)...(center.y + 1) where y >= 0 && y < 1 << root {
      for x in (center.x - 1)...(center.x + 1) where x >= 0 && x < 1 << root {
        visit(MapTileID(z: root, x: x, y: y))
      }
    }
    balance(&leaves, refining: true)
    // Preserve the refinement selected directly beneath the viewer. Budget
    // balancing may coarsen distant regions, but descending must never make the
    // ground under the camera less detailed.
    let focusTile = reference.tile(
      atSceneX: camera.x, sceneZ: camera.z, zoom: maximumZoom)
    let protectedFocusZoom = leaves.first(where: { contains($0, focusTile) })?.z ?? root
    while leaves.count > maximumLeaves
      || estimatedBytes(leaves: leaves, rootZoom: root) > workingSetBudget
    {
      // Include all ancestors so a chain of solitary leaves cannot prevent reduction.
      let parents = wanted(leaves: leaves, rootZoom: root).subtracting(leaves)
      let candidates = parents.filter { parent in leaves.filter { contains(parent, $0) }.count >= 2
      }
      let orderedCandidates = candidates.sorted(by: {
          let da = distance(to: $0, camera: camera, reference: reference)
          let db = distance(to: $1, camera: camera, reference: reference)
          if da != db { return da > db }
          if $0.z != $1.z { return $0.z > $1.z }
          return ordered($0, $1)
        })
      var accepted: Set<MapTileID>?
      for parent in orderedCandidates {
        var proposal = leaves
        merge(parent, into: &proposal)
        balance(&proposal)
        let focusZoom = proposal.first(where: { contains($0, focusTile) })?.z ?? root
        if focusZoom >= protectedFocusZoom, proposal.count < leaves.count {
          accepted = proposal
          break
        }
      }
      guard let accepted else { break }
      leaves = accepted
    }
    return leaves
  }
}

/// Camera-centred tangent-sphere projection shared with the Metal vertex shader.
/// Keeping the tangent point under the viewer preserves terrain following while
/// still producing the correct Earth-radius horizon at regional/global scale.
nonisolated enum MapGlobeProjection {
  static let radius: Float = Float(MapProjection.earthRadius)

  static func curvedPosition(_ position: SIMD3<Float>, camera: SIMD3<Float>) -> SIMD3<Float> {
    let delta = SIMD2<Float>(position.x - camera.x, position.z - camera.z)
    let arcLength = simd_length(delta)
    guard arcLength > 0.01 else { return position }
    let angle = min(arcLength / radius, .pi * 0.95)
    let horizontalScale = sin(angle) * radius / arcLength
    return SIMD3<Float>(
      camera.x + delta.x * horizontalScale,
      position.y + radius * (cos(angle) - 1),
      camera.z + delta.y * horizontalScale)
  }
}

/// Scale only new movement, so changing flight tier preserves the current position.
nonisolated struct MapNavigationState {
  private var previous: SIMD3<Float>?
  private var translation = SIMD3<Float>.zero

  mutating func transform(_ raw: simd_float4x4, speed: Float) -> simd_float4x4 {
    let next = SIMD3<Float>(raw.columns.3.x, raw.columns.3.y, raw.columns.3.z)
    translation += (next - (previous ?? .zero)) * speed
    previous = next
    var result = raw
    result.columns.3 = SIMD4<Float>(translation, 1)
    return result
  }
}

nonisolated struct MapRetryState {
  private(set) var attempts = 0
  private(set) var nextAttempt: TimeInterval = 0

  mutating func failed(at completionTime: TimeInterval) {
    attempts = min(attempts + 1, 6)
    nextAttempt = completionTime + min(pow(2, Double(attempts)), 60)
  }
}

/// Safe to read from a URLSession worker while the renderer cancels obsolete work.
nonisolated final class MapLoadCancellation: @unchecked Sendable {
  private let lock = NSLock()
  private var cancelled = false
  var isCancelled: Bool {
    lock.lock()
    defer { lock.unlock() }
    return cancelled
  }
  func cancel() {
    lock.lock()
    cancelled = true
    lock.unlock()
  }
}

nonisolated enum MapFrustum {
  /// Metal clip space is -w...w in x/y and 0...w in z, including reverse-Z.
  /// A tile is rejected only when it lies outside every eye's frustum.
  static func isVisible(
    minimum: SIMD3<Float>, maximum: SIMD3<Float>, clipFromScene: [simd_float4x4]
  ) -> Bool {
    guard !clipFromScene.isEmpty else { return true }
    return clipFromScene.contains { matrix in
      var corners: [SIMD4<Float>] = []
      for x in [minimum.x, maximum.x] {
        for y in [minimum.y, maximum.y] {
          for z in [minimum.z, maximum.z] { corners.append(matrix * SIMD4(x, y, z, 1)) }
        }
      }
      return
        !(corners.allSatisfy { $0.x < -$0.w }
        || corners.allSatisfy { $0.x > $0.w }
        || corners.allSatisfy { $0.y < -$0.w }
        || corners.allSatisfy { $0.y > $0.w }
        || corners.allSatisfy { $0.z < 0 }
        || corners.allSatisfy { $0.z > $0.w })
    }
  }
}
