import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

nonisolated struct WorldMapUniforms {
  var viewCount: UInt32
  var pad0: UInt32
  var verticalOffset: Float
  var pad1: UInt32
  var cameraScene: SIMD4<Float>
  /// x = fog start distance, y = fog end distance.
  var fogParams: SIMD4<Float>
  var navigationInverse: simd_float4x4
}

/// Streaming satellite/terrain map. Unlike the one-off Gyirong reconstruction,
/// this renderer keeps a quadtree of real Web Mercator tiles around the camera
/// and swaps them as it moves, so the player can roam the whole world.
///
/// Terrain geometry comes from openly licensed AWS Terrain Tiles (Mapzen
/// Terrarium). Optionally, Google satellite imagery from the Map Tiles API is
/// draped on top. Google content is never written to disk and is only held in
/// memory while its tile is displayed, in line with the Map Tiles API policies.
final class WorldMapRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .worldMap
  let preferredClearColor = MTLClearColor(red: 0.55, green: 0.68, blue: 0.82, alpha: 1)

  /// Live readout sink, wired by the Renderer to the pattern coordinator.
  var statusSink: ((String) -> Void)?
  var coordinateSink: ((MapCoordinate) -> Void)?
  private var highDetailRadius = 1
  private var imageryEnabled = true

  private static let skirtDepth: Float = 45
  private static let maximumConcurrentLoads = 6
  private static let maximumPendingLoads = 400
  private static let maximumLeaves = 140
  private static let gridUpdateInterval: Float = 0.22
  private static let tileRetainSeconds: Float = 3.0

  // Quadtree LOD: root spans z8; leaves refine toward z15 near the camera.
  private static let minimumZoom = 8
  private static let maximumZoom = 15
  private static let quadtreeLevels = 7
  private static let rootZoom = maximumZoom - quadtreeLevels
  private static let maxViewDistance: Float = 90_000
  private static let splitDistanceFactor: Float = 1.1
  /// A tile is only refined while it is at least this multiple of the camera's
  /// altitude in size, so flying high does not waste detail right beneath you.
  private static let minimumVisibleSpanFactor: Float = 1.0

  private static let highDetailZoomThreshold = 14
  private static let gpuBudgetBytes = 256 * 1024 * 1024
  private static let minimumClearance: Float = 30
  private static let maximumClearance: Float = 20_000
  private static let clearanceResponse: Float = 10

  private static let startLatitude = 28.281_051
  private static let startLongitude = 85.545_404
  private static let startGroundEstimate: Float = 2_800
  private static let startAltitude: Float = 1_600

  private struct PendingLoad {
    let id: MapTileID
    let imageryZoomBias: Int
    let imageryEnabled: Bool
    let generation: Int
    let requestedAt: Float
  }

  private let device: MTLDevice
  private var reference: MapSceneReference
  private let google = GoogleMapsTileClient()
  private let terrainPipeline: MTLRenderPipelineState
  private let skyPipeline: MTLRenderPipelineState
  private let opaqueDepthState: MTLDepthStencilState
  private let skyDepthState: MTLDepthStencilState
  private let skyVertexBuffer: MTLBuffer
  private let skyIndexBuffer: MTLBuffer
  private let skyIndexCount: Int
  private let placeholderTexture: MTLTexture
  private let mipmapQueue: MTLCommandQueue
  private let overlayPipeline: MTLRenderPipelineState
  private let overlayDepthState: MTLDepthStencilState
  private let overlayTexture: MTLTexture
  private let overlayTextureAspect: Float

  private let loadQueue: OperationQueue
  private let lock = NSLock()
  private var tiles: [MapTileID: WorldMapTile] = [:]
  private var readyTiles: [WorldMapTile] = []
  private var pendingLoads: [PendingLoad] = []
  private var inFlight: Set<MapTileID> = []
  private var retryCounts: [MapTileID: Int] = [:]
  private var retryAfter: [MapTileID: Float] = [:]
  private var tileLastUsed: [MapTileID: Float] = [:]
  private var currentLeaves: Set<MapTileID> = []
  private var gridGeneration = 0
  private var lastGridUpdateTime: Float = -1_000
  private var lastGridCamera: SIMD3<Float>?
  private let gridQueue = DispatchQueue(
    label: "vr-dive.worldmap.grid", qos: .userInitiated)
  private var isSolvingGrid = false
  private var solvedLeaves: Set<MapTileID>?
  private var lastSolveRequestTime: Float = -1_000
  private var verticalOffset: Float
  private var desiredClearance: Float
  private var clearanceBaselineY: Float = 0
  private var hasClearanceBaseline = false
  private var lastFrameTime: Float = -1
  private var didLogConfiguration = false
  private var navigationSpeedScale: Float = 250
  private var mapFlightTier: MapFlightTier = .cruise
  private var lastStatusTime: Float = -1
  private var lastRelocateGeneration: Int = -1
  private var lastGroundAltitude: Float = 0
  private var lastGroundZoom: Int = 0
  private var residentBytes: Int = 0

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.device = device
    self.reference = MapSceneReference(
      latitude: Self.startLatitude,
      longitude: Self.startLongitude,
      geometryZoom: Self.rootZoom)
    self.verticalOffset = -(Self.startGroundEstimate + Self.startAltitude)
    self.desiredClearance = Self.startAltitude

    terrainPipeline = try Self.makeRenderPipeline(
      device: device,
      library: library,
      vertexName: "worldMapVertex",
      fragmentName: "worldMapFragment",
      maxViewCount: maxViewCount)
    skyPipeline = try Self.makeRenderPipeline(
      device: device,
      library: library,
      vertexName: "worldMapSkyVertex",
      fragmentName: "worldMapSkyFragment",
      maxViewCount: maxViewCount)
    opaqueDepthState = Self.makeDepthState(device: device)
    skyDepthState = Self.makeDepthState(device: device)

    let sky = try Self.makeSkyDome(device: device)
    skyVertexBuffer = sky.vertexBuffer
    skyIndexBuffer = sky.indexBuffer
    skyIndexCount = sky.indexCount
    placeholderTexture = try Self.makePlaceholderTexture(device: device)
    guard let mipmapQueue = device.makeCommandQueue() else {
      throw WorldMapError.resourceAllocationFailed("command queue")
    }
    self.mipmapQueue = mipmapQueue

    overlayPipeline = try Self.makeOverlayPipeline(
      device: device,
      library: library,
      maxViewCount: maxViewCount)
    overlayDepthState = Self.makeOverlayDepthState(device: device)
    let overlay = try Self.makeOverlayTexture(device: device)
    overlayTexture = overlay.texture
    overlayTextureAspect = overlay.aspect

    let queue = OperationQueue()
    queue.name = "vr-dive.worldmap.tiles"
    queue.qualityOfService = .utility
    queue.maxConcurrentOperationCount = Self.maximumConcurrentLoads
    loadQueue = queue

    print(
      "[WorldMap] Streaming quadtree map ready: start=(\(Self.startLatitude),\(Self.startLongitude)), zoom=z\(Self.rootZoom)...z\(Self.maximumZoom), imagery=\(google.isAvailable ? "Google satellite" : "unavailable (no key)"), terrain=\(OpenDEMTileClient.attribution), imageryAttribution=\(GoogleMapsTileClient.attribution)"
    )
  }

  func synchronizeState(_ context: PatternSimulationContext) {
    mapFlightTier = context.mapFlightTier
    navigationSpeedScale = context.mapFlightTier.speedScale
    highDetailRadius = context.mapDetailLevel.imageryBiasRadius
    let imageryEnabledNow = context.mapImagerySource.providesImagery
    if imageryEnabledNow != imageryEnabled {
      imageryEnabled = imageryEnabledNow
      clearTileCache()
      didLogConfiguration = false
      print("[WorldMap] Imagery source -> \(context.mapImagerySource.displayName)")
    }
    if let request = context.mapRelocateRequest,
      context.mapRelocateGeneration != lastRelocateGeneration
    {
      lastRelocateGeneration = context.mapRelocateGeneration
      relocate(to: request)
    }
  }

  /// Drops every loaded tile and invalidates in-flight builds.
  private func clearTileCache() {
    lock.lock()
    gridGeneration += 1
    tiles.removeAll()
    readyTiles.removeAll()
    pendingLoads.removeAll()
    retryCounts.removeAll()
    retryAfter.removeAll()
    tileLastUsed.removeAll()
    currentLeaves.removeAll()
    lastGridUpdateTime = -1_000
    lastGridCamera = nil
    lastSolveRequestTime = -1_000
    solvedLeaves = nil
    lock.unlock()
  }

  func updateSimulation(_ context: PatternSimulationContext) {}

  func resetToInitialState() {
    lock.lock()
    gridGeneration += 1
    tiles.removeAll()
    readyTiles.removeAll()
    pendingLoads.removeAll()
    retryCounts.removeAll()
    retryAfter.removeAll()
    tileLastUsed.removeAll()
    currentLeaves.removeAll()
    verticalOffset = -(Self.startGroundEstimate + Self.startAltitude)
    desiredClearance = Self.startAltitude
    hasClearanceBaseline = false
    lastFrameTime = -1
    lastGridUpdateTime = -1_000
    lastGridCamera = nil
    lastSolveRequestTime = -1_000
    solvedLeaves = nil
    lock.unlock()
    print("[WorldMap] Map tiles reset.")
  }

  /// Re-anchors the local frame to a new coordinate. All tiles are rebuilt
  /// against the new origin; the caller is responsible for resetting the
  /// navigation transform so the camera snaps back to the new start point.
  private func relocate(to coordinate: MapCoordinate) {
    reference = MapSceneReference(
      latitude: coordinate.latitude,
      longitude: coordinate.longitude,
      geometryZoom: Self.rootZoom)
    lock.lock()
    gridGeneration += 1
    tiles.removeAll()
    readyTiles.removeAll()
    pendingLoads.removeAll()
    retryCounts.removeAll()
    retryAfter.removeAll()
    tileLastUsed.removeAll()
    currentLeaves.removeAll()
    verticalOffset = -(Self.startGroundEstimate + Self.startAltitude)
    desiredClearance = Self.startAltitude
    hasClearanceBaseline = false
    lastFrameTime = -1
    lastGridUpdateTime = -1_000
    lastGridCamera = nil
    lastSolveRequestTime = -1_000
    solvedLeaves = nil
    lock.unlock()
    didLogConfiguration = false
    print(
      "[WorldMap] Relocated to \(coordinate.latitude), \(coordinate.longitude).")
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    context.applyViewConfiguration(on: encoder)
    let navigation = acceleratedNavigationTransform(context.patternNavigationTransform)
    let navigationInverse = simd_inverse(navigation)
    let cameraScene = cameraScenePosition(context: context, navigation: navigation)

    let deltaTime = lastFrameTime < 0
      ? 0 : min(max(context.time - lastFrameTime, 0), 0.1)
    lastFrameTime = context.time
    updateTerrainFollowing(cameraScene: cameraScene, deltaTime: deltaTime)

    if context.time - lastGridUpdateTime >= Self.gridUpdateInterval {
      lastGridUpdateTime = context.time
      let leaves = takeSolvedLeaves() ?? currentLeaves
      applyGrid(leaves: leaves, cameraScene: cameraScene, now: context.time)
    }
    scheduleGridSolve(cameraScene: cameraScene, now: context.time)
    drainReadyTiles()
    publishStatus(cameraScene: cameraScene, now: context.time)

    // Camera forward in scene space, used to cull tiles behind the viewer.
    let viewToWorld = context.viewData.viewToWorldTransforms.first ?? matrix_identity_float4x4
    let worldForward = -SIMD3<Float>(
      viewToWorld.columns.2.x, viewToWorld.columns.2.y, viewToWorld.columns.2.z)
    let sceneForward4 = navigationInverse * SIMD4<Float>(worldForward, 0)
    let sceneForward = SIMD3<Float>(sceneForward4.x, sceneForward4.y, sceneForward4.z)
    let renderSet = makeRenderSet(cameraScene: cameraScene, sceneForward: sceneForward)

    var uniforms = WorldMapUniforms(
      viewCount: UInt32(max(context.viewData.viewCount, 1)),
      pad0: 0,
      verticalOffset: verticalOffset,
      pad1: 0,
      cameraScene: SIMD4<Float>(cameraScene, 1),
      fogParams: SIMD4<Float>(
        Self.maxViewDistance * 0.5, Self.maxViewDistance * 0.98, 0, 0),
      navigationInverse: navigationInverse)
    var viewProjectionMatrices = context.viewData.viewProjectionMatrices
    if viewProjectionMatrices.isEmpty {
      viewProjectionMatrices = [matrix_identity_float4x4]
    }

    encoder.pushDebugGroup("WorldMap sky")
    encoder.setRenderPipelineState(skyPipeline)
    encoder.setDepthStencilState(skyDepthState)
    encoder.setCullMode(.none)
    encoder.setVertexBuffer(skyVertexBuffer, offset: 0, index: 0)
    setVertexSharedData(
      encoder: encoder,
      uniforms: &uniforms,
      viewProjectionMatrices: viewProjectionMatrices)
    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: skyIndexCount,
      indexType: .uint16,
      indexBuffer: skyIndexBuffer,
      indexBufferOffset: 0)
    encoder.popDebugGroup()

    encoder.pushDebugGroup("WorldMap terrain")
    encoder.setRenderPipelineState(terrainPipeline)
    encoder.setDepthStencilState(opaqueDepthState)
    encoder.setCullMode(.none)
    setVertexSharedData(
      encoder: encoder,
      uniforms: &uniforms,
      viewProjectionMatrices: viewProjectionMatrices)
    encoder.setFragmentBytes(
      &uniforms,
      length: MemoryLayout<WorldMapUniforms>.stride,
      index: 1)
    var drawn = 0
    for id in renderSet {
      guard let tile = tiles[id] else { continue }
      encoder.setVertexBuffer(tile.vertexBuffer, offset: 0, index: 0)
      var hasTexture: UInt32 = tile.texture != nil ? 1 : 0
      encoder.setFragmentBytes(&hasTexture, length: MemoryLayout<UInt32>.stride, index: 0)
      encoder.setFragmentTexture(tile.texture ?? placeholderTexture, index: 0)
      encoder.drawIndexedPrimitives(
        type: .triangle,
        indexCount: tile.indexCount,
        indexType: .uint16,
        indexBuffer: tile.indexBuffer,
        indexBufferOffset: 0)
      drawn += 1
    }
    encoder.popDebugGroup()

    encoder.pushDebugGroup("WorldMap attribution")
    encoder.setRenderPipelineState(overlayPipeline)
    encoder.setDepthStencilState(overlayDepthState)
    encoder.setCullMode(.none)
    var overlayVertices = overlayQuadVertices(context: context)
    encoder.setVertexBytes(
      &overlayVertices,
      length: overlayVertices.count * MemoryLayout<SIMD4<Float>>.stride,
      index: 0)
    encoder.setFragmentTexture(overlayTexture, index: 0)
    encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    encoder.popDebugGroup()

    if !didLogConfiguration {
      didLogConfiguration = true
      print(
        "[WorldMap] First frame: \(currentLeaves.count) leaves, \(drawn) tiles drawn. Navigate with □ to toggle pattern mode, sticks to fly, shoulders to boost.")
    }
  }

  // MARK: - Navigation

  private func acceleratedNavigationTransform(_ transform: simd_float4x4) -> simd_float4x4 {
    var result = transform
    result.columns.3.x *= navigationSpeedScale
    result.columns.3.y *= navigationSpeedScale
    result.columns.3.z *= navigationSpeedScale
    return result
  }

  private func cameraScenePosition(
    context: PatternRenderContext,
    navigation: simd_float4x4
  ) -> SIMD3<Float> {
    let viewToWorld = context.viewData.viewToWorldTransforms.first ?? matrix_identity_float4x4
    let cameraWorld = SIMD4<Float>(
      viewToWorld.columns.3.x,
      viewToWorld.columns.3.y,
      viewToWorld.columns.3.z,
      1)
    let scene = simd_inverse(navigation) * cameraWorld
    return SIMD3<Float>(scene.x, scene.y, scene.z)
  }

  // MARK: - Quadtree

  private static func tileSpanMeters(zoom: Int, cosLatitude: Float) -> Float {
    let circumference = Float(2.0 * Double.pi * MapProjection.earthRadius) * cosLatitude
    return circumference / Float(1 << zoom)
  }

  /// Mesh density follows tile zoom. Deep leaves under the camera keep full
  /// detail; coarse leaves far away use a handful of quads, cutting the total
  /// vertex and triangle load by an order of magnitude. The stored height grid
  /// used for terrain following shrinks accordingly.
  private static func meshQuads(forZoom zoom: Int) -> Int {
    switch zoom {
    case 15: return 64
    case 14: return 48
    case 13: return 32
    case 12: return 24
    case 11: return 16
    case 10: return 12
    default: return 8
    }
  }

  /// Recursively selects leaf tiles: split while the node is still large on
  /// screen (relative to flight altitude) and close enough to the camera.
  /// Pure and `nonisolated` so it can run on the background grid queue.
  nonisolated private func collectLeaves(
    cameraScene: SIMD3<Float>,
    reference: MapSceneReference,
    cosLatitude: Float
  ) -> Set<MapTileID> {
    var leaves = Set<MapTileID>()
    let rootCenter = reference.tile(
      atSceneX: cameraScene.x,
      sceneZ: cameraScene.z,
      zoom: Self.rootZoom)
    let rootTileCount = Int(MapProjection.tileCount(zoom: Self.rootZoom))
    for deltaY in -1...1 {
      for deltaX in -1...1 {
        let x = rootCenter.x + deltaX
        let y = rootCenter.y + deltaY
        guard x >= 0, y >= 0, x < rootTileCount, y < rootTileCount else { continue }
        subdivide(
          MapTileID(z: Self.rootZoom, x: x, y: y),
          cameraScene: cameraScene,
          altitude: cameraScene.y + Self.startAltitude,
          reference: reference,
          cosLatitude: cosLatitude,
          into: &leaves)
      }
    }
    enforceBalance(&leaves)
    applyLeafBudget(&leaves, cameraScene: cameraScene, reference: reference)
    return leaves
  }

  /// Hard ceiling on visible leaves. When the moving quadtree produces more,
  /// the farthest sibling groups are merged into their parent until the count
  /// is back under budget. This keeps the per-frame triangle load bounded on
  /// weaker GPUs and during fast travel.
  nonisolated private func applyLeafBudget(
    _ leaves: inout Set<MapTileID>,
    cameraScene: SIMD3<Float>,
    reference: MapSceneReference
  ) {
    guard leaves.count > Self.maximumLeaves else { return }
    var parentGroups: [MapTileID: Int] = [:]
    for leaf in leaves where leaf.z > Self.rootZoom {
      parentGroups[leaf.parent, default: 0] += 1
    }
    let candidates = parentGroups.compactMap {
      (parent: MapTileID, count: Int) -> (parent: MapTileID, distance: Float)? in
      guard count >= 2 else { return nil }
      let center = MapProjection.tileCenterScenePosition(id: parent, reference: reference)
      let distance = hypot(center.x - cameraScene.x, center.y - cameraScene.z)
      return (parent, distance)
    }.sorted { $0.distance > $1.distance }

    for candidate in candidates {
      guard leaves.count > Self.maximumLeaves else { break }
      let removed = leaves.filter {
        $0 == candidate.parent
          || Self.isDescendant($0, of: candidate.parent, deeperThan: candidate.parent.z)
      }
      guard removed.count >= 2 else { continue }
      for id in removed { leaves.remove(id) }
      leaves.insert(candidate.parent)
    }
    enforceBalance(&leaves)
  }

  /// Splits leaves whose orthogonal neighbour is subdivided, so adjacent
  /// leaves never differ by more than one zoom level. That keeps T-junctions to
  /// a single step that the edge skirts can cover reliably.
  nonisolated private func enforceBalance(_ leaves: inout Set<MapTileID>) {
    var passes = 0
    while passes < 8 {
      passes += 1
      var toSplit = Set<MapTileID>()
      for leaf in leaves where leaf.z < Self.maximumZoom {
        for neighbor in Self.edgeNeighbors(of: leaf) {
          if leaves.contains(where: { Self.isDescendant($0, of: neighbor, deeperThan: leaf.z) }) {
            toSplit.insert(leaf)
            break
          }
        }
      }
      guard !toSplit.isEmpty else { break }
      for leaf in toSplit {
        leaves.remove(leaf)
        for deltaY in 0...1 {
          for deltaX in 0...1 {
            leaves.insert(
              MapTileID(z: leaf.z + 1, x: leaf.x * 2 + deltaX, y: leaf.y * 2 + deltaY))
          }
        }
      }
    }
  }

  private static func edgeNeighbors(of id: MapTileID) -> [MapTileID] {
    [
      MapTileID(z: id.z, x: id.x - 1, y: id.y),
      MapTileID(z: id.z, x: id.x + 1, y: id.y),
      MapTileID(z: id.z, x: id.x, y: id.y - 1),
      MapTileID(z: id.z, x: id.x, y: id.y + 1),
    ]
  }

  private static func isDescendant(
    _ candidate: MapTileID,
    of ancestor: MapTileID,
    deeperThan zoom: Int
  ) -> Bool {
    guard candidate.z > zoom, candidate.z > ancestor.z else { return false }
    let shift = candidate.z - ancestor.z
    return (candidate.x >> shift) == ancestor.x && (candidate.y >> shift) == ancestor.y
  }

  nonisolated private func subdivide(
    _ id: MapTileID,
    cameraScene: SIMD3<Float>,
    altitude: Float,
    reference: MapSceneReference,
    cosLatitude: Float,
    into leaves: inout Set<MapTileID>
  ) {
    let center = MapProjection.tileCenterScenePosition(id: id, reference: reference)
    let distance = hypot(center.x - cameraScene.x, center.y - cameraScene.z)
    guard distance <= Self.maxViewDistance else { return }
    let size = Self.tileSpanMeters(zoom: id.z, cosLatitude: cosLatitude)
    let minimumSpan = max(80, altitude * Self.minimumVisibleSpanFactor)
    if id.z < Self.maximumZoom,
      distance < size * Self.splitDistanceFactor,
      size > minimumSpan
    {
      for deltaY in 0...1 {
        for deltaX in 0...1 {
          subdivide(
            MapTileID(z: id.z + 1, x: id.x * 2 + deltaX, y: id.y * 2 + deltaY),
            cameraScene: cameraScene,
            altitude: altitude,
            reference: reference,
            cosLatitude: cosLatitude,
            into: &leaves)
        }
      }
    } else {
      leaves.insert(id)
    }
  }

  private func imageryBias(for leaf: MapTileID, cameraScene: SIMD3<Float>) -> Int {
    guard imageryEnabled, highDetailRadius > 0, leaf.z >= Self.highDetailZoomThreshold else {
      return 0
    }
    let center = MapProjection.tileCenterScenePosition(id: leaf, reference: reference)
    let size = Self.tileSpanMeters(zoom: leaf.z, cosLatitude: Float(reference.cosLatitude))
    let distance = hypot(center.x - cameraScene.x, center.y - cameraScene.z)
    return distance <= size * Float(highDetailRadius) ? 1 : 0
  }

  /// Requests a background quadtree solve when the camera has actually moved.
  /// Only one solve runs at a time; a solve that outlives a reset is discarded
  /// via the generation check.
  private func scheduleGridSolve(cameraScene: SIMD3<Float>, now: Float) {
    let moved = lastGridCamera.map { simd_distance($0, cameraScene) > 20 } ?? true
    guard moved, now - lastSolveRequestTime >= Self.gridUpdateInterval else { return }
    lock.lock()
    guard !isSolvingGrid else {
      lock.unlock()
      return
    }
    isSolvingGrid = true
    lastSolveRequestTime = now
    lastGridCamera = cameraScene
    let generation = gridGeneration
    let referenceSnapshot = reference
    let cosLatitude = Float(reference.cosLatitude)
    lock.unlock()

    gridQueue.async { [weak self] in
      guard let self else { return }
      let leaves = self.collectLeaves(
        cameraScene: cameraScene,
        reference: referenceSnapshot,
        cosLatitude: cosLatitude)
      self.lock.lock()
      if generation == self.gridGeneration {
        self.solvedLeaves = leaves
      }
      self.isSolvingGrid = false
      self.lock.unlock()
    }
  }

  private func takeSolvedLeaves() -> Set<MapTileID>? {
    lock.lock()
    defer { lock.unlock() }
    let leaves = solvedLeaves
    solvedLeaves = nil
    return leaves
  }

  private func applyGrid(
    leaves: Set<MapTileID>,
    cameraScene: SIMD3<Float>,
    now: Float
  ) {
    lock.lock()
    currentLeaves = leaves
    // Keep leaves plus every ancestor so a not-yet-loaded leaf can fall back to
    // a coarser parent instead of showing a hole.
    var wanted = Set<MapTileID>()
    for leaf in leaves {
      wanted.insert(leaf)
      var node = leaf
      while node.z > Self.rootZoom {
        node = node.parent
        wanted.insert(node)
      }
    }
    for id in wanted { tileLastUsed[id] = now }

    let readyIDs = Set(readyTiles.map { $0.id })
    let pendingIDs = Set(pendingLoads.map { $0.id })
    // Queue every wanted tile, not just the leaves, so the coarse ancestors
    // load first and cover the view immediately; the detail leaves stream in
    // afterwards.
    for tile in wanted {
      guard tiles[tile] == nil, !inFlight.contains(tile), !readyIDs.contains(tile),
        !pendingIDs.contains(tile), (retryCounts[tile] ?? 0) < 3,
        (retryAfter[tile] ?? -.greatestFiniteMagnitude) <= now
      else { continue }
      pendingLoads.append(
        PendingLoad(
          id: tile,
          imageryZoomBias: imageryBias(for: tile, cameraScene: cameraScene),
          imageryEnabled: imageryEnabled,
          generation: gridGeneration,
          requestedAt: now))
    }
    // Coarse zoom first gives progressive coverage; when over the queue budget
    // drop the finest (least critical) work.
    pendingLoads.sort { $0.id.z < $1.id.z }
    if pendingLoads.count > Self.maximumPendingLoads {
      pendingLoads.removeLast(pendingLoads.count - Self.maximumPendingLoads)
    }

    let expired = tiles.keys.filter { id in
      guard !wanted.contains(id) else { return false }
      let last = tileLastUsed[id] ?? -.greatestFiniteMagnitude
      return now - last > Self.tileRetainSeconds
    }
    for id in expired {
      tiles[id] = nil
      tileLastUsed[id] = nil
      retryCounts[id] = nil
      retryAfter[id] = nil
    }

    // Keep resident GPU memory under budget by evicting the least-recently-used
    // tiles that are no longer part of the current quadtree.
    var resident = tiles.values.reduce(0) { $0 + $1.gpuBytes }
    if resident > Self.gpuBudgetBytes {
      let evictable = tiles.keys
        .filter { !wanted.contains($0) }
        .sorted { (tileLastUsed[$0] ?? 0) < (tileLastUsed[$1] ?? 0) }
      for id in evictable {
        guard resident > Self.gpuBudgetBytes, let tile = tiles[id] else { break }
        resident -= tile.gpuBytes
        tiles[id] = nil
        tileLastUsed[id] = nil
        retryCounts[id] = nil
        retryAfter[id] = nil
      }
    }
    residentBytes = resident
    lock.unlock()

    pumpLoads()
  }

  /// Draws each leaf if loaded, otherwise the nearest loaded ancestor, so there
  /// is never both a parent and its child covering the same ground. Leaves
  /// whose footprint is entirely behind the camera are skipped.
  private func makeRenderSet(
    cameraScene: SIMD3<Float>,
    sceneForward: SIMD3<Float>
  ) -> [MapTileID] {
    lock.lock()
    let leaves = currentLeaves
    let loadedIDs = Set(tiles.keys)
    lock.unlock()
    let forwardXZ = SIMD2<Float>(sceneForward.x, sceneForward.z)
    let forwardLength = simd_length(forwardXZ)
    let cullEnabled = forwardLength > 0.15
    let forwardDirection = cullEnabled ? forwardXZ / forwardLength : SIMD2<Float>.zero
    let cosLatitude = Float(reference.cosLatitude)
    var renderSet = Set<MapTileID>()
    for leaf in leaves {
      if cullEnabled {
        let center = MapProjection.tileCenterScenePosition(id: leaf, reference: reference)
        let radius = Self.tileSpanMeters(zoom: leaf.z, cosLatitude: cosLatitude) * 0.75
        let delta = SIMD2<Float>(center.x - cameraScene.x, center.y - cameraScene.z)
        if simd_dot(delta, forwardDirection) < -radius { continue }
      }
      var node = leaf
      while true {
        if loadedIDs.contains(node) {
          renderSet.insert(node)
          break
        }
        guard node.z > Self.rootZoom else { break }
        node = node.parent
      }
    }
    // A coarse fallback tile covers its whole subtree, so drop any loaded
    // descendant that happens to be in the set to avoid double-drawing.
    for id in Array(renderSet) {
      var ancestor = id.parent
      while ancestor.z >= Self.rootZoom {
        if renderSet.contains(ancestor) {
          renderSet.remove(id)
          break
        }
        ancestor = ancestor.parent
      }
    }
    return Array(renderSet)
  }

  private func pumpLoads() {
    while true {
      lock.lock()
      guard inFlight.count < Self.maximumConcurrentLoads, !pendingLoads.isEmpty else {
        lock.unlock()
        return
      }
      let next = pendingLoads.removeFirst()
      inFlight.insert(next.id)
      lock.unlock()

      loadQueue.addOperation { [weak self] in
        guard let self else { return }
        let tile = WorldMapTileBuilder.build(
          device: self.device,
          id: next.id,
          reference: self.reference,
          meshQuads: Self.meshQuads(forZoom: next.id.z),
          imageryZoomBias: next.imageryZoomBias,
          imageryEnabled: next.imageryEnabled,
          skirtDepth: Self.skirtDepth,
          commandQueue: self.mipmapQueue,
          google: self.google)
        self.lock.lock()
        let isCurrent = next.generation == self.gridGeneration
        if let tile, isCurrent {
          self.readyTiles.append(tile)
          self.retryCounts[next.id] = nil
          self.retryAfter[next.id] = nil
        } else if !isCurrent {
          // Reset happened underneath this build; drop it silently.
        } else {
          let attempts = self.retryCounts[next.id, default: 0] + 1
          self.retryCounts[next.id] = attempts
          if attempts <= 3 {
            // Exponential backoff keeps a flaky connection from being hammered.
            let backoff = min(powf(2, Float(attempts)), 30)
            self.retryAfter[next.id] = next.requestedAt + backoff
          }
        }
        self.inFlight.remove(next.id)
        self.lock.unlock()
        self.pumpLoads()
      }
    }
  }

  private func drainReadyTiles() {
    lock.lock()
    let ready = readyTiles
    readyTiles.removeAll()
    lock.unlock()
    guard !ready.isEmpty else { return }
    for tile in ready {
      tiles[tile.id] = tile
    }
  }

  // MARK: - Terrain following

  /// Keeps the camera at a roughly constant height above the ground beneath it.
  /// The vertical stick is folded into `desiredClearance` (so the player still
  /// climbs and descends), while the terrain offset follows ridges and valleys
  /// instead of letting the camera clip into a mountain.
  private func updateTerrainFollowing(cameraScene: SIMD3<Float>, deltaTime: Float) {
    if !hasClearanceBaseline {
      clearanceBaselineY = cameraScene.y
      hasClearanceBaseline = true
    }
    let deltaY = cameraScene.y - clearanceBaselineY
    clearanceBaselineY = cameraScene.y
    if abs(deltaY) > 0.02 {
      desiredClearance = min(
        max(desiredClearance + deltaY, Self.minimumClearance),
        Self.maximumClearance)
    }
    guard let ground = terrainSample(atSceneX: cameraScene.x, sceneZ: cameraScene.z) else {
      return
    }
    lastGroundAltitude = cameraScene.y - ground.height
    lastGroundZoom = ground.zoom
    let targetOffset = cameraScene.y - ground.height - desiredClearance
    let difference = targetOffset - verticalOffset
    if difference < -2 {
      // The ground rose under the camera (ridge / fast approach) and the
      // smoothing would lag into the terrain. Lift immediately instead.
      verticalOffset = targetOffset
    } else if deltaTime <= 0 {
      verticalOffset = targetOffset
    } else {
      verticalOffset += difference * min(1, deltaTime * Self.clearanceResponse)
    }
  }

  private func terrainSample(atSceneX x: Float, sceneZ z: Float) -> (height: Float, zoom: Int)? {
    var best: WorldMapTile?
    for tile in tiles.values where tile.contains(sceneX: x, sceneZ: z) {
      if best == nil || tile.id.z > best!.id.z {
        best = tile
      }
    }
    guard let tile = best, let height = tile.height(atSceneX: x, sceneZ: z) else { return nil }
    return (height, tile.id.z)
  }

  private func publishStatus(cameraScene: SIMD3<Float>, now: Float) {
    guard let statusSink, now - lastStatusTime >= 0.5 else { return }
    lastStatusTime = now
    let coordinate = reference.coordinate(sceneX: cameraScene.x, sceneZ: cameraScene.z)
    coordinateSink?(
      MapCoordinate(latitude: coordinate.latitude, longitude: coordinate.longitude))
    let imageryNote = google.isAvailable ? "" : " · 无卫星 key"
    statusSink(
      String(
        format: "z%d · 离地 %.0f m · %.5f°N %.5f°E · %@ · %d 瓦片 · %.0f MB%@",
        lastGroundZoom,
        lastGroundAltitude,
        coordinate.latitude,
        coordinate.longitude,
        mapFlightTier.displayName,
        tiles.count,
        Double(residentBytes) / 1_048_576.0,
        imageryNote))
  }

  private func setVertexSharedData(
    encoder: MTLRenderCommandEncoder,
    uniforms: inout WorldMapUniforms,
    viewProjectionMatrices: [simd_float4x4]
  ) {
    encoder.setVertexBytes(
      &uniforms,
      length: MemoryLayout<WorldMapUniforms>.stride,
      index: 1)
    viewProjectionMatrices.withUnsafeBytes { bytes in
      if let base = bytes.baseAddress, bytes.count > 0 {
        encoder.setVertexBytes(base, length: bytes.count, index: 2)
      }
    }
  }

  // MARK: - Resource construction

  private static func makeSkyDome(
    device: MTLDevice
  ) throws -> (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int) {
    let longitudeSegments = 32
    let latitudeSegments = 16
    var vertices: [SIMD3<Float>] = []
    vertices.reserveCapacity((longitudeSegments + 1) * (latitudeSegments + 1))
    for latitude in 0...latitudeSegments {
      let latitudeAngle = -Float.pi / 2 + Float(latitude) / Float(latitudeSegments) * Float.pi
      let ringRadius = cos(latitudeAngle)
      let y = sin(latitudeAngle)
      for longitude in 0...longitudeSegments {
        let longitudeAngle = Float(longitude) / Float(longitudeSegments) * 2 * Float.pi
        vertices.append(
          SIMD3<Float>(
            ringRadius * cos(longitudeAngle),
            y,
            ringRadius * sin(longitudeAngle)))
      }
    }
    var indices: [UInt16] = []
    let rowStride = longitudeSegments + 1
    for latitude in 0..<latitudeSegments {
      for longitude in 0..<longitudeSegments {
        let a = UInt16(latitude * rowStride + longitude)
        let b = a + 1
        let c = UInt16((latitude + 1) * rowStride + longitude)
        let d = c + 1
        indices.append(contentsOf: [a, c, b, b, c, d])
      }
    }
    guard
      let vertexBuffer = device.makeBuffer(
        bytes: vertices,
        length: vertices.count * MemoryLayout<SIMD3<Float>>.stride,
        options: .storageModeShared),
      let indexBuffer = device.makeBuffer(
        bytes: indices,
        length: indices.count * MemoryLayout<UInt16>.stride,
        options: .storageModeShared)
    else { throw WorldMapError.resourceAllocationFailed("sky dome") }
    vertexBuffer.label = "WorldMap sky vertices"
    indexBuffer.label = "WorldMap sky indices"
    return (vertexBuffer, indexBuffer, indices.count)
  }

  private static func makePlaceholderTexture(device: MTLDevice) throws -> MTLTexture {
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba8Unorm,
      width: 1,
      height: 1,
      mipmapped: false)
    descriptor.usage = .shaderRead
    descriptor.storageMode = .shared
    guard let texture = device.makeTexture(descriptor: descriptor) else {
      throw WorldMapError.resourceAllocationFailed("placeholder texture")
    }
    var pixel: [UInt8] = [40, 40, 40, 255]
    texture.replace(
      region: MTLRegionMake2D(0, 0, 1, 1),
      mipmapLevel: 0,
      withBytes: &pixel,
      bytesPerRow: 4)
    texture.label = "WorldMap placeholder"
    return texture
  }

  private func overlayQuadVertices(context: PatternRenderContext) -> [SIMD4<Float>] {
    let viewport = context.viewData.viewports.first
    let viewportAspect = viewport.map { Float($0.width / max($0.height, 1)) } ?? 1.6
    let heightNDC: Float = 0.06
    let widthNDC = min(0.72, heightNDC * overlayTextureAspect / max(viewportAspect, 0.1))
    let margin: Float = 0.03
    let x0 = -1 + margin
    let y0 = -1 + margin
    let x1 = x0 + widthNDC
    let y1 = y0 + heightNDC
    // triangle strip: (x, y, u, v)
    return [
      SIMD4<Float>(x0, y0, 0, 1),
      SIMD4<Float>(x1, y0, 1, 1),
      SIMD4<Float>(x0, y1, 0, 0),
      SIMD4<Float>(x1, y1, 1, 0),
    ]
  }

  private static func makeOverlayPipeline(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    guard let vertexFunction = library.makeFunction(name: "worldMapOverlayVertex") else {
      throw WorldMapError.missingFunction("worldMapOverlayVertex")
    }
    guard let fragmentFunction = library.makeFunction(name: "worldMapOverlayFragment") else {
      throw WorldMapError.missingFunction("worldMapOverlayFragment")
    }
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = vertexFunction
    descriptor.fragmentFunction = fragmentFunction
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
    descriptor.depthAttachmentPixelFormat = .depth32Float
    descriptor.maxVertexAmplificationCount = max(maxViewCount, 1)
    if let attachment = descriptor.colorAttachments[0] {
      attachment.isBlendingEnabled = true
      attachment.rgbBlendOperation = .add
      attachment.alphaBlendOperation = .add
      attachment.sourceRGBBlendFactor = .one
      attachment.sourceAlphaBlendFactor = .one
      attachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
      attachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
    }
    return try device.makeRenderPipelineState(descriptor: descriptor)
  }

  private static func makeOverlayDepthState(device: MTLDevice) -> MTLDepthStencilState {
    let descriptor = MTLDepthStencilDescriptor()
    descriptor.depthCompareFunction = .always
    descriptor.isDepthWriteEnabled = false
    return device.makeDepthStencilState(descriptor: descriptor)!
  }

  /// Draws the required Google Maps attribution into a small premultiplied
  /// RGBA texture with CoreText. The overlay is the only place the map content
  /// attribution appears inside the immersive layer.
  private static func makeOverlayTexture(
    device: MTLDevice
  ) throws -> (texture: MTLTexture, aspect: Float) {
    let width = 640
    let height = 128
    var pixels = [UInt8](repeating: 0, count: width * height * 4)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
    let drew = pixels.withUnsafeMutableBytes { buffer -> Bool in
      guard
        let context = CGContext(
          data: buffer.baseAddress,
          width: width,
          height: height,
          bitsPerComponent: 8,
          bytesPerRow: width * 4,
          space: colorSpace,
          bitmapInfo: bitmapInfo)
      else { return false }
      context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 0.42))
      context.fill(CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
      drawOverlayLine(
        "Google Maps",
        x: 20,
        y: 68,
        size: 40,
        color: CGColor(red: 1, green: 1, blue: 1, alpha: 0.95),
        context: context)
      drawOverlayLine(
        "地形 © OpenStreetMap / AWS Terrain Tiles",
        x: 20,
        y: 26,
        size: 22,
        color: CGColor(red: 1, green: 1, blue: 1, alpha: 0.82),
        context: context)
      return true
    }
    guard drew else { throw WorldMapError.resourceAllocationFailed("overlay texture") }

    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba8Unorm,
      width: width,
      height: height,
      mipmapped: false)
    descriptor.usage = .shaderRead
    descriptor.storageMode = .shared
    guard let texture = device.makeTexture(descriptor: descriptor) else {
      throw WorldMapError.resourceAllocationFailed("overlay texture")
    }
    texture.label = "WorldMap attribution"
    pixels.withUnsafeBytes { bytes in
      guard let base = bytes.baseAddress else { return }
      texture.replace(
        region: MTLRegionMake2D(0, 0, width, height),
        mipmapLevel: 0,
        withBytes: base,
        bytesPerRow: width * 4)
    }
    return (texture, Float(width) / Float(height))
  }

  private static func drawOverlayLine(
    _ text: String,
    x: CGFloat,
    y: CGFloat,
    size: CGFloat,
    color: CGColor,
    context: CGContext
  ) {
    let font = CTFontCreateWithName("Helvetica-Bold" as CFString, size, nil)
    let attributes: [NSAttributedString.Key: Any] = [
      NSAttributedString.Key(kCTFontAttributeName as String): font,
      NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
    ]
    let attributed = NSAttributedString(string: text, attributes: attributes)
    let line = CTLineCreateWithAttributedString(attributed)
    context.textPosition = CGPoint(x: x, y: y)
    CTLineDraw(line, context)
  }

  private static func makeRenderPipeline(
    device: MTLDevice,
    library: MTLLibrary,
    vertexName: String,
    fragmentName: String,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    guard let vertexFunction = library.makeFunction(name: vertexName) else {
      throw WorldMapError.missingFunction(vertexName)
    }
    guard let fragmentFunction = library.makeFunction(name: fragmentName) else {
      throw WorldMapError.missingFunction(fragmentName)
    }
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = vertexFunction
    descriptor.fragmentFunction = fragmentFunction
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
    descriptor.depthAttachmentPixelFormat = .depth32Float
    descriptor.maxVertexAmplificationCount = max(maxViewCount, 1)
    return try device.makeRenderPipelineState(descriptor: descriptor)
  }

  private static func makeDepthState(device: MTLDevice) -> MTLDepthStencilState {
    let descriptor = MTLDepthStencilDescriptor()
    descriptor.depthCompareFunction = .greater
    descriptor.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: descriptor)!
  }
}

private enum WorldMapError: Error {
  case missingFunction(String)
  case resourceAllocationFailed(String)
}
