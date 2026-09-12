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

nonisolated struct WorldMapCityLabelVertex {
  var anchor: SIMD4<Float>
  /// x/y are clip-space offsets; z/w are texture u/v.
  var cornerUV: SIMD4<Float>
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
  private static let gridUpdateInterval: Float = 0.22
  private static let tileRetainSeconds: Float = 3.0
  private static let rootZoom = 8

  private static let minimumClearance: Float = 30
  private static let maximumClearance: Float = 2_000_000
  private static let clearanceResponse: Float = 10

  private static let startLatitude = 28.281_051
  private static let startLongitude = 85.545_404
  private static let startGroundEstimate: Float = 2_800
  private static let startAltitude: Float = 1_600

  private struct PendingLoad {
    let id: MapTileID
    let imageryZoomBias: Int
    let generation: Int
    let reference: MapSceneReference
    // nil means terrain; otherwise this is a texture-only replacement.
    let baseTile: WorldMapTile?
    let cancellation = MapLoadCancellation()

    var reservedBytes: Int {
      baseTile == nil
        ? MapStreamingPolicy.meshBytes(zoom: id.z)
        : MapStreamingPolicy.textureBytes(bias: imageryZoomBias)
    }
  }

  private struct CompletedLoad {
    let request: PendingLoad
    let tile: WorldMapTile?
    let completedAt: TimeInterval
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
  private let compassPipeline: MTLRenderPipelineState
  private let compassDepthState: MTLDepthStencilState
  private let compassVertexBuffer: MTLBuffer
  private let compassTexture: MTLTexture
  private let cityLabelPipeline: MTLRenderPipelineState
  private let cityLabelDepthState: MTLDepthStencilState
  private let cityLabelTexture: MTLTexture
  private var cityLabelVertexBuffer: MTLBuffer
  private var cityLabelVertexCount: Int
  private let placeholderTexture: MTLTexture
  private let mipmapQueue: MTLCommandQueue
  private let overlayPipeline: MTLRenderPipelineState
  private let overlayDepthState: MTLDepthStencilState
  private let overlayTexture: MTLTexture
  private let overlayTextureAspect: Float

  private let loadQueue: OperationQueue
  private let lock = NSLock()
  private var tiles: [MapTileID: WorldMapTile] = [:]
  private var readyTiles: [CompletedLoad] = []
  private var pendingLoads: [PendingLoad] = []
  private var inFlight: [MapTileID: PendingLoad] = [:]
  private var retries: [MapTileID: MapRetryState] = [:]
  private var wantedTiles: Set<MapTileID> = []
  private var tileLastUsed: [MapTileID: Float] = [:]
  private var currentLeaves: Set<MapTileID> = []
  private var displayedTiles: Set<MapTileID> = []
  private var activeRootZoom = 8
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
  private var navigationState = MapNavigationState()
  private var lastGridClearance: Float = -1
  private var mapFlightTier: MapFlightTier = .cruise
  private var lastStatusTime: Float = -1
  private var lastRelocateGeneration: Int = -1
  private var lastGroundAltitude: Float = WorldMapRenderer.startAltitude
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
    compassPipeline = try Self.makeCompassPipeline(
      device: device,
      library: library,
      maxViewCount: maxViewCount)
    compassDepthState = Self.makeCompassDepthState(device: device)
    compassVertexBuffer = try Self.makeCompassVertexBuffer(device: device)
    compassTexture = try Self.makeCompassTexture(device: device)
    cityLabelPipeline = try Self.makeCityLabelPipeline(
      device: device, library: library, maxViewCount: maxViewCount)
    cityLabelDepthState = Self.makeCompassDepthState(device: device)
    cityLabelTexture = try Self.makeCityLabelTexture(device: device)
    let cityLabels = try Self.makeCityLabelVertexBuffer(device: device, reference: reference)
    cityLabelVertexBuffer = cityLabels.buffer
    cityLabelVertexCount = cityLabels.count
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
      "[WorldMap] Streaming quadtree map ready: start=(\(Self.startLatitude),\(Self.startLongitude)), zoom=z0...z\(MapStreamingPolicy.maximumZoom), imagery=\(google.isAvailable ? "Google satellite" : "unavailable (no key)"), terrain=\(OpenDEMTileClient.attribution), imageryAttribution=\(GoogleMapsTileClient.attribution)"
    )
  }

  deinit {
    for request in inFlight.values { request.cancellation.cancel() }
  }

  func synchronizeState(_ context: PatternSimulationContext) {
    mapFlightTier = context.mapFlightTier
    navigationSpeedScale = context.mapFlightTier.speedScale
    let radius = context.mapDetailLevel.imageryBiasRadius
    if highDetailRadius != radius {
      highDetailRadius = radius
      for request in inFlight.values where request.baseTile != nil { request.cancellation.cancel() }
      retries.removeAll()
      lastGridUpdateTime = -1_000
    }
    let imageryEnabledNow = context.mapImagerySource.providesImagery
    if imageryEnabledNow != imageryEnabled {
      imageryEnabled = imageryEnabledNow
      for request in inFlight.values where request.baseTile != nil { request.cancellation.cancel() }
      for (id, tile) in tiles where !imageryEnabled {
        tiles[id] = tile.replacingTexture(nil, bias: 0)
      }
      retries.removeAll()
      lastGridUpdateTime = -1_000
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

  /// Only the render thread owns tiles and scheduling. Workers publish completions
  /// through the lock, and cancelled requests retain their slot until they finish.
  private func clearTileCache() {
    lock.lock()
    gridGeneration += 1
    solvedLeaves = nil
    lock.unlock()
    for request in inFlight.values { request.cancellation.cancel() }
    pendingLoads.removeAll()
    tiles.removeAll()
    retries.removeAll()
    tileLastUsed.removeAll()
    wantedTiles.removeAll()
    currentLeaves.removeAll()
    displayedTiles.removeAll()
    activeRootZoom = Self.rootZoom
    residentBytes = 0
    lastGridUpdateTime = -1_000
    lastGridCamera = nil
    lastGridClearance = -1
    lastSolveRequestTime = -1_000
  }

  func updateSimulation(_ context: PatternSimulationContext) {}

  func resetToInitialState() {
    clearTileCache()
    navigationState = MapNavigationState()
    verticalOffset = -(Self.startGroundEstimate + Self.startAltitude)
    desiredClearance = Self.startAltitude
    lastGroundAltitude = Self.startAltitude
    lastGroundZoom = 0
    hasClearanceBaseline = false
    lastFrameTime = -1
    didLogConfiguration = false
  }

  private func relocate(to coordinate: MapCoordinate) {
    reference = MapSceneReference(
      latitude: coordinate.latitude, longitude: coordinate.longitude,
      geometryZoom: Self.rootZoom)
    if let labels = try? Self.makeCityLabelVertexBuffer(device: device, reference: reference) {
      cityLabelVertexBuffer = labels.buffer
      cityLabelVertexCount = labels.count
    }
    resetToInitialState()
    print("[WorldMap] Relocated to \(coordinate.latitude), \(coordinate.longitude).")
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    context.applyViewConfiguration(on: encoder)
    let navigation = acceleratedNavigationTransform(context.patternNavigationTransform)
    let navigationInverse = simd_inverse(navigation)
    let cameraScene = cameraScenePosition(context: context, navigation: navigation)

    let deltaTime =
      lastFrameTime < 0
      ? 0 : min(max(context.time - lastFrameTime, 0), 0.1)
    lastFrameTime = context.time
    drainReadyTiles()
    if context.time - lastGridUpdateTime >= Self.gridUpdateInterval {
      lastGridUpdateTime = context.time
      let leaves = takeSolvedLeaves() ?? currentLeaves
      applyGrid(leaves: leaves, cameraScene: cameraScene, now: context.time)
    }
    let coverage = makeCoverageSet()
    displayedTiles = coverage
    for id in coverage { tileLastUsed[id] = context.time }
    updateTerrainFollowing(cameraScene: cameraScene, deltaTime: deltaTime, coverage: coverage)
    scheduleGridSolve(cameraScene: cameraScene, now: context.time)
    pumpLoads()
    publishStatus(cameraScene: cameraScene, now: context.time)

    let clipFromScene = context.viewData.viewProjectionMatrices.map { $0 * navigationInverse }
    let renderSet = makeRenderSet(
      coverage: coverage, cameraScene: cameraScene, clipFromScene: clipFromScene)

    let currentViewDistance = MapStreamingPolicy.viewDistance(
      clearance: max(lastGroundAltitude, Self.minimumClearance))
    var uniforms = WorldMapUniforms(
      viewCount: UInt32(max(context.viewData.viewCount, 1)),
      pad0: 0,
      verticalOffset: verticalOffset,
      pad1: 0,
      cameraScene: SIMD4<Float>(cameraScene, 1),
      fogParams: SIMD4<Float>(
        currentViewDistance * 0.72, currentViewDistance * 0.99, 0, 0),
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

    encoder.pushDebugGroup("WorldMap compass")
    encoder.setRenderPipelineState(compassPipeline)
    encoder.setDepthStencilState(compassDepthState)
    encoder.setCullMode(.none)
    encoder.setVertexBuffer(compassVertexBuffer, offset: 0, index: 0)
    setVertexSharedData(
      encoder: encoder,
      uniforms: &uniforms,
      viewProjectionMatrices: viewProjectionMatrices)
    encoder.setFragmentTexture(compassTexture, index: 0)
    encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
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

    encoder.pushDebugGroup("WorldMap city labels")
    encoder.setRenderPipelineState(cityLabelPipeline)
    encoder.setDepthStencilState(cityLabelDepthState)
    encoder.setCullMode(.none)
    encoder.setVertexBuffer(cityLabelVertexBuffer, offset: 0, index: 0)
    setVertexSharedData(
      encoder: encoder,
      uniforms: &uniforms,
      viewProjectionMatrices: viewProjectionMatrices)
    encoder.setFragmentTexture(cityLabelTexture, index: 0)
    encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: cityLabelVertexCount)
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
        "[WorldMap] First frame: \(currentLeaves.count) leaves, \(drawn) tiles drawn. Navigate with □ to toggle pattern mode, sticks to fly, shoulders to boost."
      )
    }
  }

  // MARK: - Navigation

  private func acceleratedNavigationTransform(_ transform: simd_float4x4) -> simd_float4x4 {
    navigationState.transform(transform, speed: navigationSpeedScale)
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
    let scene = navigation * cameraWorld
    return SIMD3<Float>(scene.x, scene.y, scene.z)
  }

  // MARK: - Quadtree

  private func imageryBias(for leaf: MapTileID, cameraScene: SIMD3<Float>) -> Int {
    guard imageryEnabled, highDetailRadius > 0, leaf.z >= 14 else {
      return 0
    }
    let center = MapProjection.tileCenterScenePosition(id: leaf, reference: reference)
    let size = MapStreamingPolicy.span(zoom: leaf.z, reference: reference)
    let distance = hypot(center.x - cameraScene.x, center.y - cameraScene.z)
    return distance <= size * Float(highDetailRadius) ? 1 : 0
  }

  /// Requests a background quadtree solve when the camera has actually moved.
  /// Only one solve runs at a time; a solve that outlives a reset is discarded
  /// via the generation check.
  private func scheduleGridSolve(cameraScene: SIMD3<Float>, now: Float) {
    let clearance = max(lastGroundAltitude, Self.minimumClearance)
    let moved =
      (lastGridCamera.map { simd_distance($0, cameraScene) > 20 } ?? true)
      || abs(clearance - lastGridClearance) > max(20, clearance * 0.05)
    guard moved, now - lastSolveRequestTime >= Self.gridUpdateInterval else { return }
    lock.lock()
    guard !isSolvingGrid else {
      lock.unlock()
      return
    }
    isSolvingGrid = true
    lastSolveRequestTime = now
    lastGridCamera = cameraScene
    lastGridClearance = clearance
    let generation = gridGeneration
    let referenceSnapshot = reference
    lock.unlock()

    gridQueue.async { [weak self] in
      guard let self else { return }
      let leaves = MapStreamingPolicy.select(
        camera: cameraScene, clearance: clearance, reference: referenceSnapshot)
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
    leaves: Set<MapTileID>, cameraScene: SIMD3<Float>, now: Float
  ) {
    currentLeaves = leaves
    activeRootZoom = leaves.map(\.z).min() ?? activeRootZoom
    wantedTiles = MapStreamingPolicy.wanted(leaves: leaves, rootZoom: activeRootZoom)
    // Keep finishing imagery for the coverage that was visible last frame,
    // even when an altitude-driven root change makes it no longer part of the
    // new quadtree. It remains the visual fallback until the replacement is ready.
    let activeDemand = wantedTiles.union(displayedTiles)
    let imageryAvailable = imageryEnabled && google.isAvailable
    func bias(_ id: MapTileID) -> Int {
      leaves.contains(id) ? imageryBias(for: id, cameraScene: cameraScene) : 0
    }
    for (id, request) in inFlight {
      if !activeDemand.contains(id) || request.generation != gridGeneration
        || (request.baseTile != nil
          && (!imageryAvailable || request.imageryZoomBias < bias(id)))
      {
        request.cancellation.cancel()
      }
    }
    for id in activeDemand { tileLastUsed[id] = now }
    retries = retries.filter { activeDemand.contains($0.key) }
    let expired = tiles.keys.filter {
      !wantedTiles.contains($0)
        && now - (tileLastUsed[$0] ?? -.greatestFiniteMagnitude) > Self.tileRetainSeconds
    }
    for id in expired {
      tiles[id] = nil
      tileLastUsed[id] = nil
    }

    // Rebuild the small pending list from current demand; obsolete work never
    // remains ahead of the new camera position. In-flight requests have snapshots.
    pendingLoads.removeAll()
    let uptime = ProcessInfo.processInfo.systemUptime
    for id in activeDemand {
      guard inFlight[id] == nil, (retries[id]?.nextAttempt ?? 0) <= uptime else { continue }
      let tile = tiles[id]
      let targetBias = bias(id)
      guard
        tile == nil
          || (imageryAvailable
            && (tile?.texture == nil || (tile?.imageryZoomBias ?? 0) < targetBias))
      else { continue }
      pendingLoads.append(
        PendingLoad(
          id: id, imageryZoomBias: targetBias, generation: gridGeneration,
          reference: reference, baseTile: tile))
    }
    pendingLoads.sort {
      // Establish coarse levels first. At the same level, finish imagery before
      // starting more terrain so a visible green fallback is short-lived.
      if $0.id.z != $1.id.z { return $0.id.z < $1.id.z }
      if ($0.baseTile == nil) != ($1.baseTile == nil) { return $0.baseTile != nil }
      let a = MapStreamingPolicy.distance(to: $0.id, camera: cameraScene, reference: reference)
      let b = MapStreamingPolicy.distance(to: $1.id, camera: cameraScene, reference: reference)
      return a == b ? MapStreamingPolicy.ordered($0.id, $1.id) : a < b
    }
    updateResidentBytes()
    // The selected working set already fits. Evict retained old regions early
    // enough to leave upload headroom for the incoming region.
    for id in tiles.keys.filter({ !wantedTiles.contains($0) && !displayedTiles.contains($0) }).sorted(by: {
      (tileLastUsed[$0] ?? 0) < (tileLastUsed[$1] ?? 0)
    }) {
      guard residentBytes > MapStreamingPolicy.workingSetBudget else { break }
      tiles[id] = nil
      tileLastUsed[id] = nil
      updateResidentBytes()
    }
  }

  /// Select coverage before culling, so a missing child cannot introduce an
  /// overlapping parent. Test each resulting tile against both eye frusta.
  private func makeCoverageSet() -> Set<MapTileID> {
    let imageryRequired = imageryEnabled && google.isAvailable
    return MapStreamingPolicy.presentationCoverage(
      leaves: currentLeaves,
      imageryRequired: imageryRequired,
      isLoaded: { self.tiles[$0] != nil },
      hasImagery: { self.tiles[$0]?.texture != nil })
  }

  private func makeRenderSet(
    coverage: Set<MapTileID>, cameraScene: SIMD3<Float>,
    clipFromScene: [simd_float4x4]
  )
    -> [MapTileID]
  {
    coverage.filter { id in
      guard let tile = tiles[id] else { return false }
      let flatCenter = SIMD3<Float>(
        (tile.minSceneX + tile.maxSceneX) * 0.5,
        (tile.minElevation + tile.maxElevation) * 0.5 + verticalOffset,
        (tile.minSceneZ + tile.maxSceneZ) * 0.5)
      let curvedCenter = MapGlobeProjection.curvedPosition(flatCenter, camera: cameraScene)
      let halfWidth = (tile.maxSceneX - tile.minSceneX) * 0.5
      let halfDepth = (tile.maxSceneZ - tile.minSceneZ) * 0.5
      let halfHeight = (tile.maxElevation - tile.minElevation) * 0.5
      // The tangent-sphere map is non-expanding. A sphere around the original
      // tile bounds is therefore conservative after curvature is applied.
      let radius = simd_length(SIMD3<Float>(halfWidth, halfHeight, halfDepth)) + 1_000
      return MapFrustum.isVisible(
        minimum: curvedCenter - SIMD3<Float>(repeating: radius),
        maximum: curvedCenter + SIMD3<Float>(repeating: radius),
        clipFromScene: clipFromScene)
    }.sorted(by: MapStreamingPolicy.ordered)
  }

  private func updateResidentBytes() {
    residentBytes = tiles.values.reduce(0) { $0 + $1.gpuBytes }
  }

  private func pumpLoads() {
    var reserved = inFlight.values.reduce(0) { total, request in
      // Cancelled replacements can still retain an evicted mesh/texture until
      // their worker finishes. Include that ownership in admission accounting.
      let retainedBase = request.baseTile.map { tiles[$0.id] === $0 ? 0 : $0.gpuBytes } ?? 0
      return total + request.reservedBytes + retainedBase
    }
    while inFlight.count < Self.maximumConcurrentLoads, !pendingLoads.isEmpty {
      let next = pendingLoads[0]
      guard residentBytes + reserved + next.reservedBytes <= MapStreamingPolicy.gpuBudget else {
        return
      }
      pendingLoads.removeFirst()
      inFlight[next.id] = next
      reserved += next.reservedBytes
      let device = device
      let queue = mipmapQueue
      let google = google
      loadQueue.addOperation { [weak self] in
        let tile: WorldMapTile?
        if let base = next.baseTile {
          let texture = WorldMapTileBuilder.buildSatelliteTexture(
            device: device, id: next.id, imageryZoomBias: next.imageryZoomBias,
            commandQueue: queue, google: google,
            isCancelled: { next.cancellation.isCancelled })
          tile = texture.map { base.replacingTexture($0, bias: next.imageryZoomBias) }
        } else {
          tile = WorldMapTileBuilder.build(
            device: device, id: next.id, reference: next.reference,
            meshQuads: MapStreamingPolicy.meshQuads(zoom: next.id.z), skirtDepth: Self.skirtDepth,
            isCancelled: { next.cancellation.isCancelled })
        }
        guard let self else { return }
        self.lock.lock()
        self.readyTiles.append(
          CompletedLoad(
            request: next, tile: tile, completedAt: ProcessInfo.processInfo.systemUptime))
        self.lock.unlock()
      }
    }
  }

  private func drainReadyTiles() {
    lock.lock()
    let ready = readyTiles
    readyTiles.removeAll()
    lock.unlock()
    for result in ready {
      let request = result.request
      inFlight[request.id] = nil
      guard request.generation == gridGeneration, !request.cancellation.isCancelled,
        wantedTiles.contains(request.id) || displayedTiles.contains(request.id)
      else { continue }
      if let tile = result.tile {
        tiles[tile.id] = tile
        retries[request.id] = nil
      } else {
        var retry = retries[request.id] ?? MapRetryState()
        retry.failed(at: result.completedAt)
        retries[request.id] = retry
      }
    }
    updateResidentBytes()
  }

  // MARK: - Terrain following

  /// Keeps the camera at a roughly constant height above the ground beneath it.
  /// The vertical stick is folded into `desiredClearance` (so the player still
  /// climbs and descends), while the terrain offset follows ridges and valleys
  /// instead of letting the camera clip into a mountain.
  private func updateTerrainFollowing(
    cameraScene: SIMD3<Float>, deltaTime: Float, coverage: Set<MapTileID>
  ) {
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
    guard
      let ground = terrainSample(atSceneX: cameraScene.x, sceneZ: cameraScene.z, coverage: coverage)
    else {
      lastGroundAltitude = desiredClearance
      return
    }
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
    lastGroundAltitude = cameraScene.y - (ground.height + verticalOffset)
  }

  private func terrainSample(
    atSceneX x: Float, sceneZ z: Float, coverage: Set<MapTileID>
  ) -> (height: Float, zoom: Int)? {
    var best: WorldMapTile?
    // Sample the same non-overlapping coverage that is drawn, not an invisible
    // fine mesh still waiting for its siblings to replace a coarse fallback.
    for id in coverage {
      guard let tile = tiles[id], tile.contains(sceneX: x, sceneZ: z) else { continue }
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

  private static func makeCityLabelVertexBuffer(
    device: MTLDevice, reference: MapSceneReference
  ) throws -> (buffer: MTLBuffer, count: Int) {
    let columns = 4
    let atlasWidth: Float = 2_048
    let atlasHeight = Float(((MapCityCatalog.labels.count + columns - 1) / columns) * 128)
    let halfHeight: Float = 0.018
    let halfWidth: Float = 0.072
    var vertices: [WorldMapCityLabelVertex] = []
    vertices.reserveCapacity(MapCityCatalog.labels.count * 6)

    for (index, city) in MapCityCatalog.labels.enumerated() {
      let scene = reference.scenePosition(latitude: city.latitude, longitude: city.longitude)
      let anchor = SIMD4<Float>(scene.x, 8_000, scene.y, 1)
      let column = index % columns
      let row = index / columns
      let u0 = Float(column * 512) / atlasWidth
      let u1 = Float((column + 1) * 512) / atlasWidth
      let v0 = Float(row * 128) / atlasHeight
      let v1 = Float((row + 1) * 128) / atlasHeight
      func vertex(_ x: Float, _ y: Float, _ u: Float, _ v: Float) -> WorldMapCityLabelVertex {
        WorldMapCityLabelVertex(anchor: anchor, cornerUV: SIMD4<Float>(x, y, u, v))
      }
      vertices.append(contentsOf: [
        vertex(-halfWidth, -halfHeight, u0, v1),
        vertex(halfWidth, -halfHeight, u1, v1),
        vertex(-halfWidth, halfHeight, u0, v0),
        vertex(-halfWidth, halfHeight, u0, v0),
        vertex(halfWidth, -halfHeight, u1, v1),
        vertex(halfWidth, halfHeight, u1, v0),
      ])
    }
    guard
      let buffer = device.makeBuffer(
        bytes: vertices,
        length: vertices.count * MemoryLayout<WorldMapCityLabelVertex>.stride,
        options: .storageModeShared)
    else { throw WorldMapError.resourceAllocationFailed("city label vertices") }
    buffer.label = "WorldMap city label vertices"
    return (buffer, vertices.count)
  }

  private static func makeCityLabelTexture(device: MTLDevice) throws -> MTLTexture {
    let columns = 4
    let cellWidth = 512
    let cellHeight = 128
    let width = columns * cellWidth
    let rows = (MapCityCatalog.labels.count + columns - 1) / columns
    let height = rows * cellHeight
    var pixels = [UInt8](repeating: 0, count: width * height * 4)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
    let drew = pixels.withUnsafeMutableBytes { bytes -> Bool in
      guard
        let context = CGContext(
          data: bytes.baseAddress,
          width: width,
          height: height,
          bitsPerComponent: 8,
          bytesPerRow: width * 4,
          space: colorSpace,
          bitmapInfo: bitmapInfo)
      else { return false }

      for (index, city) in MapCityCatalog.labels.enumerated() {
        let column = index % columns
        let row = index / columns
        let cell = CGRect(
          x: CGFloat(column * cellWidth),
          y: CGFloat(height - (row + 1) * cellHeight),
          width: CGFloat(cellWidth),
          height: CGFloat(cellHeight))
        let pill = cell.insetBy(dx: 12, dy: 15)
        context.addPath(
          CGPath(
            roundedRect: pill,
            cornerWidth: 28,
            cornerHeight: 28,
            transform: nil))
        context.setFillColor(CGColor(red: 0.02, green: 0.05, blue: 0.10, alpha: 0.76))
        context.fillPath()
        context.setStrokeColor(CGColor(red: 0.82, green: 0.91, blue: 1, alpha: 0.82))
        context.setLineWidth(4)
        context.addPath(
          CGPath(
            roundedRect: pill,
            cornerWidth: 28,
            cornerHeight: 28,
            transform: nil))
        context.strokePath()

        let fontSize: CGFloat = city.name.count > 11 ? 42 : 52
        let font = CTFontCreateWithName("Helvetica-Bold" as CFString, fontSize, nil)
        let attributes: [NSAttributedString.Key: Any] = [
          NSAttributedString.Key(kCTFontAttributeName as String): font,
          NSAttributedString.Key(kCTForegroundColorAttributeName as String):
            CGColor(red: 1, green: 1, blue: 1, alpha: 0.98),
          NSAttributedString.Key(kCTStrokeColorAttributeName as String):
            CGColor(red: 0, green: 0, blue: 0, alpha: 0.9),
          NSAttributedString.Key(kCTStrokeWidthAttributeName as String): -5,
        ]
        let line = CTLineCreateWithAttributedString(
          NSAttributedString(string: city.name, attributes: attributes))
        let bounds = CTLineGetBoundsWithOptions(line, [.useGlyphPathBounds])
        context.textPosition = CGPoint(
          x: cell.midX - bounds.width / 2 - bounds.minX,
          y: cell.midY - bounds.height / 2 - bounds.minY)
        CTLineDraw(line, context)
      }
      return true
    }
    guard drew else { throw WorldMapError.resourceAllocationFailed("city label texture") }

    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
    descriptor.usage = .shaderRead
    descriptor.storageMode = .shared
    guard let texture = device.makeTexture(descriptor: descriptor) else {
      throw WorldMapError.resourceAllocationFailed("city label texture")
    }
    texture.label = "WorldMap pinyin city label atlas"
    pixels.withUnsafeBytes { bytes in
      guard let base = bytes.baseAddress else { return }
      texture.replace(
        region: MTLRegionMake2D(0, 0, width, height),
        mipmapLevel: 0,
        withBytes: base,
        bytesPerRow: width * 4)
    }
    return texture
  }

  private static func makeCityLabelPipeline(
    device: MTLDevice, library: MTLLibrary, maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    guard let vertex = library.makeFunction(name: "worldMapCityLabelVertex") else {
      throw WorldMapError.missingFunction("worldMapCityLabelVertex")
    }
    guard let fragment = library.makeFunction(name: "worldMapCityLabelFragment") else {
      throw WorldMapError.missingFunction("worldMapCityLabelFragment")
    }
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = vertex
    descriptor.fragmentFunction = fragment
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
    descriptor.depthAttachmentPixelFormat = .depth32Float
    descriptor.maxVertexAmplificationCount = max(maxViewCount, 1)
    if let attachment = descriptor.colorAttachments[0] {
      attachment.isBlendingEnabled = true
      attachment.sourceRGBBlendFactor = .one
      attachment.sourceAlphaBlendFactor = .one
      attachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
      attachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
    }
    return try device.makeRenderPipelineState(descriptor: descriptor)
  }

  private static func makeCompassVertexBuffer(device: MTLDevice) throws -> MTLBuffer {
    let radius: Float = 75
    // (north, east, u, v). The shader maps north to forward (-z) and east to
    // the viewer's right (+x), then corrects the texture for viewing below it.
    let vertices: [SIMD4<Float>] = [
      SIMD4<Float>(radius, -radius, 0, 0),
      SIMD4<Float>(radius, radius, 1, 0),
      SIMD4<Float>(-radius, -radius, 0, 1),
      SIMD4<Float>(-radius, radius, 1, 1),
    ]
    guard
      let buffer = device.makeBuffer(
        bytes: vertices,
        length: vertices.count * MemoryLayout<SIMD4<Float>>.stride,
        options: .storageModeShared)
    else { throw WorldMapError.resourceAllocationFailed("compass vertices") }
    buffer.label = "WorldMap compass vertices"
    return buffer
  }

  private static func makeCompassTexture(device: MTLDevice) throws -> MTLTexture {
    let side = 1_024
    var pixels = [UInt8](repeating: 0, count: side * side * 4)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
    let drew = pixels.withUnsafeMutableBytes { bytes -> Bool in
      guard
        let context = CGContext(
          data: bytes.baseAddress,
          width: side,
          height: side,
          bitsPerComponent: 8,
          bytesPerRow: side * 4,
          space: colorSpace,
          bitmapInfo: bitmapInfo)
      else { return false }

      let center = CGPoint(x: 512, y: 512)
      context.setFillColor(CGColor(red: 0.02, green: 0.04, blue: 0.08, alpha: 0.42))
      context.fillEllipse(in: CGRect(x: 412, y: 412, width: 200, height: 200))
      context.setStrokeColor(CGColor(red: 1, green: 1, blue: 1, alpha: 0.72))
      context.setLineWidth(12)
      context.strokeEllipse(in: CGRect(x: 424, y: 424, width: 176, height: 176))

      let directions: [(label: String, tip: CGPoint, base: CGPoint, color: CGColor)] = [
        (
          "N", CGPoint(x: 512, y: 950), CGPoint(x: 512, y: 820),
          CGColor(red: 1, green: 0.16, blue: 0.10, alpha: 0.96)
        ),
        (
          "E", CGPoint(x: 950, y: 512), CGPoint(x: 820, y: 512),
          CGColor(red: 1, green: 0.78, blue: 0.10, alpha: 0.94)
        ),
        (
          "S", CGPoint(x: 512, y: 74), CGPoint(x: 512, y: 204),
          CGColor(red: 0.20, green: 0.70, blue: 1, alpha: 0.94)
        ),
        (
          "W", CGPoint(x: 74, y: 512), CGPoint(x: 204, y: 512),
          CGColor(red: 0.32, green: 1, blue: 0.52, alpha: 0.94)
        ),
      ]
      for direction in directions {
        context.setStrokeColor(direction.color)
        context.setFillColor(direction.color)
        context.setLineWidth(30)
        context.setLineCap(.round)
        context.move(to: center)
        context.addLine(to: direction.base)
        context.strokePath()

        let dx = direction.tip.x - direction.base.x
        let dy = direction.tip.y - direction.base.y
        let length = max(hypot(dx, dy), 1)
        let px = -dy / length * 58
        let py = dx / length * 58
        context.move(to: direction.tip)
        context.addLine(to: CGPoint(x: direction.base.x + px, y: direction.base.y + py))
        context.addLine(to: CGPoint(x: direction.base.x - px, y: direction.base.y - py))
        context.closePath()
        context.fillPath()

        let labelCenter = CGPoint(
          x: direction.tip.x + dx / length * 4,
          y: direction.tip.y + dy / length * 4)
        drawCompassLabel(
          direction.label,
          center: labelCenter,
          color: direction.color,
          context: context)
      }
      return true
    }
    guard drew else { throw WorldMapError.resourceAllocationFailed("compass texture") }

    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba8Unorm,
      width: side,
      height: side,
      mipmapped: false)
    descriptor.usage = .shaderRead
    descriptor.storageMode = .shared
    guard let texture = device.makeTexture(descriptor: descriptor) else {
      throw WorldMapError.resourceAllocationFailed("compass texture")
    }
    texture.label = "WorldMap cardinal compass"
    pixels.withUnsafeBytes { bytes in
      guard let base = bytes.baseAddress else { return }
      texture.replace(
        region: MTLRegionMake2D(0, 0, side, side),
        mipmapLevel: 0,
        withBytes: base,
        bytesPerRow: side * 4)
    }
    return texture
  }

  private static func drawCompassLabel(
    _ text: String,
    center: CGPoint,
    color: CGColor,
    context: CGContext
  ) {
    let font = CTFontCreateWithName("Helvetica-Bold" as CFString, 92, nil)
    let attributes: [NSAttributedString.Key: Any] = [
      NSAttributedString.Key(kCTFontAttributeName as String): font,
      NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
      NSAttributedString.Key(kCTStrokeColorAttributeName as String):
        CGColor(red: 0, green: 0, blue: 0, alpha: 0.88),
      NSAttributedString.Key(kCTStrokeWidthAttributeName as String): -8,
    ]
    let line = CTLineCreateWithAttributedString(
      NSAttributedString(string: text, attributes: attributes))
    let bounds = CTLineGetBoundsWithOptions(line, [.useGlyphPathBounds])
    context.textPosition = CGPoint(
      x: center.x - bounds.width / 2 - bounds.minX,
      y: center.y - bounds.height / 2 - bounds.minY)
    CTLineDraw(line, context)
  }

  private static func makeCompassPipeline(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    guard let vertex = library.makeFunction(name: "worldMapCompassVertex") else {
      throw WorldMapError.missingFunction("worldMapCompassVertex")
    }
    guard let fragment = library.makeFunction(name: "worldMapCompassFragment") else {
      throw WorldMapError.missingFunction("worldMapCompassFragment")
    }
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = vertex
    descriptor.fragmentFunction = fragment
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
    descriptor.depthAttachmentPixelFormat = .depth32Float
    descriptor.maxVertexAmplificationCount = max(maxViewCount, 1)
    if let attachment = descriptor.colorAttachments[0] {
      attachment.isBlendingEnabled = true
      attachment.sourceRGBBlendFactor = .one
      attachment.sourceAlphaBlendFactor = .one
      attachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
      attachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
    }
    return try device.makeRenderPipelineState(descriptor: descriptor)
  }

  private static func makeCompassDepthState(device: MTLDevice) -> MTLDepthStencilState {
    let descriptor = MTLDepthStencilDescriptor()
    descriptor.depthCompareFunction = .greater
    descriptor.isDepthWriteEnabled = false
    return device.makeDepthStencilState(descriptor: descriptor)!
  }

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
