import Foundation
import Metal
import simd

/// CPU/GPU vertex layout. The explicit trailing pad keeps the Swift stride at
/// 48 bytes, matching the Metal struct declaration exactly.
nonisolated struct WorldMapVertex {
  var position: SIMD3<Float>
  var normal: SIMD3<Float>
  var uv: SIMD2<Float>
  var pad: SIMD2<Float> = .zero
}

/// A terrain tile that has been fetched, meshed, and uploaded to the GPU.
/// Satellite imagery is optional: when the Google key or network is
/// unavailable the tile renders with an elevation-tinted fallback colour.
nonisolated final class WorldMapTile {
  let id: MapTileID
  let vertexBuffer: MTLBuffer
  let indexBuffer: MTLBuffer
  let indexCount: Int
  let texture: MTLTexture?
  let imageryZoomBias: Int
  let centerElevation: Float
  /// CPU copy of the sampled elevation grid so the renderer can query the
  /// ground height under the camera for terrain-following altitude control.
  let heightGrid: [Float]
  let gridSide: Int
  let minSceneX: Float
  let maxSceneX: Float
  let minSceneZ: Float
  let maxSceneZ: Float
  /// Approximate resident GPU bytes (buffers + texture, mipmaps included).
  let gpuBytes: Int
  let minElevation: Float
  let maxElevation: Float

  init(
    id: MapTileID,
    vertexBuffer: MTLBuffer,
    indexBuffer: MTLBuffer,
    indexCount: Int,
    texture: MTLTexture?,
    imageryZoomBias: Int,
    centerElevation: Float,
    heightGrid: [Float],
    gridSide: Int,
    minSceneX: Float,
    maxSceneX: Float,
    minSceneZ: Float,
    maxSceneZ: Float,
    gpuBytes: Int
  ) {
    self.id = id
    self.vertexBuffer = vertexBuffer
    self.indexBuffer = indexBuffer
    self.indexCount = indexCount
    self.texture = texture
    self.imageryZoomBias = imageryZoomBias
    self.centerElevation = centerElevation
    self.heightGrid = heightGrid
    self.gridSide = gridSide
    self.minSceneX = minSceneX
    self.maxSceneX = maxSceneX
    self.minSceneZ = minSceneZ
    self.maxSceneZ = maxSceneZ
    self.gpuBytes = gpuBytes
    // Skirts are at most 150 m deep; include them in the culling bounds.
    self.minElevation = (heightGrid.min() ?? 0) - 150
    self.maxElevation = heightGrid.max() ?? 0
  }

  /// Reuses the mesh and elevation grid when imagery is replaced or disabled.
  func replacingTexture(_ texture: MTLTexture?, bias: Int) -> WorldMapTile {
    WorldMapTile(
      id: id, vertexBuffer: vertexBuffer, indexBuffer: indexBuffer, indexCount: indexCount,
      texture: texture, imageryZoomBias: bias, centerElevation: centerElevation,
      heightGrid: heightGrid, gridSide: gridSide,
      minSceneX: minSceneX, maxSceneX: maxSceneX, minSceneZ: minSceneZ, maxSceneZ: maxSceneZ,
      gpuBytes: vertexBuffer.length + indexBuffer.length
        + (texture.map { MapStreamingPolicy.textureBytes(bias: $0.width == 512 ? 1 : 0) } ?? 0))
  }

  func contains(sceneX: Float, sceneZ: Float) -> Bool {
    sceneX >= minSceneX && sceneX <= maxSceneX && sceneZ >= minSceneZ && sceneZ <= maxSceneZ
  }

  func height(atSceneX x: Float, sceneZ z: Float) -> Float? {
    guard gridSide >= 2, maxSceneX > minSceneX, maxSceneZ > minSceneZ else { return nil }
    guard x >= minSceneX, x <= maxSceneX, z >= minSceneZ, z <= maxSceneZ else { return nil }
    let quads = Float(gridSide - 1)
    // Row 0 is the northern (min z) edge; column 0 is the western (min x) edge.
    let row = (z - minSceneZ) / (maxSceneZ - minSceneZ) * quads
    let column = (x - minSceneX) / (maxSceneX - minSceneX) * quads
    return Self.meshHeight(heightGrid, side: gridSide, column: column, row: row)
  }

  private static func meshHeight(_ grid: [Float], side: Int, column: Float, row: Float) -> Float {
    let clampedColumn = min(max(column, 0), Float(side - 1))
    let clampedRow = min(max(row, 0), Float(side - 1))
    let column0 = Int(clampedColumn)
    let row0 = Int(clampedRow)
    let column1 = min(column0 + 1, side - 1)
    let row1 = min(row0 + 1, side - 1)
    let tc = clampedColumn - Float(column0)
    let tr = clampedRow - Float(row0)
    // Match [a,c,b] / [b,c,d] in the actual mesh. Bilinear interpolation
    // underestimates a saddle ridge and can put the camera below its triangles.
    let a = grid[row0 * side + column0]
    let b = grid[row0 * side + column1]
    let c = grid[row1 * side + column0]
    let d = grid[row1 * side + column1]
    if tc + tr <= 1 { return a + tc * (b - a) + tr * (c - a) }
    return d + (1 - tr) * (b - d) + (1 - tc) * (c - d)
  }
}

/// A decoded elevation source for one target tile. It may come from the exact
/// tile or, when that zoom is missing from the open dataset, from a coarser
/// ancestor whose appropriate sub-rectangle is sampled instead.
nonisolated struct DEMSource {
  let heights: [Float]
  let width: Int
  let height: Int
  let originX: Float
  let originY: Float
  let span: Float

  func elevation(ox: Float, oy: Float) -> Float {
    let x = originX + ox / Float(MapProjection.tilePixelSize) * span
    let y = originY + oy / Float(MapProjection.tilePixelSize) * span
    return MapImageDecoder.bilinearSample(
      heights,
      width: width,
      height: height,
      x: x,
      y: y)
  }
}

nonisolated enum WorldMapTileBuilder {
  /// Builds terrain independently, so a slow imagery request never delays coverage.
  static func build(
    device: MTLDevice,
    id: MapTileID,
    reference: MapSceneReference,
    meshQuads: Int,
    skirtDepth: Float,
    isCancelled: () -> Bool = { false }
  ) -> WorldMapTile? {
    guard !isCancelled(), let dem = fetchDEM(id: id, isCancelled: isCancelled),
      !isCancelled()
    else { return nil }

    let quads = max(meshQuads, 1)
    let verticesPerSide = quads + 1
    let tilePixels = Float(MapProjection.tilePixelSize)
    var vertices = [WorldMapVertex]()
    vertices.reserveCapacity(verticesPerSide * verticesPerSide)
    var heightGrid = [Float]()
    heightGrid.reserveCapacity(verticesPerSide * verticesPerSide)
    var minSceneX = Float.greatestFiniteMagnitude
    var maxSceneX = -Float.greatestFiniteMagnitude
    var minSceneZ = Float.greatestFiniteMagnitude
    var maxSceneZ = -Float.greatestFiniteMagnitude

    for row in 0...quads {
      let oy = Float(row) / Float(quads) * tilePixels
      for column in 0...quads {
        let ox = Float(column) / Float(quads) * tilePixels
        let globalPixelX = Double(id.x) * MapProjection.tilePixelSize + Double(ox)
        let globalPixelY = Double(id.y) * MapProjection.tilePixelSize + Double(oy)
        let scene = reference.scenePosition(pixelX: globalPixelX, pixelY: globalPixelY, zoom: id.z)
        let elevation = dem.elevation(ox: ox, oy: oy)
        heightGrid.append(elevation)
        minSceneX = min(minSceneX, scene.x)
        maxSceneX = max(maxSceneX, scene.x)
        minSceneZ = min(minSceneZ, scene.y)
        maxSceneZ = max(maxSceneZ, scene.y)
        vertices.append(
          WorldMapVertex(
            position: SIMD3<Float>(scene.x, elevation, scene.y),
            normal: SIMD3<Float>(0, 1, 0),
            uv: SIMD2<Float>(ox / tilePixels, oy / tilePixels)))
      }
    }

    for row in 0...quads {
      for column in 0...quads {
        let index = row * verticesPerSide + column
        let left = vertices[row * verticesPerSide + max(column - 1, 0)].position
        let right = vertices[row * verticesPerSide + min(column + 1, quads)].position
        let down = vertices[max(row - 1, 0) * verticesPerSide + column].position
        let up = vertices[min(row + 1, quads) * verticesPerSide + column].position
        var normal = simd_cross(right - left, up - down)
        if simd_length_squared(normal) < 0.000_1 {
          normal = SIMD3<Float>(0, 1, 0)
        } else {
          normal = simd_normalize(normal)
        }
        vertices[index].normal = normal
      }
    }

    var indices = [UInt16]()
    indices.reserveCapacity(quads * quads * 6)
    for row in 0..<quads {
      for column in 0..<quads {
        let a = UInt16(row * verticesPerSide + column)
        let b = UInt16(row * verticesPerSide + column + 1)
        let c = UInt16((row + 1) * verticesPerSide + column)
        let d = UInt16((row + 1) * verticesPerSide + column + 1)
        indices.append(contentsOf: [a, c, b, b, c, d])
      }
    }

    let tileSpanMeters =
      Float(MapProjection.tilePixelSize)
      / Float(MapProjection.worldPixelCount(zoom: id.z))
      * Float(2.0 * Double.pi * MapProjection.earthRadius) * Float(reference.cosLatitude)
    let effectiveSkirtDepth = max(skirtDepth, min(tileSpanMeters * 0.05, 150))
    appendSkirt(
      vertices: &vertices,
      indices: &indices,
      verticesPerSide: verticesPerSide,
      depth: effectiveSkirtDepth)

    guard
      let vertexBuffer = device.makeBuffer(
        bytes: vertices,
        length: vertices.count * MemoryLayout<WorldMapVertex>.stride,
        options: .storageModeShared),
      let indexBuffer = device.makeBuffer(
        bytes: indices,
        length: indices.count * MemoryLayout<UInt16>.stride,
        options: .storageModeShared)
    else { return nil }
    vertexBuffer.label = "WorldMap tile \(id.z)/\(id.x)/\(id.y) vertices"
    indexBuffer.label = "WorldMap tile \(id.z)/\(id.x)/\(id.y) indices"

    let centerIndex = (verticesPerSide / 2) * verticesPerSide + (verticesPerSide / 2)
    let gpuBytes = vertexBuffer.length + indexBuffer.length
    return WorldMapTile(
      id: id,
      vertexBuffer: vertexBuffer,
      indexBuffer: indexBuffer,
      indexCount: indices.count,
      texture: nil,
      imageryZoomBias: 0,
      centerElevation: heightGrid[centerIndex],
      heightGrid: heightGrid,
      gridSide: verticesPerSide,
      minSceneX: minSceneX,
      maxSceneX: maxSceneX,
      minSceneZ: minSceneZ,
      maxSceneZ: maxSceneZ,
      gpuBytes: gpuBytes)
  }

  /// Loads elevation for `id`, falling back to coarser ancestors when the open
  /// dataset has no tile at that exact zoom. The returned source remembers the
  /// sub-rectangle so the same sampling code handles both cases.
  private static func fetchDEM(id: MapTileID, isCancelled: () -> Bool) -> DEMSource? {
    let maximumFallback = 4
    for delta in 0...maximumFallback {
      guard !isCancelled() else { return nil }
      let zoom = id.z - delta
      guard zoom >= 0 else { break }
      let factor = 1 << delta
      let ancestor = MapTileID(z: zoom, x: id.x / factor, y: id.y / factor)
      guard let url = OpenDEMTileClient.tileURL(id: ancestor),
        let data = MapNetwork.shared.get(url, isCancelled: isCancelled),
        let image = MapImageDecoder.decodeRGBA(data)
      else { continue }
      let span = Float(image.width) / Float(factor)
      return DEMSource(
        heights: MapImageDecoder.terrariumHeights(image),
        width: image.width,
        height: image.height,
        originX: Float(id.x % factor) * span,
        originY: Float(id.y % factor) * span,
        span: span)
    }
    return nil
  }

  /// Extends a vertical wall downward from every tile border. Adjacent tiles
  /// sample different edge DEM pixels, so their rims can differ slightly; the
  /// skirt hides that hairline crack (and any 2:1 LOD T-junction) with real
  /// geometry instead of leaving see-through gaps in the terrain.
  private static func appendSkirt(
    vertices: inout [WorldMapVertex],
    indices: inout [UInt16],
    verticesPerSide: Int,
    depth: Float
  ) {
    guard depth > 0, verticesPerSide >= 2 else { return }
    let last = verticesPerSide - 1
    var perimeter: [Int] = []
    perimeter.reserveCapacity(verticesPerSide * 4)
    for column in 0..<verticesPerSide { perimeter.append(column) }
    for row in 1..<verticesPerSide { perimeter.append(row * verticesPerSide + last) }
    for column in stride(from: last - 1, through: 0, by: -1) {
      perimeter.append(last * verticesPerSide + column)
    }
    for row in stride(from: last - 1, through: 1, by: -1) {
      perimeter.append(row * verticesPerSide)
    }
    perimeter.append(perimeter[0])

    var skirtIndices: [UInt16] = []
    skirtIndices.reserveCapacity(perimeter.count)
    for index in perimeter {
      let top = vertices[index]
      let bottom = WorldMapVertex(
        position: SIMD3<Float>(top.position.x, top.position.y - depth, top.position.z),
        normal: top.normal,
        uv: top.uv)
      skirtIndices.append(UInt16(vertices.count))
      vertices.append(bottom)
    }

    for step in 0..<(perimeter.count - 1) {
      let topA = UInt16(perimeter[step])
      let topB = UInt16(perimeter[step + 1])
      let bottomA = skirtIndices[step]
      let bottomB = skirtIndices[step + 1]
      indices.append(contentsOf: [topA, bottomA, topB, topB, bottomA, bottomB])
    }
  }

  /// Composites the 2^bias x 2^bias Google satellite tiles covering this
  /// terrain tile into one texture. No tile is persisted anywhere; the image
  /// bytes live only long enough to upload them into the GPU texture.
  static func buildSatelliteTexture(
    device: MTLDevice,
    id: MapTileID,
    imageryZoomBias: Int,
    commandQueue: MTLCommandQueue,
    google: GoogleMapsTileClient,
    isCancelled: () -> Bool = { false }
  ) -> MTLTexture? {
    guard google.isAvailable, !isCancelled() else { return nil }
    let subdivisions = 1 << max(imageryZoomBias, 0)
    let textureSize = Int(MapProjection.tilePixelSize) * subdivisions
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba8Unorm,
      width: textureSize,
      height: textureSize,
      mipmapped: textureSize > 1)
    descriptor.usage = .shaderRead
    descriptor.storageMode = .shared
    guard let texture = device.makeTexture(descriptor: descriptor) else { return nil }
    texture.label = "WorldMap satellite \(id.z)/\(id.x)/\(id.y)"

    for subY in 0..<subdivisions {
      for subX in 0..<subdivisions {
        let subID = MapTileID(
          z: id.z + max(imageryZoomBias, 0),
          x: id.x * subdivisions + subX,
          y: id.y * subdivisions + subY)
        guard !isCancelled(),
          let data = google.satelliteTileData(id: subID, isCancelled: isCancelled),
          let image = MapImageDecoder.decodeRGBA(data),
          image.width == 256, image.height == 256, !isCancelled()
        else { return nil }
        image.pixels.withUnsafeBytes { bytes in
          guard let base = bytes.baseAddress else { return }
          texture.replace(
            region: MTLRegionMake2D(subX * 256, subY * 256, image.width, image.height),
            mipmapLevel: 0,
            withBytes: base,
            bytesPerRow: image.width * 4)
        }
      }
    }
    guard !isCancelled(),
      let commandBuffer = commandQueue.makeCommandBuffer(),
      let blit = commandBuffer.makeBlitCommandEncoder()
    else { return nil }
    do {
      blit.generateMipmaps(for: texture)
      blit.endEncoding()
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
      guard commandBuffer.status == .completed, !isCancelled() else { return nil }
    }
    return texture
  }
}
