import Foundation
import simd

/// Identifies a single Web Mercator slippy-map tile.
nonisolated struct MapTileID: Hashable {
  let z: Int
  let x: Int
  let y: Int

  /// The containing tile one zoom level coarser.
  var parent: MapTileID {
    MapTileID(z: z - 1, x: x / 2, y: y / 2)
  }
}

/// Web Mercator helpers shared by the terrain (DEM) and satellite (Google)
/// tile sources. Both datasets use the same EPSG:3857 tile grid, so a tile at
/// a given (z, x, y) always covers exactly the same ground footprint.
nonisolated enum MapProjection {
  static let earthRadius: Double = 6_378_137.0
  static let tilePixelSize: Double = 256.0

  static func tileCount(zoom: Int) -> Double {
    pow(2.0, Double(zoom))
  }

  static func worldPixelCount(zoom: Int) -> Double {
    tilePixelSize * tileCount(zoom: zoom)
  }

  static func longitudeToTileX(_ longitude: Double, zoom: Int) -> Double {
    (longitude + 180.0) / 360.0 * tileCount(zoom: zoom)
  }

  static func latitudeToTileY(_ latitude: Double, zoom: Int) -> Double {
    let clamped = min(max(latitude, -85.051_128_78), 85.051_128_78)
    let latitudeRadians = clamped * .pi / 180.0
    let mercator = log(tan(.pi / 4.0 + latitudeRadians / 2.0))
    return (1.0 - mercator / .pi) / 2.0 * tileCount(zoom: zoom)
  }

  static func tileXToLongitude(_ x: Double, zoom: Int) -> Double {
    x / tileCount(zoom: zoom) * 360.0 - 180.0
  }

  static func tileYToLatitude(_ y: Double, zoom: Int) -> Double {
    let n = .pi * (1.0 - 2.0 * y / tileCount(zoom: zoom))
    return atan(sinh(n)) * 180.0 / .pi
  }

  /// Converts a global pixel coordinate at `zoom` into Mercator metres.
  /// x increases east, y increases north.
  static func mercatorMeters(pixelX: Double, pixelY: Double, zoom: Int) -> SIMD2<Double> {
    let pixelCount = worldPixelCount(zoom: zoom)
    let x = (pixelX / pixelCount - 0.5) * 2.0 * .pi * earthRadius
    let y = (0.5 - pixelY / pixelCount) * 2.0 * .pi * earthRadius
    return SIMD2(x, y)
  }

  static func longitudeLatitude(mercator: SIMD2<Double>) -> (
    longitude: Double, latitude: Double
  ) {
    let longitude = mercator.x / earthRadius * 180.0 / .pi
    let latitude = (2.0 * atan(exp(mercator.y / earthRadius)) - .pi / 2.0) * 180.0 / .pi
    return (longitude, latitude)
  }

  static func tileCenterScenePosition(
    id: MapTileID,
    reference: MapSceneReference
  ) -> SIMD2<Float> {
    let centerPixelX = (Double(id.x) + 0.5) * tilePixelSize
    let centerPixelY = (Double(id.y) + 0.5) * tilePixelSize
    let mercator = mercatorMeters(pixelX: centerPixelX, pixelY: centerPixelY, zoom: id.z)
    return reference.scenePosition(mercator: mercator)
  }
}

/// Describes the fixed local tangent frame the whole map is rendered in.
///
/// The origin sits at the configured start coordinate. Horizontal scene axes
/// follow the existing project convention: +x is north, -z is east. Mercator
/// metre deltas are multiplied by cos(latitude) so the scene reads as real
/// metres at the start location rather than inflated Mercator metres.
nonisolated struct MapSceneReference {
  let geometryZoom: Int
  let originMercator: SIMD2<Double>
  let cosLatitude: Double
  let latitude: Double
  let longitude: Double

  init(latitude: Double, longitude: Double, geometryZoom: Int) {
    self.geometryZoom = geometryZoom
    self.latitude = latitude
    self.longitude = longitude
    self.cosLatitude = max(cos(latitude * .pi / 180.0), 0.01)
    let originPixelX = MapProjection.longitudeToTileX(longitude, zoom: geometryZoom)
      * MapProjection.tilePixelSize
    let originPixelY = MapProjection.latitudeToTileY(latitude, zoom: geometryZoom)
      * MapProjection.tilePixelSize
    self.originMercator = MapProjection.mercatorMeters(
      pixelX: originPixelX, pixelY: originPixelY, zoom: geometryZoom)
  }

  /// Scene `x` (north) and `z` (negative east) in metres for a Mercator point.
  func scenePosition(mercator: SIMD2<Double>) -> SIMD2<Float> {
    let north = (mercator.y - originMercator.y) * cosLatitude
    let east = (mercator.x - originMercator.x) * cosLatitude
    return SIMD2<Float>(Float(north), Float(-east))
  }

  /// Scene `x` (north) and `z` (negative east) for a global pixel coordinate.
  func scenePosition(pixelX: Double, pixelY: Double, zoom: Int) -> SIMD2<Float> {
    scenePosition(mercator: MapProjection.mercatorMeters(pixelX: pixelX, pixelY: pixelY, zoom: zoom))
  }

  /// Maps a scene x/z position back to the tile that contains it.
  func tile(atSceneX sceneX: Float, sceneZ: Float, zoom: Int) -> MapTileID {
    let north = Double(sceneX)
    let east = -Double(sceneZ)
    let mercatorX = originMercator.x + east / cosLatitude
    let mercatorY = originMercator.y + north / cosLatitude
    let pixelCount = MapProjection.worldPixelCount(zoom: zoom)
    let pixelX = (mercatorX / (2.0 * .pi * MapProjection.earthRadius) + 0.5) * pixelCount
    let pixelY = (0.5 - mercatorY / (2.0 * .pi * MapProjection.earthRadius)) * pixelCount
    let tileCount = Int(MapProjection.tileCount(zoom: zoom))
    let tileX = min(max(Int(floor(pixelX / MapProjection.tilePixelSize)), 0), tileCount - 1)
    let tileY = min(max(Int(floor(pixelY / MapProjection.tilePixelSize)), 0), tileCount - 1)
    return MapTileID(z: zoom, x: tileX, y: tileY)
  }

  /// Geographic coordinate of a scene position, used by the flight readout.
  func coordinate(sceneX: Float, sceneZ: Float) -> (longitude: Double, latitude: Double) {
    let north = Double(sceneX)
    let east = -Double(sceneZ)
    let mercatorX = originMercator.x + east / cosLatitude
    let mercatorY = originMercator.y + north / cosLatitude
    return MapProjection.longitudeLatitude(mercator: SIMD2(mercatorX, mercatorY))
  }
}
