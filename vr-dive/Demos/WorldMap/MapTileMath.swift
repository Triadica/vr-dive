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
/// use the normal north-up viewing convention: +x is east (the viewer's right)
/// and -z is north (the viewer's forward direction). Mercator metre deltas are
/// multiplied by cos(latitude) so the scene reads as real metres at the start
/// location rather than inflated Mercator metres.
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

  /// Scene `x` (east) and `z` (negative north) in metres for a Mercator point.
  func scenePosition(mercator: SIMD2<Double>) -> SIMD2<Float> {
    let north = (mercator.y - originMercator.y) * cosLatitude
    let east = (mercator.x - originMercator.x) * cosLatitude
    return SIMD2<Float>(Float(east), Float(-north))
  }

  /// Scene `x` (east) and `z` (negative north) for a global pixel coordinate.
  func scenePosition(pixelX: Double, pixelY: Double, zoom: Int) -> SIMD2<Float> {
    scenePosition(mercator: MapProjection.mercatorMeters(pixelX: pixelX, pixelY: pixelY, zoom: zoom))
  }

  func scenePosition(latitude: Double, longitude: Double) -> SIMD2<Float> {
    let pixelX = MapProjection.longitudeToTileX(longitude, zoom: geometryZoom)
      * MapProjection.tilePixelSize
    let pixelY = MapProjection.latitudeToTileY(latitude, zoom: geometryZoom)
      * MapProjection.tilePixelSize
    return scenePosition(pixelX: pixelX, pixelY: pixelY, zoom: geometryZoom)
  }

  /// Maps a scene x/z position back to the tile that contains it.
  func tile(atSceneX sceneX: Float, sceneZ: Float, zoom: Int) -> MapTileID {
    let east = Double(sceneX)
    let north = -Double(sceneZ)
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
    let east = Double(sceneX)
    let north = -Double(sceneZ)
    let mercatorX = originMercator.x + east / cosLatitude
    let mercatorY = originMercator.y + north / cosLatitude
    return MapProjection.longitudeLatitude(mercator: SIMD2(mercatorX, mercatorY))
  }
}

nonisolated struct MapCityLabel: Sendable {
  let name: String
  let latitude: Double
  let longitude: Double
}

/// Pinyin-only labels for China's major cities and the denser Jiangsu-Zhejiang-
/// Shanghai region. Entries are ordered by display priority before the 10 km
/// geographic de-duplication pass.
nonisolated enum MapCityCatalog {
  static let minimumSpacingMeters = 10_000.0

  static let labels: [MapCityLabel] = deduplicated([
    MapCityLabel(name: "Beijing", latitude: 39.904, longitude: 116.407),
    MapCityLabel(name: "Shanghai", latitude: 31.230, longitude: 121.474),
    MapCityLabel(name: "Guangzhou", latitude: 23.129, longitude: 113.264),
    MapCityLabel(name: "Shenzhen", latitude: 22.543, longitude: 114.058),
    MapCityLabel(name: "Chongqing", latitude: 29.563, longitude: 106.551),
    MapCityLabel(name: "Chengdu", latitude: 30.572, longitude: 104.066),
    MapCityLabel(name: "Tianjin", latitude: 39.085, longitude: 117.200),
    MapCityLabel(name: "Wuhan", latitude: 30.593, longitude: 114.305),
    MapCityLabel(name: "Xi'an", latitude: 34.341, longitude: 108.940),
    MapCityLabel(name: "Nanjing", latitude: 32.060, longitude: 118.797),
    MapCityLabel(name: "Hangzhou", latitude: 30.274, longitude: 120.155),
    MapCityLabel(name: "Suzhou", latitude: 31.299, longitude: 120.585),
    MapCityLabel(name: "Zhengzhou", latitude: 34.746, longitude: 113.625),
    MapCityLabel(name: "Changsha", latitude: 28.228, longitude: 112.939),
    MapCityLabel(name: "Qingdao", latitude: 36.067, longitude: 120.383),
    MapCityLabel(name: "Jinan", latitude: 36.651, longitude: 117.120),
    MapCityLabel(name: "Shenyang", latitude: 41.805, longitude: 123.431),
    MapCityLabel(name: "Dalian", latitude: 38.914, longitude: 121.614),
    MapCityLabel(name: "Harbin", latitude: 45.803, longitude: 126.535),
    MapCityLabel(name: "Changchun", latitude: 43.817, longitude: 125.324),
    MapCityLabel(name: "Shijiazhuang", latitude: 38.042, longitude: 114.514),
    MapCityLabel(name: "Taiyuan", latitude: 37.870, longitude: 112.549),
    MapCityLabel(name: "Hefei", latitude: 31.820, longitude: 117.227),
    MapCityLabel(name: "Nanchang", latitude: 28.683, longitude: 115.858),
    MapCityLabel(name: "Fuzhou", latitude: 26.074, longitude: 119.296),
    MapCityLabel(name: "Xiamen", latitude: 24.479, longitude: 118.089),
    MapCityLabel(name: "Kunming", latitude: 25.038, longitude: 102.718),
    MapCityLabel(name: "Guiyang", latitude: 26.647, longitude: 106.630),
    MapCityLabel(name: "Nanning", latitude: 22.817, longitude: 108.366),
    MapCityLabel(name: "Haikou", latitude: 20.044, longitude: 110.199),
    MapCityLabel(name: "Lanzhou", latitude: 36.061, longitude: 103.834),
    MapCityLabel(name: "Xining", latitude: 36.617, longitude: 101.778),
    MapCityLabel(name: "Yinchuan", latitude: 38.487, longitude: 106.230),
    MapCityLabel(name: "Hohhot", latitude: 40.842, longitude: 111.749),
    MapCityLabel(name: "Urumqi", latitude: 43.825, longitude: 87.617),
    MapCityLabel(name: "Lhasa", latitude: 29.652, longitude: 91.172),
    MapCityLabel(name: "Ningbo", latitude: 29.868, longitude: 121.544),
    MapCityLabel(name: "Wuxi", latitude: 31.491, longitude: 120.312),
    MapCityLabel(name: "Changzhou", latitude: 31.811, longitude: 119.974),
    MapCityLabel(name: "Nantong", latitude: 31.980, longitude: 120.894),
    MapCityLabel(name: "Wenzhou", latitude: 27.994, longitude: 120.699),
    MapCityLabel(name: "Shaoxing", latitude: 30.030, longitude: 120.580),
    MapCityLabel(name: "Jiaxing", latitude: 30.746, longitude: 120.755),
    MapCityLabel(name: "Huzhou", latitude: 30.894, longitude: 120.086),
    MapCityLabel(name: "Jinhua", latitude: 29.079, longitude: 119.648),
    MapCityLabel(name: "Taizhou", latitude: 28.656, longitude: 121.421),
    MapCityLabel(name: "Lishui", latitude: 28.467, longitude: 119.922),
    MapCityLabel(name: "Quzhou", latitude: 28.970, longitude: 118.859),
    MapCityLabel(name: "Zhoushan", latitude: 29.985, longitude: 122.207),
    MapCityLabel(name: "Yangzhou", latitude: 32.394, longitude: 119.412),
    MapCityLabel(name: "Zhenjiang", latitude: 32.188, longitude: 119.424),
    MapCityLabel(name: "Yancheng", latitude: 33.347, longitude: 120.163),
    MapCityLabel(name: "Xuzhou", latitude: 34.205, longitude: 117.285),
    MapCityLabel(name: "Huangshan", latitude: 29.714, longitude: 118.338),
    MapCityLabel(name: "Hong Kong", latitude: 22.319, longitude: 114.169),
    MapCityLabel(name: "Macau", latitude: 22.198, longitude: 113.543),
  ])

  static func distance(_ a: MapCityLabel, _ b: MapCityLabel) -> Double {
    let latitude1 = a.latitude * .pi / 180
    let latitude2 = b.latitude * .pi / 180
    let deltaLatitude = latitude2 - latitude1
    let deltaLongitude = (b.longitude - a.longitude) * .pi / 180
    let haversine = pow(sin(deltaLatitude / 2), 2)
      + cos(latitude1) * cos(latitude2) * pow(sin(deltaLongitude / 2), 2)
    return 2 * MapProjection.earthRadius * asin(min(1, sqrt(haversine)))
  }

  private static func deduplicated(_ candidates: [MapCityLabel]) -> [MapCityLabel] {
    var result: [MapCityLabel] = []
    for candidate in candidates
    where result.allSatisfy({ distance(candidate, $0) >= minimumSpacingMeters }) {
      result.append(candidate)
    }
    return result
  }
}
