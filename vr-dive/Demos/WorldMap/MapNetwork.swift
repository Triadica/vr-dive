import CoreGraphics
import Foundation
import ImageIO
import Metal

/// Reads the Google Maps API key that the build injects from the untracked
/// `.env` file. The key never lives in source control. If it is missing the
/// map still renders open DEM terrain and simply omits satellite imagery.
nonisolated enum MapSecrets {
  static func googleMapsAPIKey() -> String? {
    value(forKey: "VITE_GOOGLE_MAPS_API_KEY")
  }

  /// Optional backend base URL (e.g. `https://maps.example.com`). When set, the
  /// client routes session and tile requests through it, so the Google key can
  /// stay server-side and never ship in the app bundle.
  static func googleMapsProxyBaseURL() -> String? {
    value(forKey: "VITE_GOOGLE_MAPS_PROXY")
  }

  private static func value(forKey key: String) -> String? {
    guard let url = Bundle.main.url(forResource: "GoogleMaps", withExtension: "env") else {
      return nil
    }
    guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
    for line in text.split(separator: "\n") {
      let trimmed = line.trimmingCharacters(in: .whitespaces)
      guard !trimmed.hasPrefix("#") else { continue }
      let parts = trimmed.split(separator: "=", maxSplits: 1)
      guard parts.count == 2 else { continue }
      let name = parts[0].trimmingCharacters(in: .whitespaces)
      if name == key {
        let value = parts[1].trimmingCharacters(in: .whitespaces)
        return value.isEmpty ? nil : value
      }
    }
    return nil
  }
}

/// A tiny synchronous HTTP layer built on an ephemeral URL session.
///
/// The session deliberately has no URL cache. Google's Map Tiles API policies
/// forbid pre-fetching, storing, or caching tile content, so tiles are only
/// held in memory for as long as they are displayed and are re-requested on
/// the next run.
nonisolated final class MapNetwork {
  static let shared = MapNetwork()

  private let session: URLSession

  private init() {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
    configuration.urlCache = nil
    configuration.timeoutIntervalForRequest = 20
    configuration.timeoutIntervalForResource = 30
    configuration.httpMaximumConnectionsPerHost = 6
    session = URLSession(configuration: configuration)
  }

  func get(_ url: URL) -> Data? {
    var request = URLRequest(url: url)
    request.httpMethod = "GET"
    request.cachePolicy = .reloadIgnoringLocalCacheData
    return perform(request)
  }

  func post(_ url: URL, jsonBody: [String: Any]) -> Data? {
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.cachePolicy = .reloadIgnoringLocalCacheData
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    guard let body = try? JSONSerialization.data(withJSONObject: jsonBody) else { return nil }
    request.httpBody = body
    return perform(request)
  }

  private func perform(_ request: URLRequest) -> Data? {
    let semaphore = DispatchSemaphore(value: 0)
    var output: Data?
    let task = session.dataTask(with: request) { data, response, _ in
      if let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) {
        output = data
      }
      semaphore.signal()
    }
    task.resume()
    _ = semaphore.wait(timeout: .now() + 30)
    return output
  }
}

nonisolated struct GoogleMapsSessionResponse: Decodable {
  let session: String
  let expiry: String
  let tileWidth: Int
  let tileHeight: Int
  let imageFormat: String
}

/// Minimal client for Google's Map Tiles API 2D satellite imagery.
///
/// This is the "custom non-web client" use case the Map Tiles API explicitly
/// supports. Session tokens are created on demand and reused until they expire.
/// Attribution ("Google Maps") is surfaced by the renderer for display.
nonisolated final class GoogleMapsTileClient {
  private let apiKey: String?
  private let proxyBase: URL?
  private let network = MapNetwork.shared
  private let lock = NSLock()
  private var cachedSession: String?
  private var cachedExpiry: Date?

  init() {
    apiKey = MapSecrets.googleMapsAPIKey()
    if let proxy = MapSecrets.googleMapsProxyBaseURL() {
      proxyBase = URL(string: proxy)
    } else {
      proxyBase = nil
    }
  }

  /// Available directly with a key, or through a configured backend proxy.
  var isAvailable: Bool { apiKey != nil || proxyBase != nil }

  /// Returns a session token, creating one if needed. Safe to call from any
  /// background queue.
  func sessionToken() -> String? {
    lock.lock()
    defer { lock.unlock() }
    if let token = cachedSession, let expiry = cachedExpiry, expiry > Date() {
      return token
    }
    let url: URL?
    if let proxyBase {
      // The proxy holds the key and exposes the same createSession shape.
      url = proxyBase.appendingPathComponent("session")
    } else if let key = apiKey {
      url = URL(string: "https://tile.googleapis.com/v1/createSession?key=\(key)")
    } else {
      url = nil
    }
    guard let url else { return nil }
    guard
      let data = network.post(
        url,
        jsonBody: ["mapType": "satellite", "language": "en-US", "region": "US"]),
      let response = try? JSONDecoder().decode(GoogleMapsSessionResponse.self, from: data)
    else {
      return nil
    }
    cachedSession = response.session
    if let seconds = TimeInterval(response.expiry) {
      cachedExpiry = Date(timeIntervalSince1970: seconds)
    } else {
      cachedExpiry = Date().addingTimeInterval(60 * 60)
    }
    return response.session
  }

  func satelliteTileURL(id: MapTileID) -> URL? {
    guard let token = sessionToken() else { return nil }
    if let proxyBase {
      var components = URLComponents(
        url: proxyBase.appendingPathComponent("2dtiles/\(id.z)/\(id.x)/\(id.y)"),
        resolvingAgainstBaseURL: false)
      components?.queryItems = [URLQueryItem(name: "session", value: token)]
      return components?.url
    }
    guard let key = apiKey else { return nil }
    return URL(
      string:
        "https://tile.googleapis.com/v1/2dtiles/\(id.z)/\(id.x)/\(id.y)?session=\(token)&key=\(key)"
    )
  }

  static let attribution = "Google Maps"
}

/// Public terrain elevation tiles (AWS Open Data / Mapzen Terrarium).
/// Openly licensed, no key required, and deliberately separate from the
/// Google content so terrain can load even when satellite imagery cannot.
nonisolated enum OpenDEMTileClient {
  static let attribution = "AWS Terrain Tiles (Mapzen Terrarium)"

  static func tileURL(id: MapTileID) -> URL? {
    URL(
      string: "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/\(id.z)/\(id.x)/\(id.y).png"
    )
  }
}

nonisolated enum MapImageDecoder {
  nonisolated struct RGBAImage {
    let width: Int
    let height: Int
    let pixels: [UInt8]
  }

  static func decodeRGBA(_ data: Data) -> RGBAImage? {
    guard let source = CGImageSourceCreateWithData(data as CFData, nil),
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else { return nil }
    let width = image.width
    let height = image.height
    guard width > 0, height > 0 else { return nil }
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
      context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
      return true
    }
    return drew ? RGBAImage(width: width, height: height, pixels: pixels) : nil
  }

  /// Decodes a Terrarium-encoded elevation PNG into metres.
  static func terrariumHeights(_ image: RGBAImage) -> [Float] {
    var heights = [Float](repeating: 0, count: image.width * image.height)
    for index in 0..<(image.width * image.height) {
      let base = index * 4
      let red = Float(image.pixels[base])
      let green = Float(image.pixels[base + 1])
      let blue = Float(image.pixels[base + 2])
      var height = (red * 256.0 + green + blue / 256.0) - 32_768.0
      // Terrarium uses very negative values for voids / no-data. Treat them as
      // sea level rather than drawing kilometre-deep pits.
      if height < -500.0 { height = 0 }
      heights[index] = height
    }
    return heights
  }

  static func bilinearSample(
    _ values: [Float],
    width: Int,
    height: Int,
    x: Float,
    y: Float
  ) -> Float {
    let clampedX = min(max(x, 0), Float(width - 1))
    let clampedY = min(max(y, 0), Float(height - 1))
    let x0 = Int(clampedX)
    let y0 = Int(clampedY)
    let x1 = min(x0 + 1, width - 1)
    let y1 = min(y0 + 1, height - 1)
    let tx = clampedX - Float(x0)
    let ty = clampedY - Float(y0)
    let top = values[y0 * width + x0] * (1 - tx) + values[y0 * width + x1] * tx
    let bottom = values[y1 * width + x0] * (1 - tx) + values[y1 * width + x1] * tx
    return top * (1 - ty) + bottom * ty
  }
}
