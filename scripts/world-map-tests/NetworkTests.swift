import CoreGraphics
import Foundation
import ImageIO
import Metal

final class MapStubProtocol: URLProtocol, @unchecked Sendable {
  static let stopped = DispatchSemaphore(value: 0)
  override class func canInit(with request: URLRequest) -> Bool { true }
  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
  override func startLoading() {
    let url = request.url!
    if url.path == "/slow" { return }
    var status = 200
    let data: Data
    if url.path == "/session" {
      data = Data(
        "{\"session\":\"fixture\",\"expiry\":\"9999999999\",\"tileWidth\":256,\"tileHeight\":256,\"imageFormat\":\"png\"}"
          .utf8)
    } else if url.path.contains("/2dtiles/") {
      if url.host == "partial.invalid" && url.path.hasSuffix("/1/1") { status = 503 }
      let width = url.host == "bad-size.invalid" ? 512 : 256
      let pixels = [UInt8](repeating: 128, count: width * 256 * 4)
      let provider = CGDataProvider(data: Data(pixels) as CFData)!
      let image = CGImage(
        width: width, height: 256, bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
      let encoded = NSMutableData()
      let destination = CGImageDestinationCreateWithData(encoded, "public.png" as CFString, 1, nil)!
      CGImageDestinationAddImage(destination, image, nil)
      check(CGImageDestinationFinalize(destination), "PNG fixture encoding failed")
      data = encoded as Data
    } else {
      status = url.path == "/failure" ? 503 : 200
      data = Data("fixture".utf8)
    }
    let response = HTTPURLResponse(
      url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: nil)!
    client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
    client?.urlProtocol(self, didLoad: data)
    client?.urlProtocolDidFinishLoading(self)
  }
  override func stopLoading() {
    if request.url?.path == "/slow" { Self.stopped.signal() }
  }
}

func testNetworkAndTextures() {
  let configuration = URLSessionConfiguration.ephemeral
  configuration.protocolClasses = [MapStubProtocol.self]
  let network = MapNetwork(configuration: configuration)
  check(
    network.get(URL(string: "https://fixture.invalid/success")!) == Data("fixture".utf8),
    "HTTP success lost")
  check(
    network.get(URL(string: "https://fixture.invalid/failure")!) == nil, "HTTP failure accepted")
  let cancellation = MapLoadCancellation()
  DispatchQueue.global().asyncAfter(deadline: .now() + 0.1) { cancellation.cancel() }
  let start = ProcessInfo.processInfo.systemUptime
  check(
    network.get(
      URL(string: "https://fixture.invalid/slow")!, isCancelled: { cancellation.isCancelled })
      == nil, "Cancelled request accepted")
  check(ProcessInfo.processInfo.systemUptime - start < 1, "Cancellation held load slot")
  check(
    MapStubProtocol.stopped.wait(timeout: .now() + 1) == .success,
    "Underlying HTTP task not cancelled")
  check(MemoryLayout<WorldMapVertex>.stride == 48, "CPU/Metal vertex stride mismatch")

  guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    print("PASS: HTTP failure and cancellation. SKIP: no Metal device for texture tests")
    return
  }
  func build(_ host: String, bias: Int) -> MTLTexture? {
    let google = GoogleMapsTileClient(
      apiKey: nil, proxyBaseURL: URL(string: "https://\(host)"), network: network)
    return WorldMapTileBuilder.buildSatelliteTexture(
      device: device, id: MapTileID(z: 1, x: 0, y: 0), imageryZoomBias: bias,
      commandQueue: queue, google: google)
  }
  let texture = build("fixture.invalid", bias: 1)
  check(
    texture?.width == 512 && texture?.mipmapLevelCount == 10, "Complete composite/mipmaps missing")
  check(build("partial.invalid", bias: 1) == nil, "Partial composite was published")
  check(build("bad-size.invalid", bias: 0) == nil, "Unexpected tile size accepted")
  // The next attempt succeeds after a failed composite; no persisted failure flag.
  check(build("fixture.invalid", bias: 1) != nil, "Imagery could not recover")
  if let texture {
    var pixel = [UInt8](repeating: 0, count: 4)
    texture.getBytes(&pixel, bytesPerRow: 4, from: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 9)
    check(pixel[0] > 100, "Mipmaps were not completed before publication")
  }
  let vertex = device.makeBuffer(length: 4 * 48, options: .storageModeShared)!
  let index = device.makeBuffer(length: 6 * 2, options: .storageModeShared)!
  let terrain = WorldMapTile(
    id: MapTileID(z: 15, x: 0, y: 0), vertexBuffer: vertex, indexBuffer: index,
    indexCount: 6, texture: nil, imageryZoomBias: 0, centerElevation: 100,
    heightGrid: [0, 100, 100, 0], gridSide: 2,
    minSceneX: 0, maxSceneX: 1, minSceneZ: 0, maxSceneZ: 1,
    gpuBytes: vertex.length + index.length)
  check(
    terrain.height(atSceneX: 0.5, sceneZ: 0.5) == 100,
    "Ground sample differs from rendered saddle ridge")
  let textured = terrain.replacingTexture(texture, bias: 1)
  let disabled = textured.replacingTexture(nil, bias: 0)
  check(textured.vertexBuffer === terrain.vertexBuffer, "Imagery update rebuilt geometry")
  check(
    disabled.texture == nil && disabled.gpuBytes == terrain.gpuBytes,
    "Disabling imagery retained texture budget")
  print(
    "PASS: HTTP cancellation, complete/partial/oversized imagery, retry recovery and Metal mipmaps")
}
