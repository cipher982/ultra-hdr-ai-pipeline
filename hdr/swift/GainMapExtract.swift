import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

func writeImage(_ img: CGImage, to url: URL, type: CFString) throws {
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, type, 1, nil) else {
        throw NSError(domain: "Extract", code: 1, userInfo: [NSLocalizedDescriptionKey: "dest create fail"])
    }
    CGImageDestinationAddImage(dest, img, nil)
    if !CGImageDestinationFinalize(dest) { throw NSError(domain: "Extract", code: 2, userInfo: [NSLocalizedDescriptionKey: "finalize fail"]) }
}

func extractGainMap(from srcURL: URL, outBase: URL, outGain: URL, outXMP: URL) throws {
    guard let src = CGImageSourceCreateWithURL(srcURL as CFURL, nil) else { throw NSError(domain: "Extract", code: 3, userInfo: [NSLocalizedDescriptionKey: "open fail"]) }
    // Write base image (item 0)
    if let base = CGImageSourceCreateImageAtIndex(src, 0, nil) {
        let t: CFString
        if #available(macOS 11.0, *) { t = UTType.jpeg.identifier as CFString } else { t = kUTTypeJPEG }
        try writeImage(base, to: outBase, type: t)
    }
    // Try ISO then Apple aux types
    var infoAny: CFDictionary?
    if #available(macOS 15.0, *) {
        if let d = CGImageSourceCopyAuxiliaryDataInfoAtIndex(src, 0, kCGImageAuxiliaryDataTypeISOGainMap) { infoAny = d }
    }
    if infoAny == nil {
        if let d = CGImageSourceCopyAuxiliaryDataInfoAtIndex(src, 0, kCGImageAuxiliaryDataTypeHDRGainMap) { infoAny = d }
    }
    guard let info = infoAny as? [CFString: Any] else { throw NSError(domain: "Extract", code: 4, userInfo: [NSLocalizedDescriptionKey: "no gainmap aux"]) }
    let data = info[kCGImageAuxiliaryDataInfoData] as! CFData
    let desc = info[kCGImageAuxiliaryDataInfoDataDescription] as! [CFString: Any]
    let w = desc[kCGImagePropertyPixelWidth] as! Int
    let h = desc[kCGImagePropertyPixelHeight] as! Int
    let bpr = desc[kCGImagePropertyBytesPerRow] as! Int
    let cs = CGColorSpaceCreateDeviceGray()
    guard let prov = CGDataProvider(data: data) else { throw NSError(domain: "Extract", code: 7, userInfo: [NSLocalizedDescriptionKey: "prov fail"]) }
    let bmpInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
    guard let gm = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: bpr, space: cs, bitmapInfo: bmpInfo, provider: prov, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
        throw NSError(domain: "Extract", code: 8, userInfo: [NSLocalizedDescriptionKey: "img build fail"]) }
    let tPNG: CFString
    if #available(macOS 11.0, *) { tPNG = UTType.png.identifier as CFString } else { tPNG = kUTTypePNG }
    try writeImage(gm, to: outGain, type: tPNG)
    // Optional: write XMP if available in the dictionary (skip if symbol unavailable)
}

if CommandLine.arguments.count < 3 {
    fputs("Usage: gainmap_extract <in.heic|.jpg> <out-stem>\n", stderr)
    exit(2)
}
let inURL = URL(fileURLWithPath: CommandLine.arguments[1])
let stem = CommandLine.arguments[2]
let baseURL = URL(fileURLWithPath: stem + "_base.jpg")
let gainURL = URL(fileURLWithPath: stem + "_gain.png")
let xmpURL = URL(fileURLWithPath: stem + "_gain.xmp")
do {
    try extractGainMap(from: inURL, outBase: baseURL, outGain: gainURL, outXMP: xmpURL)
    print("Extracted base=\(baseURL.path), gain=\(gainURL.path), xmp=\(xmpURL.path)")
} catch {
    fputs("extract error: \(error)\n", stderr)
    exit(1)
}
