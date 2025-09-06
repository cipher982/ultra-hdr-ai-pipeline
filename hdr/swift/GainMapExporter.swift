import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

// Build: swiftc -O GainMapExporter.swift -o gainmap_exporter
// Usage: gainmap_exporter <base_sdr.jpg> <gainmap_gray.png> <out.jpg> <cap_min> <cap_max> <gamma>

func loadCGImage(_ url: URL) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
}

func pixelData(from cgImage: CGImage) -> Data? {
    let width = cgImage.width
    let height = cgImage.height
    let bytesPerPixel = 1
    let bytesPerRow = width * bytesPerPixel
    var data = Data(count: height * bytesPerRow)
    data.withUnsafeMutableBytes { (ptr: UnsafeMutableRawBufferPointer) in
        if let ctx = CGContext(data: ptr.baseAddress,
                               width: width,
                               height: height,
                               bitsPerComponent: 8,
                               bytesPerRow: bytesPerRow,
                               space: CGColorSpaceCreateDeviceGray(),
                               bitmapInfo: CGImageAlphaInfo.none.rawValue) {
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
    }
    return data
}

func makeXMPMetadata(gainMinLog2: Double, gainMaxLog2: Double, gamma: Double, offsetSDR: Double, offsetHDR: Double, capMin: Double, capMax: Double) -> CGImageMetadata? {
    // Build ISO hdrgm XMP with xpacket wrapper as seen in working samples
    func fmt(_ v: Double) -> String { String(format: "%.6f", v) }
    let xmp = """
    <?xpacket begin="\u{FEFF}" id="W5M0MpCehiHzreSzNTczkc9d"?>
    <x:xmpmeta xmlns:x="adobe:ns:meta/">
      <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
        <rdf:Description xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/"
          hdrgm:Version="1.0"
          hdrgm:BaseRenditionIsHDR="False"
          hdrgm:GainMapMin="\(fmt(gainMinLog2))"
          hdrgm:GainMapMax="\(fmt(gainMaxLog2))"
          hdrgm:Gamma="\(fmt(gamma))"
          hdrgm:OffsetSDR="\(fmt(offsetSDR))"
          hdrgm:OffsetHDR="\(fmt(offsetHDR))"
          hdrgm:HDRCapacityMin="\(fmt(capMin))"
          hdrgm:HDRCapacityMax="\(fmt(capMax))"/>
      </rdf:RDF>
    </x:xmpmeta>
    <?xpacket end="w"?>
    """
    guard let data = xmp.data(using: .utf8) else { return nil }
    return CGImageMetadataCreateFromXMPData(data as CFData)
}

func destTypeForPath(_ path: String) -> CFString {
    let ext = (path as NSString).pathExtension.lowercased()
    if #available(macOS 11.0, *) {
        if ext == "heic" { return UTType.heic.identifier as CFString }
        return UTType.jpeg.identifier as CFString
    } else {
        return kUTTypeJPEG
    }
}

func copyAuxMetadata(from url: URL, preferISO: Bool) -> (meta: CGImageMetadata?, type: CFString?) {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return (nil, nil) }
    if #available(macOS 15.0, *), preferISO {
        if let info = CGImageSourceCopyAuxiliaryDataInfoAtIndex(src, 0, kCGImageAuxiliaryDataTypeISOGainMap) as? [CFString: Any] {
            let meta = unsafeBitCast(info[kCGImageAuxiliaryDataInfoMetadata]!, to: CGImageMetadata.self)
            return (meta, kCGImageAuxiliaryDataTypeISOGainMap)
        }
    }
    if let info = CGImageSourceCopyAuxiliaryDataInfoAtIndex(src, 0, kCGImageAuxiliaryDataTypeHDRGainMap) as? [CFString: Any] {
        let meta = unsafeBitCast(info[kCGImageAuxiliaryDataInfoMetadata]!, to: CGImageMetadata.self)
        return (meta, kCGImageAuxiliaryDataTypeHDRGainMap)
    }
    return (nil, nil)
}

func export(base basePath: String, gainMap gmPath: String, out outPath: String,
            gainMinLog2: Double, gainMaxLog2: Double, gamma: Double,
            offsetSDR: Double, offsetHDR: Double, capMin: Double, capMax: Double,
            refPath: String?) throws {
    let baseURL = URL(fileURLWithPath: basePath)
    let gmURL = URL(fileURLWithPath: gmPath)
    let outURL = URL(fileURLWithPath: outPath)
    guard let baseImg = loadCGImage(baseURL) else { throw NSError(domain: "GainMapExporter", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to load base image"]) }
    guard let gmImg = loadCGImage(gmURL) else { throw NSError(domain: "GainMapExporter", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to load gain map"]) }

    let destType = destTypeForPath(outPath)
    guard let dest = CGImageDestinationCreateWithURL(outURL as CFURL, destType, 1, nil) else {
        throw NSError(domain: "GainMapExporter", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create destination"])
    }

    // Add base image first
    CGImageDestinationAddImage(dest, baseImg, nil)

    // Prepare auxiliary data dictionary for HDR gain map (best-effort)
    guard let gmData = pixelData(from: gmImg) else {
        throw NSError(domain: "GainMapExporter", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to read gain map pixels"])
    }
    let width = gmImg.width
    let height = gmImg.height
    let bytesPerRow = width * 1
    var auxDesc: [CFString: Any] = [
        kCGImageAuxiliaryDataInfoData: gmData as CFData,
        kCGImageAuxiliaryDataInfoDataDescription: [
            kCGImagePropertyPixelWidth: width,
            kCGImagePropertyPixelHeight: height,
            kCGImagePropertyBytesPerRow: bytesPerRow,
            kCGImagePropertyDepth: 8,
            kCGImagePropertyColorModel: kCGImagePropertyColorModelGray
        ] as CFDictionary,
    ]
    var auxType: CFString
    if let refPath = refPath {
        let (meta, t) = copyAuxMetadata(from: URL(fileURLWithPath: refPath), preferISO: true)
        if let m = meta, let t = t {
            auxDesc[kCGImageAuxiliaryDataInfoMetadata] = m
            auxType = t
            fputs("[export] using aux metadata from ref (type=\(t))\n", stderr)
        }
        else {
            // Fallback to ISO on new OS with generated metadata
            if #available(macOS 15.0, *) { auxType = kCGImageAuxiliaryDataTypeISOGainMap } else { auxType = kCGImageAuxiliaryDataTypeHDRGainMap }
            if let meta = makeXMPMetadata(gainMinLog2: gainMinLog2, gainMaxLog2: gainMaxLog2, gamma: gamma, offsetSDR: offsetSDR, offsetHDR: offsetHDR, capMin: capMin, capMax: capMax) {
                auxDesc[kCGImageAuxiliaryDataInfoMetadata] = meta
            }
            fputs("[export] generated ISO hdrgm metadata (type=\(auxType))\n", stderr)
        }
    } else {
        if #available(macOS 15.0, *) { auxType = kCGImageAuxiliaryDataTypeISOGainMap } else { auxType = kCGImageAuxiliaryDataTypeHDRGainMap }
        if let meta = makeXMPMetadata(gainMinLog2: gainMinLog2, gainMaxLog2: gainMaxLog2, gamma: gamma, offsetSDR: offsetSDR, offsetHDR: offsetHDR, capMin: capMin, capMax: capMax) {
            auxDesc[kCGImageAuxiliaryDataInfoMetadata] = meta
        }
        fputs("[export] generated ISO hdrgm metadata (type=\(auxType))\n", stderr)
    }

    CGImageDestinationAddAuxiliaryDataInfo(dest, auxType, auxDesc as CFDictionary)
    fputs("[export] destType=\(destType) base=\(baseImg.width)x\(baseImg.height) gm=\(width)x\(height) bytesPerRow=\(bytesPerRow)\n", stderr)

    // Finalize
    if !CGImageDestinationFinalize(dest) {
        throw NSError(domain: "GainMapExporter", code: 5, userInfo: [NSLocalizedDescriptionKey: "Failed to finalize image"])
    }
}

func main() {
    let args = CommandLine.arguments
    guard args.count >= 8 else {
        fputs("Usage: gainmap_exporter <base.jpg> <gainmap.png> <out.(jpg|heic)> <gainMinLog2> <gainMaxLog2> <gamma> <offsetSDR> <offsetHDR> [capMinLog2 capMaxLog2 ref_path]\n", stderr)
        exit(2)
    }
    let base = args[1]
    let gm = args[2]
    let out = args[3]
    let gainMin = Double(args[4]) ?? 0.0
    let gainMax = Double(args[5]) ?? 3.0
    let gamma = Double(args[6]) ?? 1.0
    let offsetSDR = Double(args[7]) ?? 0.015625
    let offsetHDR = args.count > 8 ? (Double(args[8]) ?? 0.015625) : 0.015625
    let capMin = args.count > 9 ? (Double(args[9]) ?? max(0.0, gainMin)) : max(0.0, gainMin)
    let capMax = args.count > 10 ? (Double(args[10]) ?? gainMax) : gainMax
    let ref = args.count > 11 ? args[11] : nil
    do {
        try export(base: base, gainMap: gm, out: out,
                   gainMinLog2: gainMin, gainMaxLog2: gainMax, gamma: gamma,
                   offsetSDR: offsetSDR, offsetHDR: offsetHDR, capMin: capMin, capMax: capMax,
                   refPath: ref)
    } catch {
        fputs("Exporter error: \(error)\n", stderr)
        exit(1)
    }
}

main()
