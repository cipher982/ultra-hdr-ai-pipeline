import Foundation
import CoreGraphics
import ImageIO

func dumpAux(_ url: URL) {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else {
        print("ERR: cannot open: \(url.path)")
        return
    }
    let count = CGImageSourceGetCount(src)
    print("Image has \(count) item(s)")
    var error: Unmanaged<CFError>?
    if #available(macOS 15.0, *) {
        if let info = CGImageSourceCopyAuxiliaryDataInfoAtIndex(src, 0, kCGImageAuxiliaryDataTypeISOGainMap, &error) as? [CFString: Any] {
            print("Found ISO GainMap aux")
            if let meta = info[kCGImageAuxiliaryDataInfoMetadata] as? CGImageMetadata,
               let xml = CGImageMetadataCopyXMPData(meta, nil as CFDictionary?) as Data? {
                print(String(data: xml, encoding: .utf8) ?? "<xmp decode fail>")
            }
        }
    }
    if let info = CGImageSourceCopyAuxiliaryDataInfoAtIndex(src, 0, kCGImageAuxiliaryDataTypeHDRGainMap, &error) as? [CFString: Any] {
        print("Found Apple HDRGainMap aux")
        if let meta = info[kCGImageAuxiliaryDataInfoMetadata] as? CGImageMetadata,
           let xml = CGImageMetadataCopyXMPData(meta, nil as CFDictionary?) as Data? {
            print(String(data: xml, encoding: .utf8) ?? "<xmp decode fail>")
        }
    }
}

if CommandLine.arguments.count < 2 {
    print("Usage: gainmap_inspect <image.heic|.jpg>")
    exit(2)
}
dumpAux(URL(fileURLWithPath: CommandLine.arguments[1]))
