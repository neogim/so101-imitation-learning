import AVFoundation
import Accelerate
import Foundation
import Darwin

func diagnostic(_ event: String, device: AVCaptureDevice? = nil) {
    var row: [String: Any] = ["event": event]
    if let device = device {
        row["unique_id"] = device.uniqueID
        row["localized_name"] = device.localizedName
        row["model_id"] = device.modelID
        row["manufacturer"] = device.manufacturer
        row["device_type"] = device.deviceType.rawValue
        row["position"] = device.position.rawValue
        row["connected"] = device.isConnected
        row["suspended"] = device.isSuspended
        row["in_use_by_another_application"] = device.isInUseByAnotherApplication
    }
    if let data = try? JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]),
       let text = String(data: data, encoding: .utf8) {
        fputs("SO101_AVFOUNDATION_DIAGNOSTIC \(text)\n", stderr)
        fflush(stderr)
    }
}

// Binary stdout: little-endian UInt64 monotonic nanoseconds, UInt32 width,
// UInt32 height, UInt32 payload bytes, followed by packed BGRA rows.
final class Sink: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput buffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixel = CMSampleBufferGetImageBuffer(buffer) else { return }
        CVPixelBufferLockBaseAddress(pixel, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixel, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixel) else { return }
        let width = CVPixelBufferGetWidth(pixel)
        let height = CVPixelBufferGetHeight(pixel)
        let stride = CVPixelBufferGetBytesPerRow(pixel)
        guard width > 0, height > 0, width < 4096, height < 4096 else { return }
        var clock = timespec()
        clock_gettime(CLOCK_MONOTONIC, &clock)
        var timestamp = UInt64(clock.tv_sec) * 1_000_000_000 + UInt64(clock.tv_nsec)
        let outWidth = 320
        let outHeight = 240
        var w = UInt32(outWidth).littleEndian
        var h = UInt32(outHeight).littleEndian
        var length = UInt32(outWidth * outHeight * 4).littleEndian
        timestamp = timestamp.littleEndian
        var header = Data()
        withUnsafeBytes(of: &timestamp) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: &w) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: &h) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: &length) { header.append(contentsOf: $0) }
        var frame = Data(count: outWidth * outHeight * 4)
        frame.withUnsafeMutableBytes { dest in
            guard let target = dest.baseAddress else { return }
            if width == outWidth && height == outHeight {
                for row in 0..<height {
                    memcpy(target.advanced(by: row * outWidth * 4),
                           base.advanced(by: row * stride), outWidth * 4)
                }
            } else {
                var src = vImage_Buffer(data: base, height: vImagePixelCount(height),
                                        width: vImagePixelCount(width), rowBytes: stride)
                var dst = vImage_Buffer(data: target, height: vImagePixelCount(outHeight),
                                        width: vImagePixelCount(outWidth), rowBytes: outWidth * 4)
                vImageScale_ARGB8888(&src, &dst, nil, vImage_Flags(kvImageNoFlags))
            }
        }
        FileHandle.standardOutput.write(header)
        FileHandle.standardOutput.write(frame)
    }
}

guard CommandLine.arguments.count == 2 else { exit(2) }
let uniqueID = CommandLine.arguments[1]
let discovery = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.externalUnknown, .builtInWideAngleCamera],
    mediaType: .video, position: .unspecified)
for candidate in discovery.devices {
    diagnostic("discovered", device: candidate)
}
guard let device = discovery.devices.first(where: { $0.uniqueID == uniqueID }) else {
    fputs("AVFoundation uniqueID not found\n", stderr)
    exit(3)
}
diagnostic("selected", device: device)
let session = AVCaptureSession()
session.beginConfiguration()
guard let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) else {
    fputs("AVFoundation input unavailable\n", stderr)
    exit(4)
}
session.addInput(input)
diagnostic("input_added", device: input.device)
if session.canSetSessionPreset(.qvga320x240) {
    session.sessionPreset = .qvga320x240
} else if session.canSetSessionPreset(.vga640x480) {
    session.sessionPreset = .vga640x480
}
let output = AVCaptureVideoDataOutput()
output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
    Int(kCVPixelFormatType_32BGRA)]
output.alwaysDiscardsLateVideoFrames = true
let sink = Sink()
output.setSampleBufferDelegate(sink, queue: DispatchQueue(label: "so101.wrist.capture"))
guard session.canAddOutput(output) else { exit(5) }
session.addOutput(output)
session.commitConfiguration()
diagnostic("start_running_requested", device: device)
session.startRunning()
guard session.isRunning else { exit(6) }
diagnostic("capture_running", device: device)
RunLoop.current.run()
