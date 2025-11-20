import Foundation
import AVFoundation
import UIKit

class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()

    @Published var currentFrame: CVPixelBuffer?

    override init() {
        super.init()
        setupCamera()
    }

    private func setupCamera() {
        captureSession.beginConfiguration()

        // Use medium preset for performance balance
        captureSession.sessionPreset = .vga640x480

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
            print("No front camera found")
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: camera)
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
            }

            if captureSession.canAddOutput(videoOutput) {
                captureSession.addOutput(videoOutput)
                videoOutput.alwaysDiscardsLateVideoFrames = true
                videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
                videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "cameraQueue"))
            }

            captureSession.commitConfiguration()
        } catch {
            print("Error setting up camera: \(error)")
        }
    }

    func start() {
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }

    func stop() {
        captureSession.stopRunning()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Mirror the front camera
        if connection.isVideoMirroringSupported {
           connection.isVideoMirrored = true
        }
        // Correct orientation if needed (usually front cam is landscape left/right depending on device orientation)
        // For this simple MVP we might assume a fixed orientation or handle it in the shader/view.

        DispatchQueue.main.async {
            self.currentFrame = pixelBuffer
        }
    }
}
