import Vision
import CoreImage
import UIKit

class SegmentationManager: ObservableObject {

    private var request: VNGeneratePersonSegmentationRequest?

    init() {
        setupRequest()
    }

    private func setupRequest() {
        request = VNGeneratePersonSegmentationRequest()
        request?.qualityLevel = .balanced
        request?.outputPixelFormat = kCVPixelFormatType_OneComponent8
    }

    func process(pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        guard let request = request else { return nil }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

        do {
            try handler.perform([request])
            if let mask = request.results?.first as? VNPixelBufferObservation {
                return mask.pixelBuffer
            }
        } catch {
            print("Segmentation failed: \(error)")
        }
        return nil
    }
}
