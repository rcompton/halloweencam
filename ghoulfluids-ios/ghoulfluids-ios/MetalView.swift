import SwiftUI
import MetalKit
import CoreVideo

struct MetalView: UIViewRepresentable {
    @ObservedObject var camera: CameraManager
    let segmentation = SegmentationManager()
    var fluid: FluidSimulation?

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView()
        mtkView.device = MTLCreateSystemDefaultDevice()
        mtkView.delegate = context.coordinator
        mtkView.framebufferOnly = false
        mtkView.enableSetNeedsDisplay = false
        mtkView.preferredFramesPerSecond = 60
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        if let device = mtkView.device {
            context.coordinator.fluid = FluidSimulation(device: device)
            // Create texture cache for camera display
            CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &context.coordinator.textureCache)
        }

        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
    }

    class Coordinator: NSObject, MTKViewDelegate {
        var parent: MetalView
        var fluid: FluidSimulation?
        var textureCache: CVMetalTextureCache?
        var lastTime: CFTimeInterval = 0

        init(_ parent: MetalView) {
            self.parent = parent
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        }

        func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable,
                  let fluid = fluid else { return }

            let time = CACurrentMediaTime()
            let dt = lastTime == 0 ? 0.016 : Float(time - lastTime)
            lastTime = time

            var cameraTexture: MTLTexture?

            // 1. Get Camera Frame & Segmentation
            if let pixelBuffer = parent.camera.currentFrame {
                let mask = parent.segmentation.process(pixelBuffer: pixelBuffer)
                fluid.update(dt: dt, maskPixelBuffer: mask)

                // Create texture for camera background
                if let cache = textureCache {
                    let w = CVPixelBufferGetWidth(pixelBuffer)
                    let h = CVPixelBufferGetHeight(pixelBuffer)
                    var cvTex: CVMetalTexture?
                    CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, pixelBuffer, nil, .bgra8Unorm, w, h, 0, &cvTex)
                    if let cvTex = cvTex {
                        cameraTexture = CVMetalTextureGetTexture(cvTex)
                    }
                }
            } else {
                fluid.update(dt: dt, maskPixelBuffer: nil)
            }

            // 4. Render to screen
            guard let buffer = fluid.commandQueue.makeCommandBuffer() else { return }

            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = drawable.texture
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

            guard let enc = buffer.makeRenderCommandEncoder(descriptor: pass) else { return }

            // Draw Camera Background
            if let camTex = cameraTexture {
                enc.setRenderPipelineState(fluid.pipelineShowCam)
                enc.setVertexBuffer(fluid.vertexBuffer, offset: 0, index: 0)
                enc.setFragmentTexture(camTex, index: 0)
                enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }

            // Draw Fluid Dye (blended)
            // Note: pipelineShow is opaque in this MVP. Real blending needs separate pipeline setup.
            // But let's draw it anyway. If palette has black background, it might obscure camera.
            // In real app, shader should alpha blend.

            enc.setRenderPipelineState(fluid.pipelineShow)
            enc.setVertexBuffer(fluid.vertexBuffer, offset: 0, index: 0)

            enc.setFragmentTexture(fluid.dyeA, index: 0)

            var pOn = fluid.paletteOn
            var pId = fluid.paletteId
            var pId2 = fluid.paletteId2
            var pMix = fluid.paletteMix

            enc.setFragmentBytes(&pOn, length: 4, index: 0)
            enc.setFragmentBytes(&pId, length: 4, index: 1)
            enc.setFragmentBytes(&pId2, length: 4, index: 2)
            enc.setFragmentBytes(&pMix, length: 4, index: 3)

            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            enc.endEncoding()

            buffer.present(drawable)
            buffer.commit()
        }
    }
}
