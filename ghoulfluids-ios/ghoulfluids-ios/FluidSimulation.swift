import Metal
import MetalKit
import simd
import CoreVideo

class FluidSimulation {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var textureCache: CVMetalTextureCache?

    // Pipelines
    var pipelineAdvect: MTLRenderPipelineState!
    var pipelineAdvectDye: MTLRenderPipelineState!
    var pipelineSplat: MTLRenderPipelineState!
    var pipelineForce: MTLRenderPipelineState!
    var pipelineDivergence: MTLRenderPipelineState!
    var pipelineJacobi: MTLRenderPipelineState!
    var pipelineGradient: MTLRenderPipelineState!
    var pipelineCurl: MTLRenderPipelineState!
    var pipelineVorticity: MTLRenderPipelineState!
    var pipelineMaskForce: MTLRenderPipelineState!
    var pipelineMaskDye: MTLRenderPipelineState!
    var pipelineShow: MTLRenderPipelineState!
    var pipelineShowCam: MTLRenderPipelineState!
    var pipelineCopy: MTLRenderPipelineState!

    // Textures
    var velA: MTLTexture!
    var velB: MTLTexture!
    var dyeA: MTLTexture!
    var dyeB: MTLTexture!
    var prsA: MTLTexture!
    var prsB: MTLTexture!
    var div: MTLTexture!
    var curl: MTLTexture!

    var maskCurr: MTLTexture!
    var maskPrev: MTLTexture!

    // Simulation parameters
    var simW: Int = 192
    var simH: Int = 256

    // Quad buffer
    var vertexBuffer: MTLBuffer!

    // Constants
    var velDissipation: Float = 0.99
    var dyeDissipation: Float = 0.98
    var vorticityEps: Float = 2.0

    var paletteOn: Int32 = 1
    var paletteId: Int32 = 0
    var paletteId2: Int32 = 1
    var paletteMix: Float = 0.0

    init?(device: MTLDevice) {
        self.device = device
        guard let queue = device.makeCommandQueue() else { return nil }
        self.commandQueue = queue

        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)

        setupPipelines()
        setupTextures()
        setupBuffers()

        // Initial splat
        splat()
    }

    func setupPipelines() {
        guard let library = device.makeDefaultLibrary() else {
            print("Could not load default library")
            return
        }

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "vertex_main")
        descriptor.depthAttachmentPixelFormat = .invalid

        func makePipeline(fragment: String, format: MTLPixelFormat) -> MTLRenderPipelineState {
            descriptor.fragmentFunction = library.makeFunction(name: fragment)
            descriptor.colorAttachments[0].pixelFormat = format
            return try! device.makeRenderPipelineState(descriptor: descriptor)
        }

        pipelineAdvect = makePipeline(fragment: "advect_main", format: .rg16Float)
        pipelineAdvectDye = makePipeline(fragment: "advect_main", format: .rgba16Float)
        pipelineSplat = makePipeline(fragment: "splat_main", format: .rgba16Float)
        pipelineForce = makePipeline(fragment: "force_main", format: .rg16Float)
        pipelineDivergence = makePipeline(fragment: "divergence_main", format: .r16Float)
        pipelineJacobi = makePipeline(fragment: "jacobi_main", format: .r16Float)
        pipelineGradient = makePipeline(fragment: "gradient_main", format: .rg16Float)
        pipelineCurl = makePipeline(fragment: "curl_main", format: .r16Float)
        pipelineVorticity = makePipeline(fragment: "vorticity_main", format: .rg16Float)
        pipelineMaskForce = makePipeline(fragment: "mask_force_main", format: .rg16Float)
        pipelineMaskDye = makePipeline(fragment: "mask_dye_main", format: .rgba16Float)

        // Enable blending for show pipeline
        descriptor.fragmentFunction = library.makeFunction(name: "show_main")
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        descriptor.colorAttachments[0].isBlendingEnabled = true
        descriptor.colorAttachments[0].rgbBlendOperation = .add
        descriptor.colorAttachments[0].alphaBlendOperation = .add
        descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        descriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        descriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        pipelineShow = try! device.makeRenderPipelineState(descriptor: descriptor)

        // Reset blending for cam pipeline (opaque)
        descriptor.colorAttachments[0].isBlendingEnabled = false
        pipelineShowCam = makePipeline(fragment: "show_cam_main", format: .bgra8Unorm)

        // Mask update pipeline (R8 -> R16Float)
        pipelineCopy = makePipeline(fragment: "copy_main", format: .r16Float)
    }

    func setupTextures() {
        let velDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg16Float, width: simW, height: simH, mipmapped: false)
        velDesc.usage = [.shaderRead, .renderTarget]
        velA = device.makeTexture(descriptor: velDesc)
        velB = device.makeTexture(descriptor: velDesc)

        let dyeDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: simW, height: simH, mipmapped: false)
        dyeDesc.usage = [.shaderRead, .renderTarget]
        dyeA = device.makeTexture(descriptor: dyeDesc)
        dyeB = device.makeTexture(descriptor: dyeDesc)

        let scalarDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: simW, height: simH, mipmapped: false)
        scalarDesc.usage = [.shaderRead, .renderTarget]
        prsA = device.makeTexture(descriptor: scalarDesc)
        prsB = device.makeTexture(descriptor: scalarDesc)
        div = device.makeTexture(descriptor: scalarDesc)
        curl = device.makeTexture(descriptor: scalarDesc)

        // Masks are single channel
        maskCurr = device.makeTexture(descriptor: scalarDesc)
        maskPrev = device.makeTexture(descriptor: scalarDesc)

        // Clear
        clearTexture(dyeA)
        clearTexture(dyeB)
        clearTexture(velA)
        clearTexture(velB)
        clearTexture(maskCurr)
        clearTexture(maskPrev)
    }

    func clearTexture(_ tex: MTLTexture) {
        let region = MTLRegionMake2D(0, 0, tex.width, tex.height)
        // Simple zero fill would ideally use a blit or render pass, skipping for brevity
    }

    func setupBuffers() {
        let vertices: [Float] = [
            -1, -1,
             1, -1,
            -1,  1,
             1,  1
        ]
        vertexBuffer = device.makeBuffer(bytes: vertices, length: vertices.count * MemoryLayout<Float>.size, options: [])
    }

    func splat() {
        guard let buffer = commandQueue.makeCommandBuffer() else { return }

        let desc = MTLRenderPassDescriptor()
        desc.colorAttachments[0].texture = dyeA
        desc.colorAttachments[0].loadAction = .clear
        desc.colorAttachments[0].storeAction = .store
        desc.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        guard let enc = buffer.makeRenderCommandEncoder(descriptor: desc) else { return }

        enc.setRenderPipelineState(pipelineSplat)
        enc.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        enc.setFragmentTexture(dyeB, index: 0) // unused

        var point: SIMD2<Float> = [0.5, 0.5]
        var value: SIMD3<Float> = [0.9, 0.4, 0.9]
        var radius: Float = 0.15

        enc.setFragmentBytes(&point, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
        enc.setFragmentBytes(&value, length: MemoryLayout<SIMD3<Float>>.size, index: 1)
        enc.setFragmentBytes(&radius, length: MemoryLayout<Float>.size, index: 2)

        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        enc.endEncoding()
        buffer.commit()
    }

    func swap<T>(_ a: inout T, _ b: inout T) {
        let temp = a
        a = b
        b = temp
    }

    func update(dt: Float, maskPixelBuffer: CVPixelBuffer?) {
        guard let buffer = commandQueue.makeCommandBuffer() else { return }

        // 1. Update Mask
        if let mpb = maskPixelBuffer {
            updateMask(mpb, buffer: buffer)
        }

        // 2. Advect Velocity
        renderPass(buffer: buffer, pipeline: pipelineAdvect, dest: velB) { enc in
            enc.setFragmentTexture(velA, index: 0)
            enc.setFragmentTexture(velA, index: 1)
            var _dt = dt
            var _diss = velDissipation
            enc.setFragmentBytes(&_dt, length: MemoryLayout<Float>.size, index: 0)
            enc.setFragmentBytes(&_diss, length: MemoryLayout<Float>.size, index: 1)
        }
        swap(&velA, &velB)

        // 3. Mask Force
        if maskPixelBuffer != nil {
            renderPass(buffer: buffer, pipeline: pipelineMaskForce, dest: velB) { enc in
                enc.setFragmentTexture(velA, index: 0)
                enc.setFragmentTexture(maskCurr, index: 1)
                enc.setFragmentTexture(maskPrev, index: 2)
                var texel: SIMD2<Float> = [1.0/Float(simW), 1.0/Float(simH)]
                var _dt = dt
                var edgeThresh: Float = 0.1
                var ampNormal: Float = 50.0
                var ampTangent: Float = 5.0
                var useTemporal: Int32 = 1

                enc.setFragmentBytes(&texel, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
                enc.setFragmentBytes(&_dt, length: MemoryLayout<Float>.size, index: 1)
                enc.setFragmentBytes(&edgeThresh, length: MemoryLayout<Float>.size, index: 2)
                enc.setFragmentBytes(&ampNormal, length: MemoryLayout<Float>.size, index: 3)
                enc.setFragmentBytes(&ampTangent, length: MemoryLayout<Float>.size, index: 4)
                enc.setFragmentBytes(&useTemporal, length: MemoryLayout<Int32>.size, index: 5)
            }
            swap(&velA, &velB)
        }

        // 4. Divergence
        renderPass(buffer: buffer, pipeline: pipelineDivergence, dest: div) { enc in
            enc.setFragmentTexture(velA, index: 0)
            var texel: SIMD2<Float> = [1.0/Float(simW), 1.0/Float(simH)]
            enc.setFragmentBytes(&texel, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
        }

        // 5. Jacobi
        for _ in 0..<20 {
            renderPass(buffer: buffer, pipeline: pipelineJacobi, dest: prsB) { enc in
                enc.setFragmentTexture(prsA, index: 0)
                enc.setFragmentTexture(div, index: 1)
                var texel: SIMD2<Float> = [1.0/Float(simW), 1.0/Float(simH)]
                enc.setFragmentBytes(&texel, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
            }
            swap(&prsA, &prsB)
        }

        // 6. Gradient
        renderPass(buffer: buffer, pipeline: pipelineGradient, dest: velB) { enc in
            enc.setFragmentTexture(velA, index: 0)
            enc.setFragmentTexture(prsA, index: 1)
            var texel: SIMD2<Float> = [1.0/Float(simW), 1.0/Float(simH)]
            enc.setFragmentBytes(&texel, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
        }
        swap(&velA, &velB)

        // 7. Advect Dye
        renderPass(buffer: buffer, pipeline: pipelineAdvectDye, dest: dyeB) { enc in
            enc.setFragmentTexture(velA, index: 0) // vel
            enc.setFragmentTexture(dyeA, index: 1) // src
            var _dt = dt
            var _diss = dyeDissipation
            enc.setFragmentBytes(&_dt, length: MemoryLayout<Float>.size, index: 0)
            enc.setFragmentBytes(&_diss, length: MemoryLayout<Float>.size, index: 1)
        }
        swap(&dyeA, &dyeB)

        // 8. Mask Dye
        if maskPixelBuffer != nil {
            renderPass(buffer: buffer, pipeline: pipelineMaskDye, dest: dyeB) { enc in
                enc.setFragmentTexture(dyeA, index: 0)
                enc.setFragmentTexture(maskCurr, index: 1)
                var texel: SIMD2<Float> = [1.0/Float(simW), 1.0/Float(simH)]
                var edgeThresh: Float = 0.1
                var color: SIMD3<Float> = [1.0, 0.5, 0.0]
                var strength: Float = 5.0

                enc.setFragmentBytes(&texel, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
                enc.setFragmentBytes(&edgeThresh, length: MemoryLayout<Float>.size, index: 1)
                enc.setFragmentBytes(&color, length: MemoryLayout<SIMD3<Float>>.size, index: 2)
                enc.setFragmentBytes(&strength, length: MemoryLayout<Float>.size, index: 3)
            }
            swap(&dyeA, &dyeB)
        }

        buffer.commit()
    }

    func updateMask(_ pb: CVPixelBuffer, buffer: MTLCommandBuffer) {
        guard let cache = textureCache else { return }

        let width = CVPixelBufferGetWidth(pb)
        let height = CVPixelBufferGetHeight(pb)

        var cvTex: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, pb, nil, .r8Unorm, width, height, 0, &cvTex)

        guard let cvTex = cvTex, let mtlTex = CVMetalTextureGetTexture(cvTex) else { return }

        swap(&maskPrev, &maskCurr)

        // Copy mask from R8 (Vision) to R16Float (Simulation)
        renderPass(buffer: buffer, pipeline: pipelineCopy, dest: maskCurr) { enc in
            enc.setFragmentTexture(mtlTex, index: 0)
        }
    }

    func renderPass(buffer: MTLCommandBuffer, pipeline: MTLRenderPipelineState, dest: MTLTexture, configure: (MTLRenderCommandEncoder) -> Void) {
        let desc = MTLRenderPassDescriptor()
        desc.colorAttachments[0].texture = dest
        desc.colorAttachments[0].loadAction = .dontCare
        desc.colorAttachments[0].storeAction = .store

        guard let enc = buffer.makeRenderCommandEncoder(descriptor: desc) else { return }
        enc.setRenderPipelineState(pipeline)
        enc.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        configure(enc)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        enc.endEncoding()
    }
}
