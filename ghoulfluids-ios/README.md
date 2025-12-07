# GhoulFluids iOS

This directory contains the source code for the iOS version of GhoulFluids.

## How to Build

1.  **Open Xcode** and create a new project.
    *   Choose **iOS** -> **App**.
    *   Interface: **SwiftUI**.
    *   Language: **Swift**.
    *   Product Name: `ghoulfluids-ios` (or similar).

2.  **Add Source Files**:
    *   Copy all the files from this directory (`ghoulfluids-ios/`) into your new Xcode project folder.
    *   In Xcode, right-click the project navigator and choose "Add Files to..." to add them to the project.
    *   Delete the default `ContentView.swift` and `<AppName>App.swift` created by Xcode, as we provide replacements.

3.  **Info.plist Permissions**:
    *   You must add the **Camera Usage Description** to your `Info.plist`.
    *   Key: `Privacy - Camera Usage Description`
    *   Value: "Required for fluid interaction."

4.  **Build and Run**:
    *   Connect your iPhone.
    *   Select your device as the build target.
    *   Run the app.

## Architecture

*   **FluidSimulation.swift**: The core logic using Metal. It handles the stable fluids solver (Advection, Divergence, Pressure, Gradient).
*   **Shaders.metal**: The Metal Shading Language port of the original GLSL shaders.
*   **CameraManager.swift**: Handles `AVCaptureSession` to get camera frames.
*   **SegmentationManager.swift**: Uses Apple's Vision framework (`VNGeneratePersonSegmentationRequest`) to generate a person mask from the camera feed.
*   **MetalView.swift**: A SwiftUI wrapper for `MTKView` that drives the rendering loop.
