import SwiftUI

struct ContentView: View {
    @StateObject var camera = CameraManager()

    var body: some View {
        ZStack {
            MetalView(camera: camera)
                .edgesIgnoringSafeArea(.all)
                .onAppear {
                    camera.start()
                }
                .onDisappear {
                    camera.stop()
                }

            VStack {
                Spacer()
                Text("GhoulFluids iOS")
                    .foregroundColor(.white)
                    .padding()
                    .background(Color.black.opacity(0.5))
                    .cornerRadius(8)
                    .padding(.bottom, 50)
            }
        }
    }
}
