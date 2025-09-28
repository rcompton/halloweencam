from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from panda3d.core import DirectionalLight, AmbientLight

class MyApp(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()

        # --- 1. Load the Model at the Origin ---
        self.actor = Actor("zombie_gamerig.glb")
        self.actor.reparentTo(self.render)
        
        # --- 2. Stand the Model Up ---
        self.actor.setHpr(0, 90, -2)
        
        # --- 3. Add Balanced Lighting ---
        
        # This is the "fill" light. It ensures nothing is pure black
        # and provides a base color for the textures.
        ambientLight = AmbientLight("ambient light")
        ambientLight.setColor((0.5, 0.5, 0.5, 1))
        self.render.setLight(self.render.attachNewNode(ambientLight))
        
        # This is the main "key" light that creates highlights and shadows.
        directionalLight = DirectionalLight("directional light")
        directionalLight.setColor((0.1, 0.1, 0.1, 1))
        dir_light_node = self.render.attachNewNode(directionalLight)
        dir_light_node.setHpr(30, -60, 0)
        self.render.setLight(dir_light_node)
        
        # --- 4. Position the Camera ---
        self.cam.setPos(0, -7, 5)
        self.cam.lookAt(0, 0, 0)

app = MyApp()
app.run()