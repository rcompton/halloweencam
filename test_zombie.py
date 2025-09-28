import math
from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from panda3d.core import DirectionalLight, AmbientLight

class MyApp(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()

        # --- Load your final, simplified zombie model ---
        self.actor = Actor("zombie2p.glb") # <<< USE YOUR NEW FILENAME
        self.actor.reparentTo(self.render)
        self.actor.setPos(0, 0, -1)
        
        # --- Lighting and Camera ---
        ambientLight = AmbientLight("ambient light")
        ambientLight.setColor((0.5, 0.5, 0.5, 1))
        self.render.setLight(self.render.attachNewNode(ambientLight))
        dirLight = DirectionalLight("dir light")
        dirLight.setColor((0.7, 0.7, 0.7, 1))
        dir_light_node = self.render.attachNewNode(dirLight)
        dir_light_node.setHpr(0, -60, 0)
        self.render.setLight(dir_light_node)
        
        self.cam.setPos(0, -5, 0.5)
        self.cam.lookAt(self.actor)

        # --- Set up the Animation with the simplified rig's bone names ---
        self.bones_to_animate = [
            "shoulder.l",
            "shoulder.r",
            "thigh.l",
            "thigh.r",
        ]

        self.taskMgr.add(self.animate_bones_task, "AnimateBonesTask")

    def animate_bones_task(self, task):
        """This function will now work because the rig is simple."""
        amplitude = 45.0
        speed = 0.75

        for bone_name in self.bones_to_animate:
            joint = self.actor.controlJoint(None, "modelRoot", bone_name)
            
            if joint:
                angle = math.sin(task.time * speed) * amplitude
                joint.setHpr(angle, 0, 0)

        return task.cont

app = MyApp()
app.run()