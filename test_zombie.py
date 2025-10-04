import math
from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from panda3d.core import DirectionalLight, AmbientLight, TransformState

class MyApp(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()

        self.actor = Actor("zombie_final.glb") # Your simplified model
        self.actor.reparentTo(self.render)
        self.actor.setHpr(180, 90, 0)
        
        # --- Lighting and Camera ---
        ambientLight = AmbientLight("ambient light")
        ambientLight.setColor((0.4, 0.4, 0.4, 1))
        self.render.setLight(self.render.attachNewNode(ambientLight))
        directionalLight = DirectionalLight("directional light")
        directionalLight.setColor((0.8, 0.8, 0.8, 1))
        dir_light_node = self.render.attachNewNode(directionalLight)
        dir_light_node.setHpr(30, -60, 0)
        self.render.setLight(dir_light_node)
        
        self.cam.setPos(0, -8, 2)
        self.cam.lookAt(0, 0, 0)

        # --- Set up the Animation ---
        # This is the CORRECT list of bones that exist in your simplified rig
        self.bones = {
            "shoulder.l": self.actor.controlJoint(None, "modelRoot", "shoulder.l"),
            "shoulder.r": self.actor.controlJoint(None, "modelRoot", "shoulder.r"),
            "thigh.l": self.actor.controlJoint(None, "modelRoot", "thigh.l"),
            "thigh.r": self.actor.controlJoint(None, "modelRoot", "thigh.r"),
        }
        
        self.rest_transforms = {name: joint.getTransform() for name, joint in self.bones.items() if joint}
        
        self.taskMgr.add(self.animate_bones_task, "AnimateBonesTask")

    def animate_bones_task(self, task):
        speed = 1.5
        amplitude = 35.0

        # --- Arm Swing (from shoulders) ---
        arm_angle = math.sin(task.time * speed) * amplitude
        arm_rot = TransformState.makeHpr((0, arm_angle, 0))
        self.bones["shoulder.l"].setTransform(self.rest_transforms["shoulder.l"].compose(arm_rot))
        self.bones["shoulder.r"].setTransform(self.rest_transforms["shoulder.r"].compose(arm_rot))

        # --- Leg Swing (from hips) ---
        leg_angle = math.sin(task.time * speed) * (amplitude * 0.5)
        leg_rot = TransformState.makeHpr((0, -leg_angle, 0)) # Opposes arms
        self.bones["thigh.l"].setTransform(self.rest_transforms["thigh.l"].compose(leg_rot))
        self.bones["thigh.r"].setTransform(self.rest_transforms["thigh.r"].compose(leg_rot))
        
        return task.cont

app = MyApp()
app.run()