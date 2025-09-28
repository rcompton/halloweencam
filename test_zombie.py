import math
from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from panda3d.core import DirectionalLight, AmbientLight, TransformState

class MyApp(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()

        self.actor = Actor("zombie_gamerig.glb")
        self.actor.reparentTo(self.render)
        self.actor.setHpr(0, 90, 0)
        
        # --- Lighting and Camera ---
        ambientLight = AmbientLight("ambient light")
        ambientLight.setColor((0.5, 0.5, 0.5, 1))
        self.render.setLight(self.render.attachNewNode(ambientLight))
        directionalLight = DirectionalLight("directional light")
        directionalLight.setColor((0.1, 0.1, 0.1, 1))
        dir_light_node = self.render.attachNewNode(directionalLight)
        dir_light_node.setHpr(30, -60, 0)
        self.render.setLight(dir_light_node)
        
        self.cam.setPos(0, -8, 2)
        self.cam.lookAt(0, 0, 0)

        # --- Set up the Animation ---
        self.bones = {
            "spine_01.x": self.actor.controlJoint(None, "modelRoot", "spine_01.x"),
            "shoulder.l": self.actor.controlJoint(None, "modelRoot", "shoulder.l"),
            "shoulder.r": self.actor.controlJoint(None, "modelRoot", "shoulder.r"),
            "head.x": self.actor.controlJoint(None, "modelRoot", "head.x"),
            "foot.l": self.actor.controlJoint(None, "modelRoot", "foot.l"),
        }
        
        # Store the initial T-pose transform for each bone
        self.rest_transforms = {name: joint.getTransform() for name, joint in self.bones.items()}
        
        self.taskMgr.add(self.animate_bones_task, "AnimateBonesTask")

    def animate_bones_task(self, task):
        speed = 1.5
        amplitude = 52.0

        # --- Spine Rotation ---
        spine_joint = self.bones["spine_01.x"]
        rest_spine = self.rest_transforms["spine_01.x"]
        spine_angle = math.sin(task.time * speed) * amplitude
        spine_rot = TransformState.makeHpr((0, 0, spine_angle))
        final_spine_transform = rest_spine.compose(spine_rot)
        spine_joint.setTransform(final_spine_transform)

        # --- Head Rotation (Relative to Spine) ---
        head_joint = self.bones["head.x"]
        rest_head = self.rest_transforms["head.x"]
        head_angle = math.sin(task.time * speed * 1.2) * (amplitude * 1.5)
        head_rot = TransformState.makeHpr((0, 0, -head_angle))
        # We don't need to compose with the spine here because the node hierarchy
        # handles it automatically when we set the transform.
        final_head_transform = rest_head.compose(head_rot)
        head_joint.setTransform(final_head_transform)

        # --- Shoulder Rotation ---
        shoulder_l = self.bones["shoulder.l"]
        rest_l = self.rest_transforms["shoulder.l"]
        shoulder_r = self.bones["shoulder.r"]
        rest_r = self.rest_transforms["shoulder.r"]
        arm_angle = math.sin(task.time * speed * 0.8) * (amplitude * 0.5)
        arm_rot = TransformState.makeHpr((0, arm_angle, 0))
        shoulder_l.setTransform(rest_l.compose(arm_rot))
        shoulder_r.setTransform(rest_r.compose(arm_rot))
        
        return task.cont

app = MyApp()
app.run()