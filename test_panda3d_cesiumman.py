import math
import random
from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from panda3d.core import DirectionalLight, AmbientLight


class MyApp(ShowBase):
    def __init__(self):
        # Initialize the ShowBase class
        super().__init__()
        self.cam.setPos(0, -30, 6)  # Set camera position

        # --- Load the Rigged Model (without animations) ---
        self.actor = Actor("CesiumMan.glb")  # Replace with your model file
        # self.actor = Actor("models/panda-model", {"walk": "models/panda-walk"})
        self.actor.reparentTo(self.render)
        self.actor.setScale(1.9)
        self.actor.setPos(0, 0, 0)

        # --- IMPORTANT: Find and Print Bone Names ---
        # You need to know the names of the bones to control them.
        # This will print a list of all joints in your model's skeleton.
        print("Available joints:")
        for joint in self.actor.getJoints():
            print(joint.getName())

        # --- 2. Add Lighting ---
        # Without lights, 3D models often appear black or are invisible.

        # Add a simple ambient light to give everything a base color
        ambientLight = AmbientLight("ambient light")
        ambientLight.setColor((0.4, 0.4, 0.4, 1))
        self.ambientLightNodePath = self.render.attachNewNode(ambientLight)
        self.render.setLight(self.ambientLightNodePath)

        # Add a directional light to create highlights and shadows
        directionalLight = DirectionalLight("directional light")
        directionalLight.setColor((0.6, 0.6, 0.6, 1))
        self.directionalLightNodePath = self.render.attachNewNode(directionalLight)
        self.directionalLightNodePath.setHpr(
            0, -60, 0
        )  # Point the light down and forward
        self.render.setLight(self.directionalLightNodePath)

        # --- 3. Position the Camera ---
        self.cam.setPos(0, -10, 1.5)
        self.cam.lookAt(self.actor)

        # List of bone names you want to animate.
        # REPLACE these with actual names from the printed list above!
        self.bones_to_animate = [
            #"Skeleton_torso_joint_1",
            #"Skeleton_torso_joint_2",
            #"torso_joint_3",
            #"Skeleton_neck_joint_1",
            #"Skeleton_neck_joint_2",
            "Skeleton_arm_joint_L__4_",
            #"Skeleton_arm_joint_L__3_",
            "Skeleton_arm_joint_L__2_",
            "Skeleton_arm_joint_R",
            #"Skeleton_arm_joint_R__2_",
            "Skeleton_arm_joint_R__3_",
            #"leg_joint_L_1",
            #"leg_joint_L_2",
            #"leg_joint_L_3",
            "leg_joint_L_5",
            #"leg_joint_R_1",
            #"leg_joint_R_2",
            #"leg_joint_R_3",
            "leg_joint_R_5",
        ]

        # Add the animation task to the task manager
        self.taskMgr.add(self.animate_bones_task, "AnimateBonesTask")

    def animate_bones_task(self, task):
        """
        This function is called every frame to update bone rotations smoothly.
        """
        # --- Define animation properties ---
        # How far the bones should rotate back and forth (in degrees)
        amplitude = 15.0
        # How fast the bones should oscillate
        speed = 1.75

        for bone_name in self.bones_to_animate:
            joint = self.actor.controlJoint(None, "modelRoot", bone_name)

            if joint:
                # Calculate new rotations using sine waves based on time
                # We use slightly different speeds for H, P, and R for more
                # complex-looking, less robotic motion.
                h = math.sin(task.time * speed * 1.2) * amplitude
                p = math.sin(task.time * speed * 1.5) * (amplitude * 0.5) # Pitch moves less
                r = math.sin(task.time * speed * 0.9) * amplitude

                # Apply the smooth rotation to the joint
                joint.setHpr(h, p, r)

        # Tell the task manager to continue calling this task
        return task.cont

app = MyApp()
app.run()
