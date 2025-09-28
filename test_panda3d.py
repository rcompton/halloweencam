import random
from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from panda3d.core import LVector3


class MyApp(ShowBase):
    def __init__(self):
        # Initialize the ShowBase class
        super().__init__()
        self.cam.setPos(0, -30, 6)  # Set camera position

        # --- Load the Rigged Model (without animations) ---
        # self.actor = Actor("alady.glb") # Replace with your model file
        self.actor = Actor("models/panda-model", {"walk": "models/panda-walk"})
        self.actor.reparentTo(self.render)
        # self.actor.setScale(10) # Scale down the model
        self.actor.setPos(0, 0, 0)

        # --- IMPORTANT: Find and Print Bone Names ---
        # You need to know the names of the bones to control them.
        # This will print a list of all joints in your model's skeleton.
        print("Available joints:")
        for joint in self.actor.getJoints():
            print(joint.getName())

        # List of bone names you want to animate.
        # REPLACE these with actual names from the printed list above!
        self.bones_to_animate = [
            "Bone_lr_leg_hip",
            "Bone_lr_leg_upper",
            "Bone_lr_leg_lower",
            "Bone_lr_foot",
            "Bone_lr_foot_nub",
            "Bone_rr_leg_hip",
            "Bone_rr_leg_upper",
            "Bone_rr_leg_lower",
            "Bone_rr_foot",
            "Bone_rr_foot_nub",
            "Bone_spine01",
            "Bone_spine02",
            "Bone_spine03",
            "Bone_spine_nub",
            "Dummy_lf_foot_heel",
            "Dummy_lf_foot_toe",
            "Dummy_lr_foot_heel",
            "Dummy_lr_foot_toe",
            "Dummy_rf_foot_heel",
            "Dummy_rf_foot_toe",
            "Dummy_rr_foot_heel",
            "Dummy_rr_foot_toe",
            "Dummy_shoulders",
            "Bone_lf_clavicle",
            "Bone_lf_leg_upper",
            "Bone_lf_leg_lower",
            "Bone_lf_foot",
            "Bone_lf_foot_nub",
            "Bone_neck",
            "Bone_jaw01",
            "Bone_jaw02",
            "Bone_jaw03",
            "Bone_jaw_nub",
            "Bone_skull",
            "Bone_skull_nub",
            "Dummy_jaw",
            "Bone_rf_clavicle",
            "Bone_rf_leg_upper",
            "Bone_rf_leg_lower",
            "Bone_rf_foot",
            "Bone_rf_foot_nub",
        ]

        # Add the animation task to the task manager
        self.taskMgr.add(self.animate_bones_task, "AnimateBonesTask")

    def animate_bones_task(self, task):
        """
        This function is called every frame to update bone rotations.
        """
        for bone_name in self.bones_to_animate:
            # Get a handle to control the specific joint
            # The first argument (None) gets the default LOD
            # The second argument ("modelRoot") is the part of the model
            # The third is the name of the joint you want to control
            joint = self.actor.controlJoint(None, "modelRoot", bone_name)

            if joint:
                # Generate random rotation values (Heading, Pitch, Roll)
                h = random.uniform(-0.45, 0.45)
                p = random.uniform(-0.45, 0.45)
                r = random.uniform(-0.45, 0.45)

                # Apply the random rotation to the joint
                joint.setHpr(h, p, r)

        # Tell the task manager to continue calling this task
        return task.cont


app = MyApp()
app.run()
