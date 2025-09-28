import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation

from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from panda3d.core import DirectionalLight, AmbientLight, TransformState, LVector3f

# --- HELPER CLASS FOR MEDIAPIIPE AND RETARGETING ---
class PoseRetargeter:
    def __init__(self, actor, bone_map):
        self.actor = actor
        self.bone_map = bone_map
        self.t_pose_vectors = {}

        # Get the underlying character and animation bundle for advanced control
        self.character = self.actor.find('**/__Actor_modelRoot').node().get_character()
        self.bundle = self.character.get_bundle(0)

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
        
        print("Calculating T-Pose reference vectors from the model...")
        self._calculate_t_pose_vectors()

    def _calculate_t_pose_vectors(self):
        """Calculates and stores the initial vectors for each bone in the 3D model's T-pose."""
        for bone_name, (start_lm_name, end_lm_name) in self.bone_map.items():
            start_joint = self.character.find_joint(bone_name)
            
            # Find the child joint that corresponds to the end landmark
            end_joint = None
            for i in range(start_joint.get_num_children()):
                child = start_joint.get_child(i)
                # This logic is simple; a real system might need a more robust search
                if "forearm" in child.get_name().lower() or "shin" in child.get_name().lower() or "hand" in child.get_name().lower():
                    end_joint = child
                    break
            
            if start_joint and end_joint:
                start_transform = self.bundle.get_transform(start_joint)
                end_transform = self.bundle.get_transform(end_joint)
                
                start_pos = start_transform.get_pos()
                end_pos = end_transform.get_pos()

                vector = end_pos - start_pos
                vector.normalize()
                self.t_pose_vectors[bone_name] = vector
                print(f"  - Stored T-pose vector for '{bone_name}'")

    def get_rotation_quat(self, vec1, vec2):
        """Calculates the quaternion required to rotate from vec1 to vec2."""
        vec1 = np.array([vec1.x, vec1.y, vec1.z])
        vec2 = np.array([vec2.x, vec2.y, vec2.z])
        
        vec1 /= np.linalg.norm(vec1)
        vec2 /= np.linalg.norm(vec2)
        
        axis = np.cross(vec1, vec2)
        angle = np.arccos(np.dot(vec1, vec2))
        
        if np.linalg.norm(axis) < 1e-6:
            return None # Vectors are parallel, no rotation needed
            
        rot = Rotation.from_rotvec(angle * axis)
        quat = rot.as_quat() # Returns (x, y, z, w)
        return quat # Return as NumPy array

    def update_pose_from_frame(self, frame):
        """Processes a video frame and updates the 3D model's pose."""
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.pose_world_landmarks:
            return

        landmarks = results.pose_world_landmarks.landmark

        for bone_name, (start_lm_name, end_lm_name) in self.bone_map.items():
            start_lm = landmarks[self.mp_pose.PoseLandmark[start_lm_name]]
            end_lm = landmarks[self.mp_pose.PoseLandmark[end_lm_name]]

            # MediaPipe's Y-axis is up, Panda3D's Z-axis is up. We also invert X.
            live_vector = LVector3f(-(end_lm.x - start_lm.x), (end_lm.z - start_lm.z), (end_lm.y - start_lm.y))
            live_vector.normalize()
            
            if bone_name in self.t_pose_vectors:
                t_pose_vector = self.t_pose_vectors[bone_name]
                
                quat_np = self.get_rotation_quat(t_pose_vector, live_vector)
                
                if quat_np is not None:
                    # Create a Panda3D TransformState from the calculated rotation.
                    # SciPy quat is (x, y, z, w), Panda3D quat is (w, x, y, z)
                    transform = TransformState.make_quat( (quat_np[3], quat_np[0], quat_np[1], quat_np[2]) )
                    
                    # Apply the transform using the advanced, stable method
                    joint = self.character.find_joint(bone_name)
                    if joint:
                        self.bundle.control_joint(joint.get_name(), transform)

    def close(self):
        self.pose.close()

# --- MAIN PANDA3D APPLICATION ---
class MyApp(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()

        self.actor = Actor("zombie_clean.glb") # <<< YOUR CLEAN ZOMBIE FILE
        self.actor.reparentTo(self.render)
        self.actor.setPos(0, 0, -1)
        
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

        self.cap = cv2.VideoCapture("test_video.mp4") # <<< YOUR VIDEO FILE
        if not self.cap.isOpened():
            raise IOError("Cannot open video file")

        # The correct bone map for your working zombie model
        bone_landmark_map = {
            "shoulder.l": ("LEFT_SHOULDER", "LEFT_ELBOW"),
            "forearm.l": ("LEFT_ELBOW", "LEFT_WRIST"),
            "shoulder.r": ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
            "forearm.r": ("RIGHT_ELBOW", "RIGHT_WRIST"),
            "thigh.l": ("LEFT_HIP", "LEFT_KNEE"),
            "calf.l": ("LEFT_KNEE", "LEFT_ANKLE"),
            "thigh.r": ("RIGHT_HIP", "RIGHT_KNEE"),
            "calf.r": ("RIGHT_KNEE", "RIGHT_ANKLE"),
        }
        
        self.retargeter = PoseRetargeter(self.actor, bone_landmark_map)
        
        self.taskMgr.add(self.update_task, "update_task")
        self.accept('escape', self.cleanup_and_exit)

    def update_task(self, task):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return task.cont
        
        self.retargeter.update_pose_from_frame(frame)
        return task.cont

    def cleanup_and_exit(self):
        self.retargeter.close()
        self.cap.release()
        self.userExit()

app = MyApp()
app.run()