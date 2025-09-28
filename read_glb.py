import sys
from pygltflib import GLTF2

def print_hierarchy(gltf, node_index, indent=""):
    """Recursively prints the node hierarchy."""
    node = gltf.nodes[node_index]
    node_name = node.name if node.name else f"Node_{node_index}"
    
    details = []
    if node.mesh is not None:
        details.append(f"Mesh: {node.mesh}")
    if node.skin is not None:
        details.append(f"Skin: {node.skin}")
    
    print(f"{indent}- {node_name} ({', '.join(details)})")
    
    if node.children:
        for child_index in node.children:
            print_hierarchy(gltf, child_index, indent + "  ")

def inspect_glb(file_path):
    """Loads a GLB file and prints a report of its contents."""
    try:
        gltf = GLTF2.load(file_path)
        print(f"--- Inspection Report for: {file_path} ---\n")

        # --- Print Scene Info ---
        if gltf.scene is not None:
            print(f"## Default Scene: {gltf.scene}")
            root_nodes = gltf.scenes[gltf.scene].nodes
            print(f"## Node Hierarchy:")
            for node_index in root_nodes:
                print_hierarchy(gltf, node_index)
        else:
            print("## No default scene found.")
        
        print("\n----------------------------------------\n")

        # --- Print Skin (Armature) Info ---
        if gltf.skins:
            print(f"## Skeletons (Skins): Found {len(gltf.skins)}")
            for i, skin in enumerate(gltf.skins):
                skin_name = skin.name if skin.name else f"Skin_{i}"
                print(f"\n- Skeleton '{skin_name}':")
                if skin.skeleton is not None:
                    root_bone_name = gltf.nodes[skin.skeleton].name
                    print(f"  - Root Bone: '{root_bone_name}' (Node {skin.skeleton})")
                
                print(f"  - Joints ({len(skin.joints)} total):")
                for joint_index in skin.joints:
                    joint_node = gltf.nodes[joint_index]
                    joint_name = joint_node.name if joint_node.name else f"Node_{joint_index}"
                    print(f"    - '{joint_name}'")
        else:
            print("## Skeletons (Skins): None found.")

        print("\n----------------------------------------\n")
        
        # --- Print Animation Info ---
        if gltf.animations:
            print(f"## Animations: Found {len(gltf.animations)}")
            for i, anim in enumerate(gltf.animations):
                anim_name = anim.name if anim.name else f"Animation_{i}"
                print(f"- '{anim_name}'")
        else:
            print("## Animations: None found.")
            
        print("\n--- End of Report ---")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_model.py <path_to_your_file.glb>")
    else:
        inspect_glb(sys.argv[1])