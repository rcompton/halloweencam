from ghoulfluids import shaders as S

def test_shader_strings_exist():
    for name in ["VS","FS_ADVECT","FS_DIVERGENCE","FS_JACOBI","FS_GRADIENT",
                 "FS_CURL","FS_VORTICITY","FS_MASK_FORCE","FS_MASK_DYE",
                 "FS_SHOW","FS_SHOW_CAM"]:
        assert hasattr(S, name)
        assert isinstance(getattr(S, name), str)
