from raytracer import Raytracer
from auxillary_classes import Camera, Light, Sphere, Material
from PIL import Image
import numpy as np
import os

FILE_PATH = 'scene_1.obj'   # OBJ file

def build_scene(raytracer, light_intensity_multiplier):
    """Setup scene with given light intensity multiplier."""
    raytracer.setBackgroundColor((0, 0, 0))   # black background – no extra light

    # Camera
    camera = Camera([-3, 0.5, -2], 0.7, [-100, -10, -4])
    raytracer.setup_camera(camera, (1000, 1000))   # resolution

    # Load OBJ objects (floor, etc.) with checkerboard texture
    checker_mat = Material(
        diffuse_color=[0.7, 0.7, 0.7],
        specular_color=[1, 1, 1],
        ka=0.2, kd=0.5, ks=0.3, ke=32,
        texture_type='checkerboard',
        texture_scale=2.0,
        mapping_type='plane',
        color1=[1, 0, 0],   # red tile
        color2=[1, 1, 0],   # yellow tile
        kr=0.0
    )
    _, objects = getObjects(FILE_PATH)   # getObjects is in raytracer module
    for tri in objects:
        tri.material = checker_mat
    raytracer.addObject(objects)

    # Lights with variable intensity
    light_pos = [0, 5, 2]
    light_color = np.array([1.0, 1.0, 1.0]) * light_intensity_multiplier
    light1 = Light(light_pos, light_color)
    raytracer.lights.append(light1)
    raytracer.ambient_light = np.array([0.15, 0.15, 0.15]) * light_intensity_multiplier

    # Transparent sphere (glass)
    mat_glass = Material(
        diffuse_color=[0.0, 0.0, 0.0],
        specular_color=[1.0, 1.0, 1.0],
        ka=0.05, kd=0.0, ks=0.8, ke=128,
        kr=0.05, kt=0.9,
        ior=1.5, ior_outside=1.0,   # proper glass IOR
        texture_type=None
    )
    sphere_transparent = Sphere([-5.5, 1, -2], 1, mat_glass)

    # Reflective sphere
    mat_reflective = Material(
        diffuse_color=[0.6, 0.6, 0.6],
        specular_color=[1, 1, 1],
        ka=0.1, kd=0.3, ks=0.3, ke=64,
        kr=0.8,
        texture_type=None
    )
    sphere_reflective = Sphere([-7, 0.5, -1], 1, mat_reflective)

    raytracer.objects_spheres.append(sphere_transparent)
    raytracer.objects_spheres.append(sphere_reflective)

def save_image(image_data, filename):
    """Save an 8-bit RGB image from list of list of (R,G,B) tuples."""
    height = len(image_data)
    width = len(image_data[0]) if height > 0 else 0
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for i, row in enumerate(image_data):
        for j, pix in enumerate(row):
            arr[i, j] = pix
    img = Image.fromarray(arr, 'RGB')
    img.save(filename)
    print(f"Saved {filename}")

def main():
    # Output directory
    out_dir = "tone_mapped"
    os.makedirs(out_dir, exist_ok=True)

    # Light intensity levels
    intensities = {
        "low": 0.5,
        "mid": 2.0,
        "high": 10.0
    }

    Ldmax = 100.0   # cd/m^2, typical display

    reinhard_keys = {
        "low": 0.01,  # very dark scene → bright after mapping
        "mid": 0.18,  # normal Zone V
        "high": 1.0  # bright scene → more conservative mapping
    }

    for level_name, intensity in intensities.items():
        print(f"\n--- Lighting: {level_name} (intensity = {intensity}) ---")

        # ----- Ward -----
        rt = Raytracer()
        build_scene(rt, intensity)
        rt.render()
        ward_out = rt.ward_tone_mapping(rt.radiance_image, Ldmax)
        save_image(ward_out, f"tone_mapped/ward_{level_name}.jpg")

        # ----- Reinhard -----
        rt2 = Raytracer()
        build_scene(rt2, intensity)
        rt2.render()
        key = reinhard_keys[level_name]
        reinhard_out = rt2.reinhard_tone_mapping(rt2.radiance_image, Ldmax, key_value=key)
        save_image(reinhard_out, f"tone_mapped/reinhard_{level_name}.jpg")

    print("\nAll images generated. Check the 'tone_mapped' folder.")

if __name__ == "__main__":
    from raytracer import getObjects
    main()