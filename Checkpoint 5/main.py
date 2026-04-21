from raytracer import *
from auxillary_classes import *
from PIL import Image
import numpy as np

FILE_PATH = 'scene_1.obj'  # File with triangles

def main():
    myRaytracer = Raytracer()
    myRaytracer.setBackgroundColor((135, 206, 235))

    # Camera
    camera = Camera([-3, 0.5, -2], 0.7, [-100, -10, -4])
    myRaytracer.setup_camera(camera, (1000, 1000))

    # Load OBJ objects (floor, etc.)
    checker_mat = Material(
        diffuse_color=[0.7, 0.7, 0.7],
        specular_color=[1, 1, 1],
        ka=0.2, kd=0.5, ks=0.3, ke=32,
        texture_type='checkerboard',
        texture_scale=2.0,
        mapping_type='plane',
        color1=[1, 0, 0],   # red tile
        color2=[1, 1, 0],    # yellow tile
        kr = 0.0
    )
    vertices, objects = getObjects(FILE_PATH)
    for tri in objects:
        tri.material = checker_mat
    myRaytracer.addObject(objects)

    # Lights
    light1 = Light([0, 5, 2], [1, 1, 1])
    myRaytracer.lights.append(light1)
    myRaytracer.ambient_light = np.array([0.15, 0.15, 0.15])

    # 1. Stripes sphere (non‑reflective)
    mat_stripes = Material(
        diffuse_color=[1, 1, 1],
        specular_color=[1, 1, 1],
        ka=0.1, kd=0.7, ks=0.3, ke=64,
        texture_type='image',
        texture_image_path='earth.jpg',  # image path
        mapping_type='sphere',
        kr=0.0
    )
    sphere_img = Sphere([-5.5, 1.25, -2], 1, mat_stripes)

    # 2. Reflective sphere (clear, no texture)
    mat_reflective = Material(
        diffuse_color=[0.6, 0.6, 0.6],  # neutral gray
        specular_color=[1, 1, 1],
        ka=0.1, kd=0.3, ks=0.3, ke=64,
        kr=0.8,  # strong reflection
        texture_type=None  # no texture
    )

    sphere_reflective = Sphere([-7, 0.5, -1], 1, mat_reflective)

    myRaytracer.objects_spheres.append(sphere_img)
    myRaytracer.objects_spheres.append(sphere_reflective)

    # Render
    myRaytracer.render()

    # Save image
    img_array = np.array(myRaytracer.focal_plane_color, dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    img.save('Reflections.jpg')
    img.show()

if __name__ == "__main__":
    main()