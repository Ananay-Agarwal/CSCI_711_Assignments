from raytracer import *
from auxillary_classes import *
from PIL import Image
import numpy as np

FILE_PATH = 'scene_1.obj'

def main():
    myRaytracer = Raytracer()
    myRaytracer.setBackgroundColor((135, 206, 235))  # sky blue

    # Camera
    camera = Camera([-3, 0.5, -2], 0.7, [-100, -10, -4])
    myRaytracer.setup_camera(camera, (1000, 1000))

    # Load OBJ triangles
    vertices, objects = getObjects(FILE_PATH)
    myRaytracer.addObject(objects)

    # Add lights
    light1 = Light([0, 5, 2], [1, 1, 1])        # white light from above
    # light2 = Light([-2, 1, 1], [0.5, 0.5, 0.5]) # dim fill light
    myRaytracer.lights.append(light1)
    # myRaytracer.lights.append(light2)
    myRaytracer.ambient_light = np.array([0.15, 0.15, 0.15])

    # Spheres with materials
    mat_blue = Material([0, 0, 1], [1, 1, 1], 0.1, 0.6, 0.4, 64)
    mat_green = Material([0, 1, 0], [1, 1, 1], 0.1, 0.7, 0.3, 32)
    sphere1 = Sphere([-5.5, 1.25, -2], 1, mat_blue)
    sphere2 = Sphere([-7, 0.5, -1], 1, mat_green)
    myRaytracer.objects_spheres.append(sphere1)
    myRaytracer.objects_spheres.append(sphere2)

    # Render
    myRaytracer.render()

    # Save image
    img_array = np.array(myRaytracer.focal_plane_color, dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    img.save('Scene_1_Render_Phong.jpg')
    img.show()

if __name__ == "__main__":
    main()