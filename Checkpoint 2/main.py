from raytracer import *
from auxillary_classes import *
from PIL import Image
import numpy as np

FILE_PATH = 'scene_1_5.obj'

def main():
    myRaytracer = Raytracer()
    myRaytracer.setBackgroundColor((135, 206, 235))

    # camera = Camera([0.025, -2.81, 0.571], 0.05, [20.9753, -2.81, 0.571])
    camera = Camera([-4, 0.3, -0.6], 0.7, [-20, 0.3, -0.6])
    myRaytracer.setup_camera(camera, (800, 800))

    vertices, objects = getObjects(FILE_PATH)
    sphere_transparent = Sphere([-8.5, 0, -0.5], 1, (0, 0, 255))
    sphere_reflective = Sphere([-10.4, 0.874, 0.4], 1, (0, 255, 0))
    myRaytracer.objects_spheres.append(sphere_transparent)
    myRaytracer.objects_spheres.append(sphere_reflective)
    myRaytracer.addObject(objects)

    myRaytracer.render()

    img_array = np.array(myRaytracer.focal_plane_color, dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    img.show()
    img.save('Scene_1_Render.jpg')

if __name__ == "__main__":
    main()