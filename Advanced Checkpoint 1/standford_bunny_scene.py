from raytracer import *
from auxillary_classes import *
from PIL import Image
import numpy as np

FILE_PATH = 'stanford_bunny.obj'

def main():
    myRaytracer = Raytracer()
    # myRaytracer.set_use_kdtree(False)
    myRaytracer.setBackgroundColor((135, 206, 235))  # sky blue

    # Camera
    camera = Camera([-1.5, 1.25, 0.25], 0.7, [100, 0, 4])
    myRaytracer.setup_camera(camera, (100, 100))

    # Load OBJ triangles
    vertices, objects = getObjects(FILE_PATH)
    myRaytracer.addObject(objects)

    # Add lights
    light1 = Light([-2, 3, 0], [1, 1, 1])        # white light from above
    light2 = Light([-2, 1, 1], [0.5, 0.5, 0.5])  # dim fill light
    myRaytracer.lights.append(light1)
    myRaytracer.lights.append(light2)
    myRaytracer.ambient_light = np.array([0.15, 0.15, 0.15])

    # Render
    myRaytracer.render()

    # Save image
    img_array = np.array(myRaytracer.focal_plane_color, dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    img.save('Stanford_Bunny_no_kdtree.jpg')
    img.show()

if __name__ == "__main__":
    main()