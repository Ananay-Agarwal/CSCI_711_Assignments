from raytracer import *
from auxillary_classes import *
from PIL import Image

FILE_PATH = 'scene_1.obj'

def main():
    myRaytracer = Raytracer()
    myRaytracer.setBackgroundColor((135, 206, 235))

    camera = Camera([0.025, 2.81, 0.571], 0.05021, [-20.9753, 2.81, 0.571], 39.4446)
    myRaytracer.addCamera(camera, 500, 500)

    vertices, objects = getObjects(FILE_PATH)
    myRaytracer.addObject(objects)

    myRaytracer.render()

    img_array = np.array(myRaytracer.focal_plane_color, dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    img.show()
    img.save('Scene_1_Render.jpg')

if __name__ == "__main__":
    main()