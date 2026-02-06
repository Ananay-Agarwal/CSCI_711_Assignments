import math
import numpy as np
from auxillary_classes import *



def getObjects(filename):
    vertices = []
    objects = []
    try:
        with open(filename, 'r') as file:
            triangles = []
            for line in file:
                if line[0] == 'v':
                    verts = line[1:].split()
                    vertices.append(Vertex3D(round(float(verts[0]), 2),
                                             round(float(verts[1]), 2),
                                             round(float(verts[2]), 2),
                                             (255, 0, 0))) # Placeholder color
                if line[0] == 'o' and len(triangles) > 0:
                    objects += triangles
                    triangles = []
                if line[0] == 'f':
                    indices = [int(x)-1 for x in line[1:].split()]
                    triangles.append(Triangle(vertices[indices[0]], vertices[indices[1]], vertices[indices[2]]))
            objects += triangles
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return vertices, objects

class Raytracer:
    def __init__(self):
        self.background_color = (0, 0, 0)
        self.camera = None
        self.focal_plane = None
        self.focal_plane_color = None
        self.objects = []

    def setBackgroundColor(self, color):
        self.background_color = color

    def addObject(self, objects):
        self.objects.append(objects)

    def checkIntersection(self, triangle, ray):
        e1 = triangle.p1.coordinates - triangle.p0.coordinates
        e2 = triangle.p2.coordinates - triangle.p0.coordinates
        T = ray.origin - triangle.p0.coordinates
        P = np.cross(ray.direction, e2)
        Q = np.cross(T, e1)

        if np.dot(P, e1) == 0:
            return -1

        barycentric_coords = np.array([[np.dot(Q, e1)], [np.dot(P, T)], [np.dot(Q, ray.direction)]]) / np.dot(P, e1)
        w, u, v = barycentric_coords[0], barycentric_coords[1], barycentric_coords[2]

        # Add a condition to check if w is the highest for the pixel
        if w < 0 or u < 0 or v < 0 or u+v > 1:
            return -1
        else:
            return w

    def checkIntersectionSphere(self, center):
        pass

    def setFocalPlane(self, height, width, pixel_height, pixel_width, top_left, right, up):
        self.focal_plane = []
        for i in range(height):
            row = []
            for j in range(width):
                # Calculate position for pixel (j, i)
                x_offset = (j + 0.5) * pixel_width
                y_offset = (i + 0.5) * pixel_height

                point = (top_left + right * x_offset - up * y_offset)

                row.append(point)
            self.focal_plane.append(row)

    def addCamera(self, camera, height, width):
        self.camera = camera
        forward = camera.lookat - camera.position

        right = np.cross(camera.up, forward)
        right = right / np.linalg.norm(right)

        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        focal_center = camera.position + forward * camera.focal_length
        plane_height = 2 * camera.focal_length * np.tan(camera.fov / 2)
        plane_width = plane_height * (width / height)

        pixel_width = plane_width / width
        pixel_height = plane_height / height

        top_left = (focal_center - right * (plane_width / 2) + up * (plane_height / 2))

        self.setFocalPlane(height, width, pixel_height, pixel_width, top_left, right, up)

    def render(self):
        self.focal_plane_color = []
        for i in range(0, len(self.focal_plane)):
            row = []
            for j in range(0, len(self.focal_plane[0])):
                # Spawn a ray
                ray = Ray(self.camera.position, self.focal_plane[i][j])
                ray_length = math.inf
                color = self.background_color
                for obj in self.objects:
                    for triangle in obj:
                        w = self.checkIntersection(triangle, ray)
                        if 0 < w < ray_length:
                            ray_length = w
                            color = triangle.color
                # print(f"[{i}, {j}] = {color}")
                row.append(color)
            self.focal_plane_color.append(row)