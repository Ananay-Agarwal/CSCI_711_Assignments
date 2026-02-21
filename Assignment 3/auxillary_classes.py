import numpy as np


class Vertex3D:
    def __init__(self, x, y, z, color=(0, 0, 0)):
        self.coordinates = np.array([x, y, z])
        self.color = color

    def __str__(self) -> str:
        """
        Returns the string representation of the vertex
        :return: String
        """
        return str("X=" + str(self.x) + " Y=" + str(self.y) + " Z=" + str(self.z))

    def __repr__(self) -> str:
        """
        Returns the representation of the vertex class
        :return:
        """
        return str("X=" + str(self.x) + " Y=" + str(self.y) + " Z=" + str(self.z))

class Triangle:
    def __init__(self, vertex1, vertex2, vertex3):
        self.p0 = vertex1
        self.p1 = vertex2
        self.p2 = vertex3

    def __str__(self) -> str:
        """
        Returns the string representation of the vertex
        :return: String
        """
        return str("T1: " + str(self.p0) + "\tT2:" + str(self.p1) + "\tT3:" + str(self.p2))

    def __repr__(self) -> str:
        """
        Returns the representation of the vertex class
        :return:
        """
        return str("T1: " + str(self.p0) + "\tT2:" + str(self.p1) + "\tT3:" + str(self.p2))

class Camera:
    def __init__(self, position, focal_length, lookat):
        self.position = np.array(position)
        self.focal_length = focal_length
        self.lookat = np.array(lookat)
        self.up = np.array((0, 1, 0))

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

class Sphere:
    def __init__(self, center, radius, color):
        self.center = np.array(center)
        self.radius = radius
        self.color = color