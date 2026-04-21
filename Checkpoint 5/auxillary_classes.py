import numpy as np
from PIL import Image

class Vertex3D:
    def __init__(self, x, y, z, color=(0, 0, 0)):
        self.coordinates = np.array([x, y, z], dtype=float)
        self.color = color

    def __str__(self):
        return f"X={self.coordinates[0]} Y={self.coordinates[1]} Z={self.coordinates[2]}"

class Triangle:
    def __init__(self, vertex1, vertex2, vertex3, material=None):
        self.p0 = vertex1
        self.p1 = vertex2
        self.p2 = vertex3
        edge1 = self.p1.coordinates - self.p0.coordinates
        edge2 = self.p2.coordinates - self.p0.coordinates
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        self.normal = normal / norm if norm > 1e-8 else np.array([0, 1, 0])
        self.material = material if material else Material([0.7,0.7,0.7], [1,1,1], 0.1, 0.7, 0.3, 32)

class Material:
    def __init__(self, diffuse_color, specular_color, ka, kd, ks, ke,
                 texture_type=None, texture_image_path=None, texture_scale=1.0,
                 mapping_type='plane', color1=None, color2=None, kr=0.0):
        self.diffuse_color = np.array(diffuse_color, dtype=float)
        self.specular_color = np.array(specular_color, dtype=float)
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.ke = ke
        self.kr = kr                     # reflection coefficient

        # Texture properties
        self.texture_type = texture_type          # 'checkerboard', 'stripes', 'image'
        self.texture_scale = texture_scale
        self.mapping_type = mapping_type          # 'plane', 'sphere', 'plane_y'
        self.color1 = np.array(color1, dtype=float) if color1 else np.array([1,0,0])
        self.color2 = np.array(color2, dtype=float) if color2 else np.array([0,0,0])
        self.texture_image = None
        if texture_type == 'image' and texture_image_path:
            img = Image.open(texture_image_path).convert('RGB')
            self.texture_image = np.array(img) / 255.0   # normalize to [0,1]

class Camera:
    def __init__(self, position, focal_length, lookat):
        self.position = np.array(position, dtype=float)
        self.focal_length = focal_length
        self.lookat = np.array(lookat, dtype=float)

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin, dtype=float)
        dir_norm = np.linalg.norm(direction)
        self.direction = direction / dir_norm if dir_norm > 1e-8 else np.array([0,0,1])

class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.material = material

class Light:
    def __init__(self, position, color):
        self.position = np.array(position, dtype=float)
        self.color = np.array(color, dtype=float)