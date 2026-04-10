import numpy as np

class Vertex3D:
    def __init__(self, x, y, z, color=(0, 0, 0)):
        self.coordinates = np.array([x, y, z], dtype=float)
        self.color = color   # not used for shading, kept for compatibility

    def __str__(self):
        return f"X={self.coordinates[0]} Y={self.coordinates[1]} Z={self.coordinates[2]}"

    def __repr__(self):
        return str(self)


class Triangle:
    def __init__(self, vertex1, vertex2, vertex3, material=None):
        self.p0 = vertex1
        self.p1 = vertex2
        self.p2 = vertex3

        # Compute flat normal (counter‑clockwise order assumed)
        edge1 = self.p1.coordinates - self.p0.coordinates
        edge2 = self.p2.coordinates - self.p0.coordinates
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        self.normal = normal / norm if norm > 1e-8 else np.array([0, 1, 0])

        # Material properties (default if none provided)
        if material is None:
            # Default: gray diffuse, white specular, typical coefficients
            self.diffuse_color = np.array([0.7, 0.7, 0.7])
            self.specular_color = np.array([1.0, 1.0, 1.0])
            self.ka = 0.1
            self.kd = 0.7
            self.ks = 0.3
            self.ke = 32.0
        else:
            self.diffuse_color = material.diffuse_color
            self.specular_color = material.specular_color
            self.ka = material.ka
            self.kd = material.kd
            self.ks = material.ks
            self.ke = material.ke

    def __str__(self):
        return f"Triangle: {self.p0}, {self.p1}, {self.p2}"

    def __repr__(self):
        return str(self)


class Material:
    """Simple material container for Phong shading."""
    def __init__(self, diffuse_color, specular_color, ka, kd, ks, ke):
        self.diffuse_color = np.array(diffuse_color, dtype=float)
        self.specular_color = np.array(specular_color, dtype=float)
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.ke = ke


class Camera:
    def __init__(self, position, focal_length, lookat):
        self.position = np.array(position, dtype=float)
        self.focal_length = focal_length
        self.lookat = np.array(lookat, dtype=float)
        self.up = np.array([0, 1, 0], dtype=float)


class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin, dtype=float)
        # Normalize direction
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 1e-8:
            self.direction = np.array(direction, dtype=float) / dir_norm
        else:
            self.direction = np.array([0, 0, 1], dtype=float)


class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.material = material

    # For backward compatibility, provide color property (used in old code)
    @property
    def color(self):
        # Convert diffuse color to 0-255 integer tuple (if needed)
        return tuple(int(c*255) for c in self.material.diffuse_color)


class Light:
    def __init__(self, position, color):
        self.position = np.array(position, dtype=float)
        self.color = np.array(color, dtype=float)   # RGB in [0,1]