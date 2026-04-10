import math
from auxillary_classes import *



def getObjects(filename):
    vertices = []
    objects = []
    try:
        with open(filename, 'r') as file:
            triangles = []
            for line in file:
                if line[0] == "c":
                    c = line[1:].split()
                if line[0] == 'v':
                    verts = line[1:].split()
                    vertices.append(Vertex3D(round(float(verts[0]), 2),
                                             round(float(verts[1]), 2),
                                             round(float(verts[2]), 2),
                                             c)) # Placeholder color
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
        self.objects_spheres = []

    def setBackgroundColor(self, color):
        self.background_color = color

    def addObject(self, objects):
        self.objects.append(objects)

    def checkIntersection(self, triangle, ray):
        """
        Check Intersection of ray with triangle
        :param triangle: Triangle Object
        :param ray: Ray Object
        :return: Distance of intersection if Intersection exists, else -1
        """
        EPSILON = 1e-8

        # Get triangle vertices
        v0 = triangle.p0.coordinates
        v1 = triangle.p1.coordinates
        v2 = triangle.p2.coordinates

        # Calculate edges
        edge1 = v1 - v0
        edge2 = v2 - v0

        # Begin calculating determinant
        h = np.cross(ray.direction, edge2)
        det = np.dot(edge1, h)

        # Check if ray is parallel to triangle
        if abs(det) < EPSILON:
            return -1  # No intersection (ray is parallel)

        inv_det = 1.0 / det

        # Calculate vector from v0 to ray origin
        s = ray.origin - v0

        # Calculate u parameter and test bound
        u = np.dot(s, h) * inv_det
        if u < 0.0 or u > 1.0:
            return -1  # Intersection point is outside triangle

        # Prepare to test v parameter
        q = np.cross(s, edge1)

        # Calculate v parameter and test bound
        v = np.dot(ray.direction, q) * inv_det
        if v < 0.0 or u + v > 1.0:
            return -1  # Intersection point is outside triangle

        # Calculate t - distance to intersection point
        t = np.dot(edge2, q) * inv_det

        # Check if intersection is in front of ray origin
        if t > EPSILON:
            return t  # Valid intersection, return distance
        else:
            return -1  # Intersection is behind ray origin

    def checkIntersectionSphere(self, ray: 'Ray', sphere_center, sphere_radius):
        """
        Compute the intersection of a ray with a sphere.
        :param ray: Ray object
        :param sphere_center: Centre of the sphere
        :param sphere_radius: Radius of the sphere
        :return: dict or None: If an intersection exists, returns a dictionary with:
                    't'          : float – distance parameter along the ray (point = origin + t * direction)
                    'point'      : tuple – world coordinates of the intersection
                    'normal'     : tuple – unit normal at the intersection (pointing outward)
                    'front_face' : bool  – True if the ray hits the sphere from outside
                If no intersection, returns None.
        """
        o = ray.origin
        d = ray.direction - o
        c = sphere_center
        r = sphere_radius

        # Direction must be non-zero
        if np.linalg.norm(d) < 1e-12:
            return None  # degenerate ray

        # Solve quadratic: |o + t*d - c|^2 = r^2
        # Expand: (d·d) t^2 + 2 d·(o-c) t + (o-c)·(o-c) - r^2 = 0
        oc = o - c
        a = np.dot(d, d)
        b = 2.0 * np.dot(oc, d)
        c_ = np.dot(oc, oc) - r * r
        discriminant = b * b - 4.0 * a * c_

        if discriminant < 0:
            return None

        sqrt_disc = np.sqrt(discriminant)
        # Two possible t values
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        # Choose the smallest positive t
        t = None
        if t1 > 1e-8:
            t = t1
        if t2 > 1e-8 and (t is None or t2 < t):
            t = t2

        if t is None:
            return None  # both intersections are behind the origin

        # Compute intersection point and normal
        point = o + t * d
        normal = (point - c) / r
        # Determine if ray hits from outside (front face)
        front_face = np.dot(d, normal) < 0  # direction opposite to normal when hitting outside
        # Ensure normal always points against the incoming ray if front_face is true,
        # otherwise flip normal to point outward. (Common shading convention)
        if not front_face:
            normal = -normal

        return {
            't': t,
            'point': tuple(point),
            'normal': tuple(normal),
            'front_face': front_face
        }

    # def setFocalPlane(self, height, width, pixel_height, pixel_width, top_left, right, up):
    #     self.focal_plane = []
    #     for i in range(height):
    #         row = []
    #         for j in range(width):
    #             # Calculate position for pixel (j, i)
    #             x_offset = (j + 0.5) * pixel_width
    #             y_offset = (i + 0.5) * pixel_height
    #
    #             point = (top_left + right * x_offset - up * y_offset)
    #
    #             row.append(point)
    #         self.focal_plane.append(row)
    #
    # def addCamera(self, camera, height, width):
    #     self.camera = camera
    #     forward = camera.lookat - camera.position
    #
    #     right = np.cross(camera.up, forward)
    #     right = right / np.linalg.norm(right)
    #
    #     up = np.cross(forward, right)
    #     up = up / np.linalg.norm(up)
    #
    #     focal_center = camera.position + forward * camera.focal_length
    #     plane_height = 2 * camera.focal_length * np.tan(camera.fov / 2)
    #     plane_width = plane_height * (width / height)
    #
    #     pixel_width = plane_width / width
    #     pixel_height = plane_height / height
    #
    #     top_left = (focal_center - right * (plane_width / 2) + up * (plane_height / 2))
    #
    #     self.setFocalPlane(height, width, pixel_height, pixel_width, top_left, right, up)

    def setup_camera(self, camera, resolution):
        """
        Generate a 2D list of pixel positions on the focal plane in world coordinates.
        :param camera: Camera object
        :param resolution: Resolution of the image
        :return: None
        """
        self.camera = camera
        width_px, height_px = resolution
        aspect = width_px / height_px

        # Compute camera basis vectors
        forward = np.array(camera.lookat) - np.array(camera.position)
        forward = forward / np.linalg.norm(forward)

        world_up = np.array([0, 1, 0])

        # Right vector: perpendicular to forward and world_up
        right = np.cross(world_up, forward)
        if np.linalg.norm(right) < 1e-6:
            if abs(forward[0]) < 1e-6:
                right = np.array([1, 0, 0])
            else:
                right = np.array([0, 0, 1])
        else:
            right = right / np.linalg.norm(right)

        # Recompute up to guarantee orthogonality and unit length
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        # Physical size of the focal plane (in world units)
        film_height = 2.0
        film_width = aspect * film_height

        # Generate pixel centres
        self.focal_plane = []
        for i in range(height_px):
            row = []
            y = (film_height / 2) - (i + 0.5) * film_height / height_px
            for j in range(width_px):
                x = (j + 0.5) * film_width / width_px - film_width / 2
                # Transform to world coordinates
                world_point = (np.array(camera.position) +
                               x * right +
                               y * up +
                               camera.focal_length * forward)
                row.append(tuple(world_point))
            self.focal_plane.append(row)

    def render(self):
        """
        Render the image
        :return: None
        """
        self.focal_plane_color = []
        for i in range(0, len(self.focal_plane)):
            row = []
            for j in range(0, len(self.focal_plane[0])):

                # Spawn a ray
                ray = Ray(self.camera.position, self.focal_plane[i][j])
                ray_length = math.inf
                color = self.background_color

                # Check intersection with triangles
                for obj in self.objects:
                    for triangle in obj:
                        w = self.checkIntersection(triangle, ray)
                        if 0 < w < ray_length:
                            ray_length = w
                            color = triangle.p0.color

                # Check intersection with spheres
                for sphere in self.objects_spheres:
                    w = self.checkIntersectionSphere(ray, sphere.center, sphere.radius)
                    if w is not None:
                        if 0 < w['t'] < ray_length:
                            ray_length = w['t']
                            color = sphere.color
                # print(f"[{i}, {j}] = {color}")
                row.append(color)
            self.focal_plane_color.append(row)