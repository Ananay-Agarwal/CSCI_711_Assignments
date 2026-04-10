import math
import numpy as np
from auxillary_classes import *

def reflect_vector(incident, normal):
    """Return reflection of incident vector about normal.
       incident and normal are assumed normalized."""
    return incident - 2.0 * np.dot(incident, normal) * normal

def getObjects(filename):
    vertices = []
    objects = []
    default_material = Material([0.7,0.7,0.7], [1,1,1], 0.1, 0.7, 0.3, 32)
    try:
        with open(filename, 'r') as file:
            triangles = []
            current_color = [0.7, 0.7, 0.7]  # fallback
            for line in file:
                if line[0] == 'c':
                    # optional color hint, store as RGB 0-1
                    parts = line[1:].split()
                    if len(parts) >= 3:
                        current_color = [float(p)/255.0 for p in parts[:3]]
                if line[0] == 'v':
                    verts = line[1:].split()
                    v = Vertex3D(round(float(verts[0]),2),
                                 round(float(verts[1]),2),
                                 round(float(verts[2]),2),
                                 tuple(current_color))
                    vertices.append(v)
                if line[0] == 'o' and len(triangles) > 0:
                    objects += triangles
                    triangles = []
                if line[0] == 'f':
                    indices = [int(x)-1 for x in line[1:].split()]
                    # Create material per triangle using current_color as diffuse
                    mat = Material(current_color, [1,1,1], 0.1, 0.7, 0.3, 32)
                    tri = Triangle(vertices[indices[0]],
                                   vertices[indices[1]],
                                   vertices[indices[2]],
                                   material=mat)
                    triangles.append(tri)
            objects += triangles
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return vertices, objects


class KDNode:
    __slots__ = ('bbox_min', 'bbox_max', 'left', 'right', 'triangles', 'axis')
    def __init__(self, bbox_min, bbox_max, left=None, right=None, triangles=None, axis=None):
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.left = left
        self.right = right
        self.triangles = triangles
        self.axis = axis


class KDTree:
    def __init__(self, triangles, leaf_size=8, max_depth=20):
        self.leaf_size = leaf_size
        self.max_depth = max_depth
        self.root = self._build(triangles, depth=0)

    def _compute_bbox(self, triangles):
        min_ = np.full(3, np.inf)
        max_ = np.full(3, -np.inf)
        for tri in triangles:
            for v in (tri.p0.coordinates, tri.p1.coordinates, tri.p2.coordinates):
                min_ = np.minimum(min_, v)
                max_ = np.maximum(max_, v)
        return min_, max_

    def _centroid(self, tri):
        return (tri.p0.coordinates + tri.p1.coordinates + tri.p2.coordinates) / 3.0

    def _build(self, triangles, depth):
        if not triangles:
            return None
        bbox_min, bbox_max = self._compute_bbox(triangles)
        if len(triangles) <= self.leaf_size or depth >= self.max_depth:
            return KDNode(bbox_min, bbox_max, triangles=triangles)
        axis = depth % 3
        triangles.sort(key=lambda t: self._centroid(t)[axis])
        median = len(triangles)//2
        left_child = self._build(triangles[:median], depth+1)
        right_child = self._build(triangles[median:], depth+1)
        return KDNode(bbox_min, bbox_max, left=left_child, right=right_child, axis=axis)

    def _intersect_aabb(self, ray, t_min, t_max, bbox_min, bbox_max):
        inv_dir = 1.0 / ray.direction
        tx1 = (bbox_min[0] - ray.origin[0]) * inv_dir[0]
        tx2 = (bbox_max[0] - ray.origin[0]) * inv_dir[0]
        tmin = min(tx1, tx2)
        tmax = max(tx1, tx2)
        ty1 = (bbox_min[1] - ray.origin[1]) * inv_dir[1]
        ty2 = (bbox_max[1] - ray.origin[1]) * inv_dir[1]
        tmin = max(tmin, min(ty1, ty2))
        tmax = min(tmax, max(ty1, ty2))
        tz1 = (bbox_min[2] - ray.origin[2]) * inv_dir[2]
        tz2 = (bbox_max[2] - ray.origin[2]) * inv_dir[2]
        tmin = max(tmin, min(tz1, tz2))
        tmax = min(tmax, max(tz1, tz2))
        if tmax >= tmin and tmax > 0 and tmin < tmax:
            return max(tmin, 0.0), tmax
        return None

    def _intersect_triangles(self, ray, triangles):
        closest_t = np.inf
        closest_tri = None
        for tri in triangles:
            t = self._intersect_triangle(ray, tri)
            if 0 < t < closest_t:
                closest_t = t
                closest_tri = tri
        return closest_t, closest_tri

    def _intersect_triangle(self, ray, triangle):
        EPS = 1e-8
        v0, v1, v2 = triangle.p0.coordinates, triangle.p1.coordinates, triangle.p2.coordinates
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(ray.direction, edge2)
        det = np.dot(edge1, h)
        if abs(det) < EPS:
            return np.inf
        inv_det = 1.0 / det
        s = ray.origin - v0
        u = np.dot(s, h) * inv_det
        if u < 0.0 or u > 1.0:
            return np.inf
        q = np.cross(s, edge1)
        v = np.dot(ray.direction, q) * inv_det
        if v < 0.0 or u+v > 1.0:
            return np.inf
        t = np.dot(edge2, q) * inv_det
        return t if t > EPS else np.inf

    def _traverse(self, ray, node, t_min, t_max):
        if node is None:
            return np.inf, None
        hit_interval = self._intersect_aabb(ray, t_min, t_max, node.bbox_min, node.bbox_max)
        if hit_interval is None:
            return np.inf, None
        if node.triangles is not None:
            return self._intersect_triangles(ray, node.triangles)
        axis = node.axis
        if ray.direction[axis] < 0:
            first, second = node.right, node.left
        else:
            first, second = node.left, node.right
        t_hit, tri_hit = self._traverse(ray, first, t_min, t_max)
        if t_hit < np.inf:
            t_max = t_hit
        t2, tri2 = self._traverse(ray, second, t_min, t_max)
        if t2 < t_hit:
            t_hit, tri_hit = t2, tri2
        return t_hit, tri_hit

    def intersect(self, ray):
        return self._traverse(ray, self.root, 0.0, np.inf)

    def any_intersection_before(self, ray, max_t):
        """Return True if any triangle intersection with t < max_t exists."""
        # simple stack traversal that stops at first hit
        stack = [(self.root, 0.0, max_t)]
        while stack:
            node, tmin, tmax = stack.pop()
            if node is None:
                continue
            hit_int = self._intersect_aabb(ray, tmin, tmax, node.bbox_min, node.bbox_max)
            if hit_int is None:
                continue
            if node.triangles is not None:
                for tri in node.triangles:
                    t = self._intersect_triangle(ray, tri)
                    if 0 < t < max_t:
                        return True
                continue
            # internal node: push both children
            stack.append((node.left, tmin, tmax))
            stack.append((node.right, tmin, tmax))
        return False


class Raytracer:
    def __init__(self):
        self.background_color = np.array([0, 0, 0], dtype=float)
        self.camera = None
        self.focal_plane = None
        self.focal_plane_color = None
        self.objects = []          # list of triangle groups
        self.objects_spheres = []
        self.kdtree = None
        # Lighting
        self.lights = []           # list of Light objects
        self.ambient_light = np.array([0.2, 0.2, 0.2], dtype=float)

    def setBackgroundColor(self, color):
        self.background_color = np.array(color, dtype=float) / 255.0

    def addObject(self, objects):
        self.objects.append(objects)

    def checkIntersectionSphere(self, ray, sphere):
        o = ray.origin
        d = ray.direction
        c = sphere.center
        r = sphere.radius
        oc = o - c
        a = np.dot(d, d)
        b = 2.0 * np.dot(oc, d)
        c_ = np.dot(oc, oc) - r*r
        disc = b*b - 4*a*c_
        if disc < 0:
            return None
        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        t = None
        if t1 > 1e-8:
            t = t1
        if t2 > 1e-8 and (t is None or t2 < t):
            t = t2
        if t is None:
            return None
        point = o + t * d
        normal = (point - c) / r
        front_face = np.dot(d, normal) < 0
        if not front_face:
            normal = -normal
        return {
            't': t,
            'point': point,
            'normal': normal,
            'front_face': front_face,
            'material': sphere.material
        }

    def setup_camera(self, camera, resolution):
        self.camera = camera
        width_px, height_px = resolution
        aspect = width_px / height_px
        forward = camera.lookat - camera.position
        forward = forward / np.linalg.norm(forward)
        world_up = np.array([0, 1, 0])
        right = np.cross(world_up, forward)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1,0,0]) if abs(forward[0])<1e-6 else np.array([0,0,1])
        else:
            right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)
        film_height = 2.0
        film_width = aspect * film_height
        self.focal_plane = []
        for i in range(height_px):
            row = []
            y = (film_height/2) - (i+0.5)*film_height/height_px
            for j in range(width_px):
                x = (j+0.5)*film_width/width_px - film_width/2
                world_point = camera.position + x*right + y*up + camera.focal_length*forward
                row.append(world_point)
            self.focal_plane.append(row)

    def is_shadowed(self, point, light_pos, normal, epsilon=1e-4):
        """Return True if any object (sphere or triangle) blocks the light."""
        light_dir = light_pos - point
        dist_to_light = np.linalg.norm(light_dir)
        if dist_to_light < 1e-6:
            return False
        light_dir = light_dir / dist_to_light
        # Offset origin slightly along normal to avoid self-intersection
        shadow_origin = point + epsilon * normal
        shadow_ray = Ray(shadow_origin, light_dir)

        # Check spheres
        for sphere in self.objects_spheres:
            hit = self.checkIntersectionSphere(shadow_ray, sphere)
            if hit is not None and hit['t'] < dist_to_light:
                return True

        # Check triangles via kd-tree
        if self.kdtree is not None:
            if self.kdtree.any_intersection_before(shadow_ray, dist_to_light):
                return True
        return False

    def shade(self, hit_point, normal, material, view_dir):
        """Compute Phong shading at hit_point."""
        # Ambient
        ambient = self.ambient_light * material.diffuse_color * material.ka

        # Accumulate diffuse + specular from all lights
        diffuse_sum = np.zeros(3)
        specular_sum = np.zeros(3)

        for light in self.lights:
            light_dir = light.position - hit_point
            dist_to_light = np.linalg.norm(light_dir)
            if dist_to_light < 1e-6:
                continue
            light_dir = light_dir / dist_to_light

            # Shadow test
            if self.is_shadowed(hit_point, light.position, normal):
                continue

            # Diffuse
            ndotl = max(0.0, np.dot(normal, light_dir))
            diffuse = ndotl * material.diffuse_color * light.color
            diffuse_sum += diffuse

            # Specular
            # R = 2*(N·L)*N - L
            reflect_dir = 2.0 * ndotl * normal - light_dir
            vdotr = max(0.0, np.dot(view_dir, reflect_dir))
            specular = pow(vdotr, material.ke) * material.specular_color * light.color
            specular_sum += specular

        result = ambient + material.kd * diffuse_sum + material.ks * specular_sum
        return np.clip(result, 0.0, 1.0)

    def render(self):
        # Flatten all triangles into a single list for kd-tree
        all_triangles = []
        for group in self.objects:
            all_triangles.extend(group)
        if all_triangles:
            self.kdtree = KDTree(all_triangles, leaf_size=8, max_depth=20)
        else:
            self.kdtree = None

        self.focal_plane_color = []
        for i, row in enumerate(self.focal_plane):
            img_row = []
            for j, pixel_world in enumerate(row):
                ray = Ray(self.camera.position, pixel_world - self.camera.position)
                best_t = np.inf
                hit_material = None
                hit_point = None
                hit_normal = None

                # Triangle intersection via kd-tree
                if self.kdtree is not None:
                    t_hit, tri = self.kdtree.intersect(ray)
                    if t_hit < best_t and tri is not None:
                        best_t = t_hit
                        hit_point = ray.origin + t_hit * ray.direction
                        hit_normal = tri.normal
                        hit_material = tri

                # Sphere intersection
                for sphere in self.objects_spheres:
                    hit = self.checkIntersectionSphere(ray, sphere)
                    if hit is not None and hit['t'] < best_t:
                        best_t = hit['t']
                        hit_point = hit['point']
                        hit_normal = hit['normal']
                        hit_material = hit['material']

                if hit_material is None:
                    color = self.background_color
                else:
                    # View direction from hit point to camera
                    view_dir = self.camera.position - hit_point
                    view_dir = view_dir / np.linalg.norm(view_dir)
                    color = self.shade(hit_point, hit_normal, hit_material, view_dir)

                # Convert 0-1 float to 0-255 int
                img_row.append(tuple(int(c*255) for c in color))
            self.focal_plane_color.append(img_row)