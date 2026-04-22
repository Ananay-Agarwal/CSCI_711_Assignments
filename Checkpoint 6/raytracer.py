from auxillary_classes import *
import time

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
                                   material=default_material)
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
        """Slab test. Returns (t_near, t_far) if hit, else None."""
        tmin = t_min
        tmax = t_max
        for i in range(3):
            if abs(ray.direction[i]) < 1e-8:
                # Ray parallel to slab – must start inside the slab
                if ray.origin[i] < bbox_min[i] or ray.origin[i] > bbox_max[i]:
                    return None
            else:
                inv_d = 1.0 / ray.direction[i]
                t1 = (bbox_min[i] - ray.origin[i]) * inv_d
                t2 = (bbox_max[i] - ray.origin[i]) * inv_d
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmax < tmin:
                    return None
        return tmin, tmax

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
        self.use_kdtree = True
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
        # Reflection recursion limit
        self.max_depth = 8

    def set_use_kdtree(self, use: bool):
        """Enable or disable kd-tree acceleration."""
        self.use_kdtree = use

    # Add a standalone triangle intersection method (copied from KDTree)
    def intersect_triangle(self, ray, triangle):
        """Möller–Trumbore algorithm. Returns distance t if hit, else np.inf."""
        EPSILON = 1e-8
        v0 = triangle.p0.coordinates
        v1 = triangle.p1.coordinates
        v2 = triangle.p2.coordinates

        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(ray.direction, edge2)
        det = np.dot(edge1, h)

        if abs(det) < EPSILON:
            return np.inf

        inv_det = 1.0 / det
        s = ray.origin - v0
        u = np.dot(s, h) * inv_det
        if u < 0.0 or u > 1.0:
            return np.inf

        q = np.cross(s, edge1)
        v = np.dot(ray.direction, q) * inv_det
        if v < 0.0 or u + v > 1.0:
            return np.inf

        t = np.dot(edge2, q) * inv_det
        return t if t > EPSILON else np.inf

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

    def shadow_attenuation(self, point, light_pos, normal, epsilon=1e-4):
        """Return a multiplier for light intensity (0..1) considering transparent objects."""
        light_dir = light_pos - point
        dist_to_light = np.linalg.norm(light_dir)
        if dist_to_light < 1e-6:
            return 1.0
        light_dir = light_dir / dist_to_light
        shadow_origin = point + epsilon * normal
        shadow_ray = Ray(shadow_origin, light_dir)

        transmittance = 1.0

        # Check spheres
        for sphere in self.objects_spheres:
            hit = self.checkIntersectionSphere(shadow_ray, sphere)
            if hit is not None and hit['t'] < dist_to_light:
                if hit['material'].kt > 0:
                    transmittance *= hit['material'].kt
                else:
                    return 0.0  # opaque -> full shadow
                # Continue checking further objects behind this one

        # Check triangles via kd-tree
        if self.kdtree is not None:
            # We need to find *all* intersections up to dist_to_light
            # For simplicity, we'll do a custom traversal that accumulates transmittance
            # (A full implementation would need to sort hits, but we'll approximate)
            stack = [(self.kdtree.root, 0.0, dist_to_light)]
            while stack:
                node, tmin, tmax = stack.pop()
                if node is None:
                    continue
                hit_int = self.kdtree._intersect_aabb(shadow_ray, tmin, tmax, node.bbox_min, node.bbox_max)
                if hit_int is None:
                    continue
                if node.triangles is not None:
                    for tri in node.triangles:
                        t = self.kdtree._intersect_triangle(shadow_ray, tri)
                        if 0 < t < dist_to_light:
                            if tri.material.kt > 0:
                                transmittance *= tri.material.kt
                            else:
                                return 0.0
                    continue
                stack.append((node.left, tmin, tmax))
                stack.append((node.right, tmin, tmax))

        return transmittance

    def get_uv(self, hit_point, hit_object, material):
        """Return (u, v) texture coordinates."""
        if material.mapping_type == 'sphere' and isinstance(hit_object, Sphere):
            # Spherical mapping
            p = hit_point - hit_object.center
            p = p / hit_object.radius
            u = 0.5 + np.arctan2(p[2], p[0]) / (2 * np.pi)
            v = 0.5 - np.arcsin(np.clip(p[1], -1, 1)) / np.pi
            return u, v
        elif material.mapping_type == 'plane_y':
            u = hit_point[0] * material.texture_scale
            v = hit_point[2] * material.texture_scale
            return u, v
        else:
            # Planar mapping based on dominant axis of hit normal
            # For triangles, compute normal from hit_object
            if isinstance(hit_object, Triangle):
                n = hit_object.normal
            else:
                n = (hit_point - hit_object.center) / hit_object.radius
            abs_n = np.abs(n)
            if abs_n[1] > abs_n[0] and abs_n[1] > abs_n[2]:
                u = hit_point[0] * material.texture_scale
                v = hit_point[2] * material.texture_scale
            elif abs_n[0] > abs_n[1] and abs_n[0] > abs_n[2]:
                u = hit_point[1] * material.texture_scale
                v = hit_point[2] * material.texture_scale
            else:
                u = hit_point[0] * material.texture_scale
                v = hit_point[1] * material.texture_scale
            return u, v

    def texture_color(self, u, v, material):
        """Return RGB color from texture at (u,v)."""
        if material.texture_type == 'checkerboard':
            row = int(np.floor(u)) & 1
            col = int(np.floor(v)) & 1
            if (row + col) % 2 == 0:
                return material.color1
            else:
                return material.color2
        elif material.texture_type == 'stripes':
            # vertical stripes
            if int(np.floor(u * 2)) % 2 == 0:
                return material.color1
            else:
                return material.color2
        elif material.texture_type == 'image' and material.texture_image is not None:
            # wrap u,v to [0,1)
            u = u - np.floor(u)
            v = v - np.floor(v)
            h, w, _ = material.texture_image.shape
            x = int(u * (w - 1))
            y = int(v * (h - 1))
            return material.texture_image[y, x]
        else:
            return material.diffuse_color

    def local_illumination(self, hit_point, normal, hit_object, view_dir):
        """Compute Phong shading with procedural textures (no reflection)."""
        # Extract material from hit object
        if isinstance(hit_object, Sphere):
            material = hit_object.material
        else:  # Triangle
            material = hit_object.material

        # Get texture color (replaces diffuse)
        if material.texture_type is not None:
            u, v = self.get_uv(hit_point, hit_object, material)
            tex_col = self.texture_color(u, v, material)
        else:
            tex_col = material.diffuse_color

        # Ambient
        ambient = self.ambient_light * tex_col * material.ka

        diffuse_sum = np.zeros(3)
        specular_sum = np.zeros(3)

        for light in self.lights:
            light_dir = light.position - hit_point
            dist = np.linalg.norm(light_dir)
            if dist < 1e-6: continue
            light_dir /= dist

            attenuation = self.shadow_attenuation(hit_point, light.position, normal)

            ndotl = max(0.0, np.dot(normal, light_dir))
            diffuse = ndotl * tex_col * light.color
            diffuse_sum += diffuse

            reflect_dir = 2.0 * ndotl * normal - light_dir
            vdotr = max(0.0, np.dot(view_dir, reflect_dir))
            specular = pow(vdotr, material.ke) * material.specular_color * light.color
            specular_sum += specular

            if attenuation > 0:
                ndotl = max(0.0, np.dot(normal, light_dir))
                diffuse = ndotl * tex_col * light.color * attenuation
                diffuse_sum += diffuse
                # Specular also gets attenuated
                vdotr = max(0.0, np.dot(view_dir, reflect_dir))
                specular = pow(vdotr, material.ke) * material.specular_color * light.color * attenuation
                specular_sum += specular

        result = ambient + material.kd * diffuse_sum + material.ks * specular_sum
        return np.clip(result, 0.0, 1.0)

    def illuminate(self, ray, depth):
        """Recursive ray tracing: local illumination + reflection."""
        # Find closest intersection
        best_t = np.inf
        hit_point = None
        hit_normal = None
        hit_material = None
        hit_object = None

        # Flatten all triangles for kd-tree or brute force (reuse render logic)
        all_triangles = []
        for group in self.objects:
            all_triangles.extend(group)

        # Triangle intersection
        if self.kdtree is not None:
            t_hit, tri = self.kdtree.intersect(ray)
            if t_hit < best_t and tri is not None:
                best_t = t_hit
                hit_point = ray.origin + t_hit * ray.direction
                hit_normal = tri.normal
                hit_material = tri.material
                hit_object = tri
        else:
            for tri in all_triangles:
                t = self.intersect_triangle(ray, tri)
                if 0 < t < best_t:
                    best_t = t
                    hit_point = ray.origin + t * ray.direction
                    hit_normal = tri.normal
                    hit_material = tri.material
                    hit_object = tri

        # Sphere intersection
        for sphere in self.objects_spheres:
            hit = self.checkIntersectionSphere(ray, sphere)
            if hit is not None and hit['t'] < best_t:
                best_t = hit['t']
                hit_point = hit['point']
                hit_normal = hit['normal']
                hit_material = hit['material']
                hit_object = sphere

        # No intersection -> background
        if hit_material is None:
            return self.background_color

        # View direction for local illumination (towards camera)
        view_dir = self.camera.position - hit_point
        view_dir = view_dir / np.linalg.norm(view_dir)

        # Local color
        color = self.local_illumination(hit_point, hit_normal, hit_object, view_dir)

        # Reflection
        if depth < self.max_depth and hit_material.kr > 0:
            reflected_dir = reflect_vector(ray.direction, hit_normal)
            reflection_origin = hit_point + 1e-4 * hit_normal
            reflection_ray = Ray(reflection_origin, reflected_dir)
            reflected_color = self.illuminate(reflection_ray, depth + 1)
            color += hit_material.kr * reflected_color
            color = np.clip(color, 0.0, 1.0)

        # Transmission
        if depth < self.max_depth and hit_material.kt > 0:
            # Determine outward normal and whether we are entering or exiting
            if isinstance(hit_object, Sphere):
                outward_normal = (hit_point - hit_object.center) / hit_object.radius
            else:
                outward_normal = hit_normal

            entering = np.dot(ray.direction, outward_normal) < 0
            if entering:
                eta_i = hit_material.ior_outside
                eta_t = hit_material.ior
                normal_adj = outward_normal
            else:
                eta_i = hit_material.ior
                eta_t = hit_material.ior_outside
                normal_adj = -outward_normal

            eta = eta_i / eta_t
            cos_theta_i = -np.dot(ray.direction, normal_adj)
            cos_theta_i = max(0.0, min(1.0, cos_theta_i))
            sin2_theta_t = eta * eta * (1.0 - cos_theta_i * cos_theta_i)

            if sin2_theta_t > 1.0:
                # Total internal reflection
                transmitted_dir = reflect_vector(ray.direction, normal_adj)
            else:
                sqrt_term = np.sqrt(1.0 - sin2_theta_t)
                transmitted_dir = eta * ray.direction + (eta * cos_theta_i - sqrt_term) * normal_adj
                transmitted_dir = transmitted_dir / np.linalg.norm(transmitted_dir)

            trans_origin = hit_point + 1e-4 * transmitted_dir
            trans_ray = Ray(trans_origin, transmitted_dir)
            trans_color = self.illuminate(trans_ray, depth + 1)
            color += hit_material.kt * trans_color
            color = np.clip(color, 0.0, 1.0)

        return color

    def render(self):
        """Render the image using either kd-tree or brute-force."""
        # Flatten all triangles
        all_triangles = []
        for group in self.objects:
            all_triangles.extend(group)

        # Build kd-tree only if requested and there are triangles
        if self.use_kdtree and all_triangles:
            print("Creating kdtree....", end="")
            start = time.perf_counter()
            self.kdtree = KDTree(all_triangles, leaf_size=8, max_depth=20)
            end = time.perf_counter()
            print(f"done in {end - start:.6f} seconds.")
        else:
            self.kdtree = None   # brute-force will be used

        print("Starting Render....", end="")
        start = time.perf_counter()
        self.focal_plane_color = []
        for i, row in enumerate(self.focal_plane):
            img_row = []
            for j, pixel_world in enumerate(row):
                ray = Ray(self.camera.position, pixel_world - self.camera.position)
                color = self.illuminate(ray, depth=1)
                img_row.append(tuple(int(c*255) for c in color))
            self.focal_plane_color.append(img_row)
        end = time.perf_counter()
        print(f"done in {end - start:.6f} seconds.")