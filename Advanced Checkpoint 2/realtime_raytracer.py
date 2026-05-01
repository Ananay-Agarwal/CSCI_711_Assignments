import taichi as ti
import numpy as np

# Initialize with CUDA (uses RTX card if available)
ti.init(arch=ti.cuda, default_fp=ti.f32)

# Window dimensions
WIDTH, HEIGHT = 800, 600
ASPECT_RATIO = WIDTH / HEIGHT

# Scene parameters
FOV = 1.2  # Field of view (radians)
MAX_BOUNCES = 3
EPS = 1e-4

# Camera control
camera_angle_x = 0.0
camera_angle_y = 0.5
camera_distance = 5.0
last_mouse_pos = (0, 0)


# Ray tracer data structures
@ti.dataclass
class Ray:
    origin: ti.math.vec3
    direction: ti.math.vec3


@ti.dataclass
class HitRecord:
    hit: ti.i32
    point: ti.math.vec3
    normal: ti.math.vec3
    t: ti.f32
    material_type: ti.i32  # 0: diffuse, 1: ground
    albedo: ti.math.vec3


# Define scene objects as fields
num_spheres = 4
sphere_centers = ti.Vector.field(3, dtype=ti.f32, shape=num_spheres)
sphere_radii = ti.field(dtype=ti.f32, shape=num_spheres)
sphere_albedos = ti.Vector.field(3, dtype=ti.f32, shape=num_spheres)

# Ground plane parameters
ground_y = -1.5
ground_albedo = ti.math.vec3(0.5, 0.5, 0.5)

# Light
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
light_pos[None] = ti.math.vec3(3.0, 5.0, 2.0)
light_color = ti.math.vec3(1.0, 1.0, 0.9) * 1.5
ambient = ti.math.vec3(0.1, 0.1, 0.15)

# Output buffer
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))


def init_scene():
    # Sphere 0: red ball
    sphere_centers[0] = ti.math.vec3(-1.2, -0.5, 0.0)
    sphere_radii[0] = 0.8
    sphere_albedos[0] = ti.math.vec3(0.9, 0.2, 0.2)

    # Sphere 1: green ball
    sphere_centers[1] = ti.math.vec3(1.5, -0.2, -1.0)
    sphere_radii[1] = 0.7
    sphere_albedos[1] = ti.math.vec3(0.2, 0.8, 0.2)

    # Sphere 2: blue ball
    sphere_centers[2] = ti.math.vec3(0.0, 0.2, 1.2)
    sphere_radii[2] = 0.6
    sphere_albedos[2] = ti.math.vec3(0.2, 0.3, 1.0)

    # Sphere 3: yellow small ball
    sphere_centers[3] = ti.math.vec3(-0.5, -1.0, 2.0)
    sphere_radii[3] = 0.5
    sphere_albedos[3] = ti.math.vec3(0.9, 0.8, 0.2)


@ti.func
def intersect_sphere(ray, center, radius):
    oc = ray.origin - center
    a = ray.direction.dot(ray.direction)
    b = 2.0 * oc.dot(ray.direction)
    c = oc.dot(oc) - radius * radius
    discriminant = b * b - 4.0 * a * c

    t = -1.0
    if discriminant >= 0.0:
        sqrt_d = ti.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)
        if t1 > EPS:
            t = t1
        elif t2 > EPS:
            t = t2
    return t


@ti.func
def intersect_ground(ray):
    t = -1.0
    if abs(ray.direction.y) >= EPS:
        t_candidate = (ground_y - ray.origin.y) / ray.direction.y
        if t_candidate > EPS:
            t = t_candidate
    return t


@ti.func
def trace_world(ray):
    hit_record = HitRecord()
    hit_record.hit = 0
    closest_t = 1e9

    # Check spheres
    for i in range(num_spheres):
        t = intersect_sphere(ray, sphere_centers[i], sphere_radii[i])
        if t > 0 and t < closest_t:
            closest_t = t
            hit_record.hit = 1
            hit_record.t = t
            hit_record.point = ray.origin + ray.direction * t
            hit_record.normal = (hit_record.point - sphere_centers[i]).normalized()
            hit_record.material_type = 0  # diffuse sphere
            hit_record.albedo = sphere_albedos[i]

    # Check ground
    t_ground = intersect_ground(ray)
    if t_ground > 0 and t_ground < closest_t:
        closest_t = t_ground
        hit_record.hit = 1
        hit_record.t = t_ground
        hit_record.point = ray.origin + ray.direction * t_ground
        hit_record.normal = ti.math.vec3(0.0, 1.0, 0.0)
        hit_record.material_type = 1  # ground
        # Checkerboard pattern
        xz = ti.math.floor(hit_record.point.x * 2.0) + ti.math.floor(hit_record.point.z * 2.0)
        if ti.abs(xz) % 2 == 0:
            hit_record.albedo = ti.math.vec3(0.3, 0.3, 0.3)
        else:
            hit_record.albedo = ti.math.vec3(0.7, 0.7, 0.7)

    return hit_record


@ti.func
def compute_shadow(ray, light_dist):
    shadow_hit = trace_world(ray)
    # If we hit something before the light, we're in shadow
    return shadow_hit.hit == 1 and shadow_hit.t < light_dist - EPS


@ti.func
def calculate_lighting(hit, view_dir):
    light_dir = (light_pos[None] - hit.point).normalized()
    light_dist = (light_pos[None] - hit.point).norm()
    shadow_ray = Ray(origin=hit.point + hit.normal * EPS, direction=light_dir)
    in_shadow = compute_shadow(shadow_ray, light_dist)

    color = ambient * hit.albedo
    if not in_shadow:
        diff = max(0.0, hit.normal.dot(light_dir))
        intensity = light_color * diff
        reflect_dir = (2.0 * hit.normal.dot(light_dir) * hit.normal - light_dir).normalized()
        spec = max(0.0, reflect_dir.dot(-view_dir)) ** 32
        spec_color = ti.math.vec3(0.5, 0.5, 0.5) * spec
        color = (ambient + intensity) * hit.albedo + spec_color
    return color


@ti.func
def trace_ray(ray, max_depth):
    color = ti.math.vec3(0.0)
    throughput = ti.math.vec3(1.0)
    current_ray = ray

    for _ in range(max_depth):
        hit = trace_world(current_ray)

        if not hit.hit:
            # Sky color contribution
            t = (current_ray.direction.y + 1.0) * 0.5
            sky_color = (1.0 - t) * ti.math.vec3(1.0, 1.0, 1.0) + t * ti.math.vec3(0.5, 0.7, 1.0)
            color += throughput * sky_color
            break

        # Direct lighting at this hit point
        view_dir = -current_ray.direction
        direct = calculate_lighting(hit, view_dir)
        color += throughput * direct

        # Update throughput for indirect bounce (diffuse reflection)
        throughput *= hit.albedo * 0.3   # attenuation factor

        # Generate new random bounce direction (cosine‑weighted hemisphere)
        n = hit.normal
        u1 = ti.random()
        u2 = ti.random()
        r = ti.sqrt(u1)
        theta = 2.0 * 3.14159 * u2
        x = r * ti.cos(theta)
        y = r * ti.sin(theta)
        z = ti.sqrt(1.0 - u1)
        bounce_dir = ti.math.vec3(x, y, z)
        if bounce_dir.dot(n) < 0.0:
            bounce_dir = -bounce_dir

        # Create the next ray from the hit point
        current_ray = Ray(origin=hit.point + n * EPS, direction=bounce_dir)

    return color


@ti.kernel
def render(cam_pos: ti.math.vec3, cam_target: ti.math.vec3):
    # Compute camera basis
    forward = (cam_target - cam_pos).normalized()
    right = ti.math.vec3(0.0, 1.0, 0.0).cross(forward).normalized()
    up = forward.cross(right).normalized()

    for i, j in ti.ndrange(WIDTH, HEIGHT):
        # Normalized screen coordinates (-1 to 1)
        u = (2.0 * i / WIDTH - 1.0) * ti.tan(FOV / 2.0) * ASPECT_RATIO
        v = (1.0 - 2.0 * j / HEIGHT) * ti.tan(FOV / 2.0)

        # Ray direction
        ray_dir = (forward + right * u + up * v).normalized()
        ray = Ray(origin=cam_pos, direction=ray_dir)

        # Accumulate color
        color = trace_ray(ray, MAX_BOUNCES)

        # Gamma correction and clamping
        color = ti.math.clamp(color, 0.0, 1.0)
        color = ti.math.pow(color, 1.0 / 2.2)
        pixels[i, j] = color


def main():
    global camera_angle_x, camera_angle_y, last_mouse_pos, camera_distance
    init_scene()

    window = ti.ui.Window("Real-Time Ray Tracer (RTX GPU)", (WIDTH, HEIGHT),
                          vsync=True, show_window=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    # Mouse tracking state
    last_mouse_x = -1
    last_mouse_y = -1
    mouse_sensitivity = 0.8

    # Animation time
    t = 0.0

    def update_camera():
        center = ti.math.vec3(0.0, 0.0, 0.0)
        x = camera_distance * ti.cos(camera_angle_x) * ti.cos(camera_angle_y)
        z = camera_distance * ti.sin(camera_angle_x) * ti.cos(camera_angle_y)
        y = camera_distance * ti.sin(camera_angle_y) + 1.0
        return ti.math.vec3(x, y, z), center

    print("Rendering on GPU...")

    while window.running:
        t += 0.005

        # Animate light position
        radius = 4.0
        light_x = ti.math.sin(t) * radius
        light_z = ti.math.cos(t * 0.7) * radius
        light_y = 3.5 + ti.math.sin(t * 1.3) * 1.2
        light_pos[None] = ti.math.vec3(light_x, light_y, light_z)

        camera_distance = max(2.0, min(12.0, camera_distance))

        if window.is_pressed(ti.ui.LMB):
            # Get current cursor position (pixel coordinates)
            mouse_x, mouse_y = window.get_cursor_pos()
            if last_mouse_x != -1 and last_mouse_y != -1:
                # Compute delta and update angles
                dx = mouse_x - last_mouse_x
                dy = mouse_y - last_mouse_y
                camera_angle_x += dx * mouse_sensitivity
                camera_angle_y += dy * mouse_sensitivity
                # Clamp vertical angle to avoid flipping (approx ±85 degrees)
                camera_angle_y = max(-1.48, min(1.48, camera_angle_y))
            # Update last known mouse position
            last_mouse_x, last_mouse_y = mouse_x, mouse_y
        else:
            # Reset tracking when button is released
            last_mouse_x, last_mouse_y = -1, -1

        cam_pos, cam_target = update_camera()
        render(cam_pos, cam_target)

        canvas.set_image(pixels)
        window.show()

        # Exit
        if window.is_pressed(ti.ui.ESCAPE):
            break


if __name__ == "__main__":
    main()