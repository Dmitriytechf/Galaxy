# type: ignore
import taichi as ti
import taichi.math as tm


ti.init(arch=ti.gpu)

# Количество частиц
N = 1_000_000

N_BULGE = int(N * 0.20)          # плотное ядро
N_ARMS  = int(N * 0.55)          # рукава
N_DISK  = N - N_BULGE - N_ARMS   # размытый диск между рукавами

STRETCH_X = 1.3     # растяжение Галактики вдоль X
STRETCH_Z = 0.9    # сжатие вдоль Z

# Позиции частиц: 1D поле из 3D-векторов
pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
color = ti.Vector.field(3, dtype=ti.f32, shape=N)

@ti.func 
def random_dir() -> tm.vec3:
    '''Равномерное направление на сфере.'''
    z = ti.random() * 2 - 1
    phi = ti.random() * 2 * tm.pi
    r_xy = tm.sqrt(1 - z * z)
    return tm.vec3(r_xy * tm.cos(phi), r_xy * tm.sin(phi), z)


@ti.func 
def star_color(r: ti.f32) -> tm.vec3:
    '''
    Цвет по радиусу: жёлто-белый в центре, голубой на краю.
    '''
    t = ti.min(r, 1.0)
    # тёплый центр
    warm = tm.vec3(1.00, 0.85, 0.55)
    # холодная периферия
    cold = tm.vec3(0.55, 0.75, 1.00)
    return warm * (1.0 - t) + cold * t


@ti.kernel
def init_particles():
    '''
    Инициализируем начальные позиции частиц.
    Milky Way-like: плотный балдж, два главных рукава,
    два вторичных и размытый диск между ними.
    '''
    # Балдж
    for i in range(N_BULGE):
        # плотнее к центру, но с длинным хвостом наружу
        r = 0.15 * ti.pow(ti.random(), 1.5)
        d = random_dir()
        pos[i] = tm.vec3(d.x * r * 2 , d.y * r * 1.2, d.z * r * 1.0)
        color[i] = star_color(r) * 2.7

    # Рукава
    for i in range(N_ARMS):
        idx = N_BULGE + i
        r = 0.10 + 0.90 * ti.pow(ti.random(), 2.2)
        # 4 рукава
        arm_id = 0
        arm_raw = ti.random()
        if arm_raw < 0.35:
            arm_id = 0
        elif arm_raw < 0.70:
            arm_id = 1
        elif arm_raw < 0.85:
            arm_id = 2
        else:
            arm_id = 3
        # угол базы рукава
        theta_arm = ti.cast(arm_id, ti.f32) * (tm.pi * 0.5)

        # логарифмическая спираль — как у настоящих галактик
        theta = theta_arm + 2.8 * tm.log(r / 0.10 + 1.0)

        # разброс растёт с радиусом: рукава расплываются к краю
        spread = 0.08 + 0.45 * r
        theta += (ti.random() + ti.random() + ti.random() - 1.5) * spread

        # толщина диска по Y
        y = (ti.random() + ti.random() + ti.random() - 1.5) * 0.03

        pos[idx] = tm.vec3(r * tm.cos(theta) * STRETCH_X, y, r * tm.sin(theta) * STRETCH_Z)
        color[idx] = star_color(r) * 2.0

    # Размытый диск между рукавами
    for i in range(N_DISK):
        idx = N_BULGE + N_ARMS + i
        r = 0.10 + 0.90 * ti.pow(ti.random(), 3.0)
        # угол — равномерный по всему диску, без привязки к рукавам
        theta = ti.random() * 2 * tm.pi
        theta += 0.4 * tm.sin(2 * theta + 4.0 * r)
        y = (ti.random() + ti.random() + ti.random() - 1.5) * 0.05
        pos[idx] = tm.vec3(r * tm.cos(theta) * STRETCH_X, y, r * tm.sin(theta) * STRETCH_Z)
        color[idx] = star_color(r) * 1.8   # чуть тусклее рукавов


init_particles()

window = ti.ui.Window("Galaxy", (1024, 768), vsync=True)
canvas = window.get_canvas() # поверхность для 2D-рисования.
scene = window.get_scene() # 3D-сцена
camera = ti.ui.Camera()

# Камера смотрит в начало координат
camera.position(0, 1.8, 2.5)
camera.lookat(0, 0, 0)
scene.set_camera(camera)

scene.point_light(pos=(5, 5, 5), color=(1, 1, 1))
scene.ambient_light((0.3, 0.3, 0.3))

# render loop — аналог while True
while window.running:
    # Управление камерой (ПКМ)
    camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.LMB)
    scene.set_camera(camera)
    canvas.set_background_color((0.0, 0.0, 0.05))
    
    # Рисуем частицы
    scene.particles(pos, radius=0.0025, per_vertex_color=color) # было (0.0, 0.8, 1.0)
    
    canvas.scene(scene)
    window.show()
