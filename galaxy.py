import taichi as ti
import taichi.math as tm


ti.init(arch=ti.gpu)

# Количество частиц
N = 1_000_000

# Позиции частиц: 1D поле из 3D-векторов
pos = ti.Vector.field(3, dtype=ti.f32, shape=N)

@ti.kernel
def init_particles():
    for i in range(N):
        pos[i] = tm.vec3(
            ti.random() * 2 - 1,
            ti.random() * 2 - 1,
            ti.random() * 2 - 1
        )

init_particles()

window = ti.ui.Window("Galaxy", (1024, 768), vsync=True)
canvas = window.get_canvas() # поверхность для 2D-рисования.
scene = window.get_scene() # 3D-сцена
camera = ti.ui.Camera()

# Камера смотрит в начало координат
camera.position(0, 0, 3)
camera.lookat(0, 0, 0)
scene.set_camera(camera)

scene.point_light(pos=(5, 5, 5), color=(1, 1, 1))
scene.ambient_light((0.3, 0.3, 0.3))

while window.running:
    # Управление камерой (ПКМ)
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    
    # Рисуем частицы
    scene.particles(pos, radius=0.001, color=(0.9, 0.8, 0.6))
    
    canvas.scene(scene)
    window.show()
