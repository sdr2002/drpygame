import math
from typing import Tuple, List, Any, Dict

import pygame
from pygame import Surface
from pygame.time import Clock

from drpygames.lqr.solve import plot_lqr_trajectory
from drpygames.programs.simple_pendulum.pendulum import Pendulum, RenderingMode
from drpygames.programs.utils.rgb import BLACK, COLORS

""" PYGAME animation implementation of simple pendulum

Simple Pendulum Tutorial | Python  https://www.youtube.com/watch?v=9C_i8jERrL4
"""


def init_surface(size: Tuple[float, float], title: str) -> Tuple[Surface, Clock]:
    pygame.init()
    pygame.display.set_caption(title)
    surface: Surface = pygame.display.set_mode(size)
    clock: Clock = pygame.time.Clock()

    return surface, clock


def run():
    width: float = 800
    height: float = 800

    dt: float = 0.1
    fps: float = 10. * 1. / dt
    assert dt < 0.5, "dt is foo coarse, set below 0.5"

    surface, clock = init_surface((width, height), 'Simple Pendulum')

    num_pendulum: int = 1

    control_mode1 = {'rendering_mode': RenderingMode.PHYSICAL, 'ang_s0': math.pi * 0.3, 'control': False}
    control_mode2 = {'rendering_mode': RenderingMode.PLANNED, 'ang_s0': math.pi * 0.025,
                     'control': True, 'ang_sf': 0., 'ang_vf': 0., 'n_steps': 300, 'rho': 10.}

    control_mode: Dict[str, Any] = control_mode1

    pendulums: List[Pendulum] = [Pendulum(
        rendering_mode=control_mode.get('rendering_mode', RenderingMode.PHYSICAL),
        ang_s0=control_mode['ang_s0'],
        dt=dt, max_iter=control_mode.get('n_steps', int(1e4)),
        damping=7e-3, gravity=0.09,
        ball_m=100, pole_l=(200 + 100 * i),
        pivot_x=width // 2, pivot_y=height // 2, ball_color=c,
        control=control_mode['control'],
        n_steps=control_mode.get('n_steps', int(1e3)),
        ang_sf=control_mode.get('ang_sf', None), ang_vf=control_mode.get('ang_vf', None),
        rho=control_mode.get('rho', None)
    ) for i, c in enumerate(COLORS[:num_pendulum])]

    stop: bool = False
    ind: int = 0
    while not stop:
        clock.tick(fps)
        surface.fill(BLACK)

        for event in pygame.event.get():
            stop = event.type == pygame.QUIT

        for pobj in pendulums:
            stop = pobj.is_done()
            if stop:
                pobj.terminate()
                continue

            pobj.step()
            pobj.draw(surface)

        pygame.display.flip()
        ind += 1
    pygame.quit()

    if control_mode.get('planned', False):
        plot_lqr_trajectory(pendulums[0].get_trajectory(), pendulums[0].lqr_design)


if __name__ == "__main__":
    run()
