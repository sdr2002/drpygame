import math
from typing import Tuple, List

import pygame
from pygame import Surface
from pygame.time import Clock

from drpygames.lqr.solve import plot_lqr_trajectory
from drpygames.programs.simple_pendulum.pendulum import Pendulum
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

    fps: float = 100
    dt: float = 0.1
    assert dt < 0.5, "dt is foo coarse, set below 0.5"

    surface, clock = init_surface((width, height), 'Simple Pendulum')

    num_pendulum: int = 1

    # control_mode = {'planned': False, 'ang_s0': math.pi * 0.3}
    # control_mode = {'planned': True, 'iterative': False, 'ang_s0': math.pi*0.025}
    control_mode = {'planned': True, 'iterative': True, 'ang_s0': math.pi * 0.9}

    pendulums: List[Pendulum] = [Pendulum(
        planned=control_mode['planned'], iterative=control_mode.get('iterative', False),
        ang_s0=control_mode['ang_s0'],
        tau=dt, n_steps=int(600), rho=1000,
        damping=5e-3, gravity=0.01,
        ball_m=100, pole_l=(200+100*i),
        pivot_x=width//2, pivot_y=height//2, ball_color=c)
        for i, c in enumerate(COLORS[:num_pendulum])
    ]

    stop: bool = False
    while not stop:
        clock.tick(fps)
        surface.fill(BLACK)

        for event in pygame.event.get():
            stop = event.type == pygame.QUIT

        for pobj in pendulums:
            stop = pobj.is_done()
            if stop:
                continue

            pobj.step()
            pobj.draw(surface)

            if pobj.planned and pobj.iterative:
                if pobj.iter % int(1./dt) == 0:
                    # iter every 1 sec, not justifiable but am tired
                    pobj.update_dynamics()

        pygame.display.flip()
    pygame.quit()

    if control_mode.get('planned', False):
        plot_lqr_trajectory(pendulums[0].get_trajectory(), pendulums[0].lqr_design)


if __name__ == "__main__":
    run()
