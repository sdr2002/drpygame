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

    dt: float = 0.1
    fps: float = 10. * 1./dt
    assert dt < 0.5, "dt is foo coarse, set below 0.5"

    surface, clock = init_surface((width, height), 'Simple Pendulum')

    num_pendulum: int = 1

    # control_mode = {'planned': False, 'ang_s0': math.pi * 0.3}
    # control_mode = {'planned': True, 'reeval': False, 'ang_s0': math.pi*0.025, 'n_steps': 300, 'rho': 10.}
    control_mode = {'planned': True, 'reeval': True, 'ang_s0': math.pi * 0.9, 'n_steps': 400, 'rho': 1e2}

    pendulums: List[Pendulum] = [Pendulum(
        planned=control_mode['planned'], reeval=control_mode.get('reeval', False),
        ang_s0=control_mode['ang_s0'],
        tau=dt, n_steps=control_mode.get('n_steps', None), rho=control_mode.get('rho', None),
        damping=7e-3, gravity=0.09,
        ball_m=100, pole_l=(200+100*i),
        pivot_x=width//2, pivot_y=height//2, ball_color=c)
        for i, c in enumerate(COLORS[:num_pendulum])
    ]

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

            if pobj.planned and pobj.reeval:
                # if (pobj.iter % 10 == 0) and (math.fabs(pobj.ang_s) >= math.pi * 0.02):
                if pobj.check_to_reevaluate() and (math.fabs(pobj.ang_s) >= math.pi * 2e-2):
                    print(f'LQR reset triggered at i={ind}')
                    # For every 10N+1st iter while the angle is not small enough
                    pobj.reevaluate()

        pygame.display.flip()
        ind += 1
    pygame.quit()

    if control_mode.get('planned', False):
        plot_lqr_trajectory(pendulums[0].get_trajectory(), pendulums[0].lqr_design)


if __name__ == "__main__":
    run()
