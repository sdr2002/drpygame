import math
from enum import Enum
from typing import Tuple, Optional

import numpy as np
import pygame
from numpy import ndarray
from pygame import Surface

from drpygames.lqr.solve import LqrTrajectory, solve_lqr_lti, LqrDesignLti
from drpygames.programs.utils.rgb import WHITE

printer = print


class RenderingMode(Enum):
    PHYSICAL = "natural"
    PLANNED = "planned"


def setup_pendulum_lqr_lti(
        n_steps: int = 1000, dt: float = 0.4,
        ang_s0: float = math.pi / 2, ang_v0: float = 0.,
        damping: float = 0.01, gravity: float = 0.01,
        rho: float = 0.3,
) -> LqrDesignLti:
    """Pendulum Linear dynamics without gravity"""
    x_0: ndarray[float] = np.array([[ang_s0, ang_v0]], dtype=float).T  # [angle, d(angle)/dt]

    # I just put gravity and air resistance as unit-less ratios. Couldn't tame up physics quick.
    A: ndarray[float] = np.array([
        [1., dt],
        [dt * gravity * (math.sin(ang_s0) / (ang_s0 + 1e-12)), 1. - (dt * damping)]
    ], dtype=float)
    B: ndarray[float] = np.array([
        [0., 0.],
        [0., dt]
    ], dtype=float)

    Q: ndarray[float] = 1. * np.eye(2, 2, dtype=float)
    R: ndarray[float] = rho * np.eye(2, 2, dtype=float)
    S_T: ndarray[float] = np.max([rho, 1.]) * np.diag([10., 1.])

    return LqrDesignLti(
        task='Pendulum',
        x_0=x_0, A=A, B=B, Q=Q, R=R, S_T=S_T, n_steps=n_steps,
        hyperparameters=dict(dt=dt, damping=damping, gravity=gravity, rho=rho)
    )


class Pendulum:
    def __init__(
            self, rendering_mode=RenderingMode.PHYSICAL,
            pivot_x=0., pivot_y=0., ball_m=0.01, pole_l=200., ball_color='red',
            ang_s0=math.pi / 2., ang_v0=0.,
            gravity: float = 0.01, damping: float = 0.01, dt: float = 0.4,
            max_iter: int = int(1e4),
            control: bool = False, n_steps: int = int(1e3),
            ang_sf: Optional[float] = None, ang_vf: Optional[float] = None,
            rho: Optional[float] = 10.
    ):

        """ Environment Spec """
        self.dt: float = dt
        self.max_iter: int = max_iter

        self.damping = damping
        self.gravity = gravity
        self.ball_m = ball_m
        self.pole_l = pole_l
        # Graphical Objects Spec
        self.pivot: Tuple[float, float] = (pivot_x, pivot_y)
        self.ball_x: float = 200.
        self.ball_y: float = 400.
        self.ball_color = ball_color

        """ Setup dynamics - naive """
        self.rendering_mode: RenderingMode = rendering_mode  # if you use traj planner like lqr, turn this on

        # Initial and terminal condition
        self.ang_s = ang_s0  # state of the dynamics
        self.ang_v = ang_v0

        self.ang_sf = ang_sf
        self.ang_vf = ang_vf

        # Planning
        self.planned: bool = control
        if not self.planned:
            err_msg_rendering_mode: str = f"Unplanned run must have RenderingMode.NATURAL, but {self.rendering_mode}"
            assert self.rendering_mode == RenderingMode.PHYSICAL, err_msg_rendering_mode

        if self.planned:
            err_msg_terminal_condition: str = "Terminal state must be specified for planning"
            assert not (ang_sf is None) and not (ang_vf is None), err_msg_terminal_condition

            self.lqr_design: LqrDesignLti = setup_pendulum_lqr_lti(
                n_steps=n_steps,
                ang_s0=ang_s0, ang_v0=ang_v0,
                dt=dt, rho=rho,
                gravity=gravity, damping=damping
            )
            self.trajectory: LqrTrajectory = solve_lqr_lti(self.lqr_design)

        self.iter: int = 0

    def step(self):
        """Run dynamics and update the state

        torque_net := ml^2 * dd(angle) = l * (-)(l*mg*sin(angle)) + torque_external(angle)
        """

        # Dynamics - Native
        if self.rendering_mode == RenderingMode.PHYSICAL:
            new_ang_a: float
            if self.planned:
                raise NotImplementedError("Physical rendering mode is not yet supported for planned traj")
            new_ang_a = (self.gravity * math.sin(self.ang_s)) - (self.damping * self.ang_v)
            new_ang_v: float = self.ang_v + self.dt * new_ang_a
            new_ang_s: float = self.ang_s + self.dt * self.ang_v
            self.ang_v = new_ang_v
            self.ang_s = new_ang_s
        elif self.rendering_mode == RenderingMode.PLANNED:
            # Dynamics from planning
            self.ang_s, self.ang_v = self.trajectory.x_t_list[self.iter][:, 0].tolist()
        else:
            raise ValueError("Rendering must be natural or planned")

        self.ball_x = self.pivot[0] + self.pole_l * math.sin(self.ang_s)
        self.ball_y = self.pivot[1] - self.pole_l * math.cos(self.ang_s)

        # Count iter
        self.iter += 1

    def draw(self, surface: Surface):
        pos: Tuple[float, float] = (self.ball_x, self.ball_y)
        pygame.draw.line(surface, WHITE, self.pivot, pos)
        pygame.draw.circle(surface, self.ball_color, pos, radius=15)

    def is_done(self) -> bool:
        if self.iter >= self.max_iter:
            printer("Maximum iteration reached")
            return True

        if self.planned:
            converged_sf: bool = math.fabs(self.ang_sf - self.ang_s) < math.pi * 1e-6
            converged_vf: bool = math.fabs(self.ang_vf - self.ang_v) < math.pi * 1e-6
            if converged_sf and converged_vf:
                printer(f"State converged to the goal state at iter {self.iter:.2g}")
                return True
            elif self.iter >= self.lqr_design.n_steps:
                printer("Planned iteration exhausted")
            else:
                return False
        return False

    def get_trajectory(self) -> LqrTrajectory:
        return self.trajectory

    def terminate(self):
        pass
