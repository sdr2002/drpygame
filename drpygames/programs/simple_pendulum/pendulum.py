import math
from typing import Tuple, List

import numpy as np
import pygame
from numpy import ndarray
from pygame import Surface

from drpygames.lqr.solve import LqrTrajectory, solve_lqr_lti, LqrDesignLti
from drpygames.programs.utils.rgb import WHITE


def setup_pendulum_lqr_lti(
        n_steps: int = 10000, reeval: bool = False,
        ang_s0: float = math.pi / 2, ang_v0: float = 0., dt: float = 0.4,
        damping: float = 0.01, rho: float = 0.3, gravity: float = 0.01,
) -> LqrDesignLti:
    """Pendulum Linear dynamics without gravity"""
    x_0: ndarray = np.array([[ang_s0, ang_v0]], dtype=float).T  # [angle, d(angle)/dt]

    # I just put gravity and air resistance as unit-less ratios. Couldn't tame up physics quick.
    A: ndarray = np.array([
        [1, dt],
        [dt*gravity*(math.sin(ang_s0)/(ang_s0+1e-9)), 1.-(dt*damping)]
    ], dtype=float)
    B: ndarray = np.array([
        [0., 0.],
        [0., dt]
    ], dtype=float)

    Q: ndarray = 1 * np.eye(2, 2, dtype=float)
    R: ndarray = rho * np.eye(2, 2, dtype=float)
    S_T: ndarray = np.max([rho, 1.]) * np.diag([10., 1.])

    return LqrDesignLti(
        task='Pendulum' if not reeval else 'Pendulum_reLQR',
        x_0=x_0, A=A, B=B, Q=Q, R=R, S_T=S_T, n_steps=n_steps,
        hyperparameters=dict(dt=dt, damping=damping, gravity=gravity, rho=rho)
    )


class Pendulum:
    def __init__(
            self, pivot_x=0., pivot_y=0., ball_m=0.01, pole_l=200.,
            ang_s0=math.pi / 2., ang_v0=0.,
            gravity: float = 0.01, damping: float = 0.01, tau: float = 0.4,
            n_steps: int = int(1e4), rho: float = 10.,
            planned: bool = False, reeval: bool = False,
            ball_color='red'
    ):
        # Graphical Objects Spec
        self.pivot: Tuple[float, float] = (pivot_x, pivot_y)

        self.ball_x: float = 200.
        self.ball_y: float = 400.

        # Setup dynamics T1 - naive
        self.planned: bool = False  # if you use traj planner like lqr, turn this on

        # self.ball_m = ball_m
        self.pole_l = pole_l

        self.ang_s = ang_s0  # state of the dynamics
        self.ang_v = ang_v0

        self.damping = damping
        self.gravity = gravity

        self.ball_color = ball_color

        self.dt: float = tau

        # Setup dynamics T2 - LQR applicable
        self.planned: bool = planned
        if self.planned:
            self.reeval = reeval

            self.lqr_design: LqrDesignLti = setup_pendulum_lqr_lti(
                n_steps=n_steps, reeval=self.reeval,
                ang_s0=ang_s0, ang_v0=ang_v0,
                dt=tau, rho=rho,
                gravity=gravity, damping=damping
            )
            self.trajectory: LqrTrajectory = solve_lqr_lti(self.lqr_design)

            if self.reeval:
                self.trajectory_list: List[LqrTrajectory] = []

        self.iter: int = 0

    def check_to_reevaluate(self):
        angle_0: float = self.lqr_design.x_0[0, 0]
        new_angle_0: float = self.trajectory.x_t_list[self.iter][0, 0]

        return math.fabs(new_angle_0 - angle_0) >= math.pi * 1e-2

    def reevaluate(self):
        assert self.reeval, "This is reevaluating LQR mode and is not activated"

        new_ang_s0, new_ang_v0 = self.trajectory.x_t_list[self.iter][:, 0]
        self.lqr_design = setup_pendulum_lqr_lti(
            n_steps=self.lqr_design.n_steps - self.iter,
            reeval=self.reeval,
            ang_s0=new_ang_s0, ang_v0=new_ang_v0,
            dt=self.dt, rho=self.lqr_design.hyperparameters['rho'],
            gravity=self.gravity, damping=self.damping
        )

        self.trajectory_list.append(self.trajectory.copy_til_i(self.iter - 1))
        self.trajectory: LqrTrajectory = solve_lqr_lti(self.lqr_design)

        self.iter = 0

    def step(self):
        """Run dynamics and update the state

        torque_net := ml^2 * dd(angle) = l * (-)(l*mg*sin(angle)) + torque_external(angle)
        """

        # Dynamics - Native
        if not self.planned:
            # ang_a = self.trajectory.u_t_list[self.iter][1, 0]
            # self.ang_s += self.ang_v * self.dt
            new_ang_v: float = self.ang_v + self.dt*self.gravity*math.sin(self.ang_s) - (self.dt * self.damping * self.ang_v)
            new_ang_s: float = self.ang_s + self.ang_v * self.dt
            self.ang_v = new_ang_v
            self.ang_s = new_ang_s

        # Dynamics - LQR linear model
        if self.planned:
            self.ang_s = self.trajectory.x_t_list[self.iter][0, 0]

        self.ball_x = self.pivot[0] + self.pole_l * math.sin(self.ang_s)
        self.ball_y = self.pivot[1] - self.pole_l * math.cos(self.ang_s)

        # Count iter
        self.iter += 1

    def draw(self, surface: Surface):
        pos: Tuple[float, float] = (self.ball_x, self.ball_y)
        pygame.draw.line(surface, WHITE, self.pivot, pos)
        pygame.draw.circle(surface, self.ball_color, pos, radius=15)

    def is_done(self) -> bool:
        if self.planned:
            return self.iter >= self.lqr_design.n_steps
        else:
            return False

    def get_trajectory(self) -> LqrTrajectory:
        if self.reeval:
            # flatten and return
            x_t_list: List[ndarray] = [x_t for traj in self.trajectory_list for x_t in traj.x_t_list]
            u_t_list: List[ndarray] = [u_t for traj in self.trajectory_list for u_t in traj.u_t_list]
            return LqrTrajectory(x_t_list, u_t_list)
        else:
            return self.trajectory

    def terminate(self):
        if self.reeval:
            self.reevaluate()
