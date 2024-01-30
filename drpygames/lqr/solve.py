"""
https://stanford.edu/class/ee363/lectures/dlqr.pdf
Ch1.Page24
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import numpy as np
from numpy import ndarray
from numpy.linalg import inv
from plotly import graph_objects as go
from plotly.subplots import make_subplots

ndarrflt = ndarray[float]


def transition(A: ndarrflt, x_t: ndarrflt, B: ndarrflt, u_t: ndarrflt) -> ndarrflt:
    """Get the next state x_tpp from x_t and u_t with linear dynamics"""
    return A @ x_t + B @ u_t


def innerproduct(state: ndarrflt, measure: ndarrflt) -> ndarrflt:
    """Do <state|measure|state> matrix operation"""
    return state.T @ measure @ state


def brek(B: ndarrflt, R: ndarrflt, S_tpp: ndarrflt) -> ndarrflt:
    """Do R + <B|S_tpp|B>"""
    return R + innerproduct(B, S_tpp)


def inv_brek(B: ndarrflt, R: ndarrflt, S_tpp: ndarrflt) -> ndarrflt:
    """Do (R + <B|S_tpp|B>)^-1"""
    return inv(brek(B, R, S_tpp))


def kalmangain(A: ndarrflt, B: ndarrflt, R: ndarrflt, S_tpp: ndarrflt) -> ndarrflt:
    """Get Kalman Gain matrix at time t from t+1 Solution matrix and LQR parameters"""
    return -inv_brek(B, R, S_tpp) @ B.T @ S_tpp @ A


def rewind_solution(A: ndarrflt, B: ndarrflt, Q: ndarrflt, R: ndarrflt, S_tpp: ndarrflt) -> ndarrflt:
    """Get Solution matrix at time t from t+1 Solution matrix and LQR parameters"""
    term2: ndarrflt = innerproduct(A, S_tpp)

    term3_1: ndarrflt = inv_brek(B, S_tpp, R)
    term3_2: ndarrflt = innerproduct(B, term3_1)
    term3_3: ndarrflt = innerproduct(S_tpp, term3_2)
    term3: ndarrflt = innerproduct(A, term3_3)

    return Q + term2 - term3


@dataclass
class LqrDesign:
    task: str

    x_0: ndarrflt

    S_T: ndarrflt
    n_steps: int

    hyperparameters: Optional[Dict[str, Any]]

    def check_dimensionality(self):
        raise NotImplementedError


@dataclass
class LqrDesignLti(LqrDesign):
    A: ndarrflt
    B: ndarrflt
    Q: ndarrflt
    R: ndarrflt


@dataclass
class LqrDesignLtv(LqrDesign):
    A_list: List[ndarrflt]
    B_list: List[ndarrflt]
    Q_list: List[ndarrflt]
    R_list: List[ndarrflt]


@dataclass
class LqrTrajectory:
    x_t_list: List[ndarrflt]
    u_t_list: List[ndarrflt]

    def copy_til_i(self, ind_i: int):
        x_t_list: List[ndarrflt] = [x_t for x_t in self.x_t_list[:ind_i + 1]]
        u_t_list: List[ndarrflt] = [u_t for u_t in self.u_t_list[:ind_i + 1]]
        return LqrTrajectory(x_t_list, u_t_list)

    def check_dimentionality(self):
        assert len(self.x_t_list) == len(self.u_t_list), "len({x}) must be len({u})+1"

        raise NotImplementedError


def solve_lqr_lti(lqr_design: LqrDesignLti) -> LqrTrajectory:
    """Do this EXACTLY: this literally writes down what's described in Ch1.p23 
    
    This is for noob's meaning that if you are a beginner like me, you absolutely want to do in this way.
    If any of the matrix is exploding or np inv op is failing (not pinv), you designed matrices wrong.
    Routine is not computationally efficient, but you can see things block by block with this.
    
    Summary of LQR solution via DP
    1. set PN := Qf
    2. for t = N, . . . , 1, Pt−1 := Q + AT PtA − AT PtB(R + BT PtB)−1BT PtA
    3. for t = 0, . . . , N − 1, define Kt := −(R + BT Pt+1B)−1BT Pt+1A
    4. for t = 0, . . . , N − 1, optimal u is given by ulqrt = Ktxt
    • optimal u is a linear function of the state (called linear state feedback)
    • recursion for min cost-to-go runs backward in time
    """
    x_0: ndarrflt = lqr_design.x_0
    A: ndarrflt = lqr_design.A
    B: ndarrflt = lqr_design.B

    Q: ndarrflt = lqr_design.Q
    R: ndarrflt = lqr_design.R
    S_T: ndarrflt = lqr_design.S_T
    n_steps: int = lqr_design.n_steps

    # Run
    S_tpp_list: List[ndarrflt] = [S_T]
    for t in range(n_steps, 0, -1):
        S_t: ndarrflt = rewind_solution(A, B, Q, R, S_tpp_list[-1])
        S_tpp_list.append(S_t)
    S_tpp_list = S_tpp_list[::-1]

    K_t_list: List[ndarrflt] = []
    for t in range(0, n_steps, 1):
        K_t: ndarrflt = kalmangain(A, B, R, S_tpp_list[t + 1])
        K_t_list.append(K_t)

    x_t_list: List[ndarrflt] = [x_0]
    u_t_list: List[ndarrflt] = []
    for t in range(0, n_steps, 1):
        u_t: ndarrflt = K_t_list[t] @ x_t_list[t]
        u_t_list.append(u_t)

        if t < n_steps:
            x_tpp: ndarrflt = transition(A, x_t_list[t], B, u_t)
            x_t_list.append(x_tpp)

    plan_result: LqrTrajectory = LqrTrajectory(x_t_list=x_t_list, u_t_list=u_t_list)
    return plan_result


def solve_lqr_ltv(lqr_design: LqrDesignLtv) -> LqrTrajectory:
    """Time-varying linear dynamics solver by Kalman gain
    """
    x_0: ndarrflt = lqr_design.x_0
    A_list: List[ndarrflt] = lqr_design.A_list
    B_list: List[ndarrflt] = lqr_design.B_list

    Q_list: List[ndarrflt] = lqr_design.Q_list
    R_list: List[ndarrflt] = lqr_design.R_list
    S_T: ndarrflt = lqr_design.S_T
    n_steps: int = lqr_design.n_steps

    # Run
    S_tpp_list: List[ndarrflt] = [S_T]
    for t in range(n_steps, 0, -1):
        S_t: ndarrflt = rewind_solution(A_list[t], B_list[t], Q_list[t], R_list[t], S_tpp_list[-1])
        S_tpp_list.append(S_t)
    S_tpp_list = S_tpp_list[::-1]

    K_t_list: List[ndarrflt] = []
    for t in range(0, n_steps, 1):
        K_t: ndarrflt = kalmangain(A_list[t], B_list[t], R_list[t], S_tpp_list[t + 1])
        K_t_list.append(K_t)

    x_t_list: List[ndarrflt] = [x_0]
    u_t_list: List[ndarrflt] = []
    for t in range(0, n_steps, 1):
        u_t: ndarrflt = K_t_list[t] @ x_t_list[t]
        u_t_list.append(u_t)

        # if t < n_steps:
        x_tpp: ndarrflt = transition(A_list[t], x_t_list[t], B_list[t], u_t)
        x_t_list.append(x_tpp)

    plan_result: LqrTrajectory = LqrTrajectory(x_t_list=x_t_list, u_t_list=u_t_list)
    return plan_result


def plot_lqr_trajectory(trajectory: LqrTrajectory, lqrdesign: LqrDesign):
    x_dim: int = trajectory.x_t_list[0].shape[0]
    x_arr_dict: Dict[int, ndarrflt] = {}
    for ind_x in range(x_dim):
        x_arr_dict[ind_x] = np.asarray([x_t[ind_x, 0] for x_t in trajectory.x_t_list])

    u_dim: int = trajectory.u_t_list[0].shape[0]
    u_arr_dict: Dict[int, ndarrflt] = {}
    for ind_u in range(u_dim):
        u_arr_dict[ind_u] = np.asarray([u_t[ind_u, 0] for u_t in trajectory.u_t_list])

    fig = make_subplots(rows=1, cols=1, x_title="Timestep", specs=[[{'secondary_y': True}]])
    x_axis = np.arange(len(trajectory.x_t_list))
    for ind_x, x_arr in x_arr_dict.items():
        fig.add_trace(go.Scatter(x=x_axis, y=x_arr, name=f'x{ind_x}'), row=1, col=1, secondary_y=False)
    for ind_u, u_arr in u_arr_dict.items():
        fig.add_trace(go.Scatter(x=x_axis[:-1], y=u_arr, name=f'u{ind_u}'), row=1, col=1, secondary_y=True)

    fig.update_layout(
        title=f'State-Control trajectory for {lqrdesign.task} <br>'
              f'- {" / ".join([f"{k}:{v}" for k, v in lqrdesign.hyperparameters.items()])}',
        width=1200, height=750
    )
    fig.update_yaxes(title_text="State", secondary_y=False)
    fig.update_yaxes(title_text="Control", secondary_y=True)

    fig.show()


def setup_example1(rho: float = 0.3) -> LqrDesignLti:
    """Toy example of Defining dynamics and control objective parameters"""
    x_0: ndarrflt = np.array([[1, 0]], dtype=float).T
    n_steps: int = 20
    # x_T: ndfloat = np.array([[0, 0]], dtype=float).T # final state is indirectly represented by S_T

    A_Rair: ndarrflt = np.array([[1, 1], [0, 0.7]], dtype=float)
    # A_R0: ndfloat = np.array([[1, 1], [0, 1]], dtype=float)
    A: ndarrflt = A_Rair  # A_R0
    B: ndarrflt = np.array([[0, 0], [0, 1]], dtype=float)

    Q: ndarrflt = 1 * np.eye(2, 2, dtype=float)
    R: ndarrflt = rho * np.eye(2, 2, dtype=float)
    S_T: ndarrflt = 1 * np.eye(2, 2, dtype=float)

    return LqrDesignLti(
        task='Example1', x_0=x_0, A=A, B=B, Q=Q, R=R, S_T=S_T, n_steps=n_steps,
        hyperparameters={'rho': rho}
    )


def setup_example2_1(risk_aversion: float = 1e-7) -> LqrDesignLti:
    """
    state: [inventory, price]
    control: [slice, None]
    """
    assert risk_aversion >= 0., "risk_aversion must be non-negative."
    # but you can run and see what happens there ;)

    x_0: ndarray[float] = np.array([[1e5, 0.]]).T
    n_steps: int = 100

    A: ndarray[float] = np.array([
        [1., 0.],
        [0., 1.]
    ], dtype=float)
    B: ndarray[float] = np.array([
        [-1., 0.],
        [2.5 * 1e-7, 0.]
    ], dtype=float)

    ignore: float = 1e-12
    Q: ndarray[float] = risk_aversion * np.diag([1., ignore])
    R: ndarray[float] = np.diag([2.5 * 1e-6, ignore])
    S_T: ndarray[float] = np.diag([np.max([risk_aversion, 1e-7]) * 1e2, ignore])

    return LqrDesignLti(
        task='Example2_lti', x_0=x_0, A=A, B=B,
        Q=Q, R=R, S_T=S_T, n_steps=n_steps,
        hyperparameters={'risk_aversion': risk_aversion}
    )


def setup_example2_2(risk_aversion: float = 1e-7) -> LqrDesignLtv:
    """
    state: [inventory, price]
    control: [slice, n/a]
    """
    assert risk_aversion >= 0., "risk_aversion must be non-negative."

    x_0: ndarray[float] = np.array([[1e5, 0.]]).T
    n_steps: int = 100

    A: ndarray[float] = np.array([
        [1., 0.],
        [0., 1.]
    ], dtype=float)
    A_list: List[ndarray[float]] = [A for _ in range(n_steps + 1)]
    v_profile = np.linspace(-1, 1, n_steps + 1) ** 2. + 0.5
    v_profile /= np.mean(v_profile)
    B_list: List[ndarray[float]] = [np.array([
        [-1., 0.],
        [(2.5 * 1e-7) * v_profile[t], 0.]
    ], dtype=float) for t in range(n_steps + 1)]

    ignore: float = 1e-12
    Q: ndarray[float] = risk_aversion * np.diag([1., ignore])
    Q_list: List[ndarray[float]] = [Q for _ in range(n_steps + 1)]
    R_list: List[ndarray[float]] = [np.diag([(2.5 * 1e-6) * v_profile[t], ignore]) for t in range(n_steps + 1)]
    S_T: ndarray[float] = np.diag([np.max([risk_aversion, 1e-7]) * 1e2, ignore])

    return LqrDesignLtv(
        task='Example2_ltv', x_0=x_0, A_list=A_list, B_list=B_list,
        Q_list=Q_list, R_list=R_list, S_T=S_T, n_steps=n_steps,
        hyperparameters={'risk_aversion': risk_aversion}
    )


if __name__ == "__main__":
    # lqr_design: LqrDesignLti = setup_example1(rho=10.)
    lqr_design: LqrDesignLti = setup_example2_1(risk_aversion=1e-8)
    trajectory: LqrTrajectory = solve_lqr_lti(lqr_design)

    plot_lqr_trajectory(trajectory, lqr_design)

    # lqr_design: LqrDesignLtv = setup_example2_2(risk_aversion=1e-8)
    # trajectory: LqrTrajectory = solve_lqr_ltv(lqr_design)
    #
    # plot_lqr_trajectory(trajectory, lqr_design)
