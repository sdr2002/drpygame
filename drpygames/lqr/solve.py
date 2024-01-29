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


def transition(A: ndarray, x_t: ndarray, B: ndarray, u_t: ndarray) -> ndarray:
    """Get the next state x_tpp from x_t and u_t with linear dynamics"""
    return A @ x_t + B @ u_t


def innerproduct(state: ndarray, measure: ndarray) -> ndarray:
    """Do <state|measure|state> matrix operation"""
    return state.T @ measure @ state


def brek(B: ndarray, R: ndarray, S_tpp: ndarray) -> ndarray:
    """Do R + <B|S_tpp|B>"""
    return R + innerproduct(B, S_tpp)


def inv_brek(B: ndarray, R: ndarray, S_tpp: ndarray) -> ndarray:
    """Do (R + <B|S_tpp|B>)^-1"""
    return inv(brek(B, R, S_tpp))


def kalmangain(A: ndarray, B: ndarray, R: ndarray, S_tpp: ndarray) -> ndarray:
    """Get Kalman Gain matrix at time t from t+1 Solution matrix and LQR parameters"""
    return -inv_brek(B, R, S_tpp) @ B.T @ S_tpp @ A


def rewind_solution(A: ndarray, B: ndarray, Q: ndarray, R: ndarray, S_tpp: ndarray) -> ndarray:
    """Get Solution matrix at time t from t+1 Solution matrix and LQR parameters"""
    term2: ndarray = innerproduct(A, S_tpp)

    term3_1: ndarray = inv_brek(B, S_tpp, R)
    term3_2: ndarray = innerproduct(B, term3_1)
    term3_3: ndarray = innerproduct(S_tpp, term3_2)
    term3: ndarray = innerproduct(A, term3_3)

    return Q + term2 - term3


@dataclass
class LqrDesign:
    task: str

    x_0: ndarray

    Q: ndarray
    R: ndarray
    S_T: ndarray
    n_steps: int

    hyperparameters: Optional[Dict[str, Any]]

    def check_dimensionality(self):
        raise NotImplementedError


@dataclass
class LqrDesignLti(LqrDesign):
    A: ndarray
    B: ndarray

@dataclass
class LqrDesignLtv(LqrDesign):
    A_list: List[ndarray]
    B_list: List[ndarray]


@dataclass
class LqrTrajectory:
    x_t_list: List[ndarray]
    u_t_list: List[ndarray]

    def copy_til_i(self, ind_i: int):
        x_t_list: List[ndarray] = [x_t for x_t in self.x_t_list[:ind_i+1]]
        u_t_list: List[ndarray] = [u_t for u_t in self.u_t_list[:ind_i+1]]
        return LqrTrajectory(x_t_list, u_t_list)

    def check_dimentionality(self):
        assert len(self.x_t_list) == len(self.u_t_list), "len({x}) must be len({u})+1"

        raise NotImplementedError


def solve_lqr_lti(lqrdesign: LqrDesignLti) -> LqrTrajectory:
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
    x_0: ndarray = lqrdesign.x_0
    A: ndarray = lqrdesign.A
    B: ndarray = lqrdesign.B

    Q: ndarray = lqrdesign.Q
    R: ndarray = lqrdesign.R
    S_T: ndarray = lqrdesign.S_T
    n_steps: int = lqrdesign.n_steps

    # Run
    S_tpp_list: List[ndarray] = [S_T]
    for t in range(n_steps, 0, -1):
        S_t: ndarray = rewind_solution(A, B, Q, R, S_tpp_list[-1])
        S_tpp_list.append(S_t)
    S_tpp_list = S_tpp_list[::-1]

    K_t_list: List[ndarray] = []
    for t in range(0, n_steps, 1):
        K_t: ndarray = kalmangain(A, B, R, S_tpp_list[t + 1])
        K_t_list.append(K_t)

    x_t_list: List[ndarray] = [x_0]
    u_t_list: List[ndarray] = []
    for t in range(0, n_steps, 1):
        u_t: ndarray = K_t_list[t] @ x_t_list[t]
        u_t_list.append(u_t)

        if t < n_steps:
            x_tpp: ndarray = transition(A, x_t_list[t], B, u_t)
            x_t_list.append(x_tpp)

    plan_result: LqrTrajectory = LqrTrajectory(x_t_list=x_t_list, u_t_list=u_t_list)
    return plan_result


def solve_lqr_ltv(lqrdesign: LqrDesignLtv) -> LqrTrajectory:
    """Indeed, most of dynamics we deal with are with time-variant linear maps"""
    x_0: ndarray = lqrdesign.x_0
    A_list: List[ndarray] = lqrdesign.A_list
    B_list: List[ndarray] = lqrdesign.B_list

    Q: ndarray = lqrdesign.Q
    R: ndarray = lqrdesign.R
    S_T: ndarray = lqrdesign.S_T
    n_steps: int = lqrdesign.n_steps

    # Run
    S_tpp_list: List[ndarray] = [S_T]
    for t in range(n_steps, 0, -1):
        S_t: ndarray = rewind_solution(A_list[t], B_list[t], Q, R, S_tpp_list[-1])
        S_tpp_list.append(S_t)
    S_tpp_list = S_tpp_list[::-1]

    K_t_list: List[ndarray] = []
    for t in range(0, n_steps, 1):
        K_t: ndarray = kalmangain(A_list[t], B_list[t], R, S_tpp_list[t + 1])
        K_t_list.append(K_t)

    x_t_list: List[ndarray] = [x_0]
    u_t_list: List[ndarray] = []
    for t in range(0, n_steps, 1):
        u_t: ndarray = K_t_list[t] @ x_t_list[t]
        u_t_list.append(u_t)

        if t < n_steps:
            x_tpp: ndarray = transition(A_list[t], x_t_list[t], B_list[t], u_t)
            x_t_list.append(x_tpp)

    plan_result: LqrTrajectory = LqrTrajectory(x_t_list=x_t_list, u_t_list=u_t_list)
    return plan_result


def plot_lqr_trajectory(trajectory: LqrTrajectory, lqrdesign: LqrDesign):
    x_dim: int = trajectory.x_t_list[0].shape[0]
    x_arr_dict: Dict[int, ndarray] = {}
    for ind_x in range(x_dim):
        x_arr_dict[ind_x] = np.asarray([x_t[ind_x, 0] for x_t in trajectory.x_t_list])

    u_dim: int = trajectory.u_t_list[0].shape[0]
    u_arr_dict: Dict[int, ndarray] = {}
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
    x_0: ndarray = np.array([[1, 0]], dtype=float).T
    n_steps: int = 20
    # x_T: ndarray = np.array([[0, 0]], dtype=float).T # final state is indirectly represented by S_T

    A_Rair: ndarray = np.array([[1, 1], [0, 0.7]], dtype=float)
    # A_R0: ndarray = np.array([[1, 1], [0, 1]], dtype=float)
    A: ndarray = A_Rair  # A_R0
    B: ndarray = np.array([[0, 0], [0, 1]], dtype=float)

    Q: ndarray = 1 * np.eye(2, 2, dtype=float)
    R: ndarray = rho * np.eye(2, 2, dtype=float)
    S_T: ndarray = 1 * np.eye(2, 2, dtype=float)

    return LqrDesignLti(
        task='Example1', x_0=x_0, A=A, B=B, Q=Q, R=R, S_T=S_T, n_steps=n_steps,
        hyperparameters={'rho': rho}
    )


if __name__ == "__main__":
    lqrdesign: LqrDesignLti = setup_example1(rho=10.)
    trajectory: LqrTrajectory = solve_lqr_lti(lqrdesign)
    plot_lqr_trajectory(trajectory, lqrdesign)
