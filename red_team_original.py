import numpy as np


class RedTeam():
    def __init__(self, terminal_time, s_b, s_c, s_d, var_z = 0.05, var_y = 0.05, alpha = 0.05, beta=0.05):
        """ s_b, s_c, s_d: misdirection pattern to test for, each of shape (terminal_time,) """

        # Dynamics info
        self.var_z = var_z
        self.var_y = var_y

        # Time horizon
        self.terminal_time = terminal_time

        # Deception info (copied, so the red team's knowledge is independent of the source arrays)
        self.s_b = np.array(s_b, dtype=float)
        self.s_c = np.array(s_c, dtype=float)
        self.s_d = np.array(s_d, dtype=float)

        # Probability of error
        self.alpha = alpha
        self.beta = beta
        # Testing parameters
        self.U = (1-self.beta)/self.alpha
        self.B = self.beta/(1-self.alpha)

    def calculate_L_T(self, observed_trajectory):
        """ Mean over trajectories of log L_T. observed_trajectory: (num_traj, 2, len_traj) """

        return np.mean(self.sequential_hypothesis_test(observed_trajectory)[:, -1])

    def sequential_hypothesis_test(self, observed_trajectory):
        """ log L_t for each trajectory. observed_trajectory: (num_traj, 2, len_traj) -> (num_traj, len_traj) """

        return self.sequential_log_likelihood_ratio(observed_trajectory)

    def sequential_log_likelihood_ratio(self, observed_trajectory):
        """
        Cumulative log-likelihood ratio log L_t of "misdirected" vs. "nominal" dynamics,
        computed with the red team's own knowledge (s_b, s_c, s_d, var_y).

        observed_trajectory: (num_traj, 2, len_traj), states ordered as (v, x)
        Returns: (num_traj, len_traj), with log L_0 = 0
        """
        observed_trajectory = np.asarray(observed_trajectory)
        len_traj = observed_trajectory.shape[2]

        v_prev = observed_trajectory[:, 0, :-1]
        x_prev = observed_trajectory[:, 1, :-1]
        x = observed_trajectory[:, 1, 1:]
        s_b_t = self.s_b[:len_traj-1]
        s_c_t = self.s_c[:len_traj-1]
        s_d_t = self.s_d[:len_traj-1]

        residual = x - v_prev - x_prev
        increments = (1/(2*self.var_y))*(residual**2 - (residual - s_b_t*v_prev - s_c_t*x_prev - s_d_t)**2)

        log_L_t = np.zeros(observed_trajectory.shape[::2])
        log_L_t[:, 1:] = np.cumsum(increments, axis=1)

        return log_L_t