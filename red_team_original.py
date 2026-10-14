import numpy as np


def as_matrix(value, n_dim):
    """ Scalar -> value*I_n, otherwise an (n_dim, n_dim) array """
    value = np.asarray(value, dtype=float)
    if value.ndim == 0:
        return value*np.eye(n_dim)
    assert value.shape == (n_dim, n_dim), f"expected a scalar or an ({n_dim}, {n_dim}) matrix, got shape {value.shape}"
    return value


def matrix_parameter(name):
    """ Attribute that accepts a scalar or an (n, n) matrix and is always stored as an (n, n) matrix """
    def getter(self): return getattr(self, "_" + name)
    def setter(self, value): setattr(self, "_" + name, as_matrix(value, self.n_dim))
    return property(getter, setter)


class RedTeam():

    var_z = matrix_parameter("var_z")
    var_y = matrix_parameter("var_y")

    def __init__(self, terminal_time, s_b, s_c, s_d, n_dim = 1, var_z = 0.05, var_y = 0.05, alpha = 0.05, beta=0.05):
        """
        s_b, s_c: misdirection pattern F^b_t, F^c_t to test for, each of shape (terminal_time, n, n)
                  (for n_dim = 1, (terminal_time,) sequences are also accepted)
        s_d: f^d_t, shape (terminal_time, n)
        var_z, var_y: scalar (meaning scalar*I_n) or (n, n) covariance matrices; can be changed after construction
        """

        self.n_dim = n_dim

        # Dynamics info
        self.var_z = var_z
        self.var_y = var_y

        # Time horizon
        self.terminal_time = terminal_time

        # Deception info (copied, so the red team's knowledge is independent of the source arrays)
        self.s_b = np.array(s_b, dtype=float).reshape(-1, n_dim, n_dim)
        self.s_c = np.array(s_c, dtype=float).reshape(-1, n_dim, n_dim)
        self.s_d = np.array(s_d, dtype=float).reshape(-1, n_dim)

        # Probability of error
        self.alpha = alpha
        self.beta = beta
        # Testing parameters
        self.U = (1-self.beta)/self.alpha
        self.B = self.beta/(1-self.alpha)

    @property
    def var_y_inv(self):
        return np.linalg.inv(self.var_y)

    def calculate_L_T(self, observed_trajectory):
        """ Mean over trajectories of log L_T. observed_trajectory: (num_traj, 2n, len_traj) """

        return np.mean(self.sequential_hypothesis_test(observed_trajectory)[:, -1])

    def sequential_hypothesis_test(self, observed_trajectory):
        """ log L_t for each trajectory. observed_trajectory: (num_traj, 2n, len_traj) -> (num_traj, len_traj) """

        return self.sequential_log_likelihood_ratio(observed_trajectory)

    def sequential_log_likelihood_ratio(self, observed_trajectory):
        """
        Cumulative log-likelihood ratio log L_t of "misdirected" vs. "nominal" dynamics (paper Eq. (8)),
        computed with the red team's own knowledge (s_b, s_c, s_d, var_y).

        observed_trajectory: (num_traj, 2n, len_traj), states ordered as (v, p)
        Returns: (num_traj, len_traj), with log L_0 = 0
        """
        observed_trajectory = np.asarray(observed_trajectory)
        n = self.n_dim
        len_traj = observed_trajectory.shape[2]

        v_prev = observed_trajectory[:, :n, :-1]  # (num_traj, n, len_traj-1)
        p_prev = observed_trajectory[:, n:, :-1]
        p = observed_trajectory[:, n:, 1:]

        residual = p - v_prev - p_prev
        deviation = np.einsum('tij,kjt->kit', self.s_b[:len_traj-1], v_prev) + np.einsum('tij,kjt->kit', self.s_c[:len_traj-1], p_prev) \
                    + self.s_d[:len_traj-1].T
        shifted = residual - deviation
        increments = 0.5*(np.einsum('kit,ij,kjt->kt', residual, self.var_y_inv, residual)
                          - np.einsum('kit,ij,kjt->kt', shifted, self.var_y_inv, shifted))

        log_L_t = np.zeros(observed_trajectory.shape[::2])
        log_L_t[:, 1:] = np.cumsum(increments, axis=1)

        return log_L_t
