import numpy as np
import scipy.linalg
import scipy.stats
from tqdm import tqdm


def score_confidence_interval(p_hat, num_samples, significance = 0.05):
    """
    (1 - significance) score confidence interval for a Monte Carlo estimate p_hat from num_samples samples (paper Lemma 2):
    [(p_hat + kappa/2 - delta)/(1 + kappa), (p_hat + kappa/2 + delta)/(1 + kappa)] with z = Q^{-1}(significance/2),
    kappa = z^2/n, delta = z*sqrt(p_hat(1 - p_hat)/n + z^2/(4n^2)). Works elementwise; returns (lower, upper).
    """
    p_hat = np.asarray(p_hat, dtype=float)
    z = scipy.stats.norm.isf(significance/2)
    kappa = z**2/num_samples
    delta = z*np.sqrt(p_hat*(1-p_hat)/num_samples + z**2/(4*num_samples**2))
    center = p_hat + kappa/2

    return (center - delta)/(1 + kappa), (center + delta)/(1 + kappa)


def as_matrix(value, n_dim):
    """ Scalar -> value*I_n, otherwise an (n_dim, n_dim) array """
    value = np.asarray(value, dtype=float)
    if value.ndim == 0:
        return value*np.eye(n_dim)
    assert value.shape == (n_dim, n_dim), f"expected a scalar or an ({n_dim}, {n_dim}) matrix, got shape {value.shape}"
    return value


def as_vector(value, n_dim):
    """ Scalar -> value*1_n, otherwise an (n_dim,) array """
    value = np.asarray(value, dtype=float)
    if value.ndim == 0:
        return value*np.ones(n_dim)
    assert value.shape == (n_dim,), f"expected a scalar or an ({n_dim},) vector, got shape {value.shape}"
    return value


def symmetric_sqrt(matrix):
    """
    Symmetric positive semi-definite square root M^{1/2} of a symmetric PSD matrix, via its eigendecomposition.
    Unlike scipy.linalg.sqrtm, it is reliable for singular matrices (e.g. a deterministic initial state).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (eigenvectors*np.sqrt(np.maximum(eigenvalues, 0)))@eigenvectors.T


def psd_sqrt(matrix):
    """ Factor S with S S^T = matrix, for positive semi-definite matrices (e.g. a deterministic initial state, covariance 0) """
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        return eigenvectors*np.sqrt(np.maximum(eigenvalues, 0))


def matrix_parameter(name):
    """ Attribute that accepts a scalar or an (n, n) matrix and is always stored as an (n, n) matrix """
    def getter(self): return getattr(self, "_" + name)
    def setter(self, value): setattr(self, "_" + name, as_matrix(value, self.n_dim))
    return property(getter, setter)


def vector_parameter(name):
    """ Attribute that accepts a scalar or an (n,) vector and is always stored as an (n,) vector """
    def getter(self): return getattr(self, "_" + name)
    def setter(self, value): setattr(self, "_" + name, as_vector(value, self.n_dim))
    return property(getter, setter)


class BlueTeam():
    """
    Blue team solving the LQR problem with a deception (misdirection) term, in matrix form.

    State: x = (v, p) in R^{2n}, control: u = (alpha, beta) in R^{2n},
    dynamics: x_{t+1} = A x_t + B u_t + w_t with A = [[I, 0], [I, I]], B = I, w_t ~ N(0, diag(var_z, var_y)).
    Horizon T = terminal_time: controls u_0, ..., u_{T-1}, states x_0, ..., x_T (trajectory length T+1).
    The misdirection pattern must be set before calling set_matrices():
        s_b, s_c: (terminal_time, n, n) matrices F^b_t, F^c_t;  s_d: (terminal_time, n) vectors f^d_t.
    For n_dim = 1, s_b and s_c may also be given as (terminal_time,) sequences.

    var_z, var_y and the cost weights r_alpha, r_beta, r_v, t_v: scalar (meaning scalar*I_n) or (n, n) matrices.
    They can be changed after construction; quantities derived from them are recomputed when used.
    Defaults follow the paper's numerical setup (Section V): deterministic x_0 = (v_0, p_0) = (1, 4),
    R_alpha = 1, R_beta = 10, R_v = T_v = 1, bar_v_t = 1 - t/T. A random initial state can be set through
    exp_v_0, exp_p_0, var_v_0, var_p_0, cov_v_p_0.
    random_noise: if False, the team's random generator is seeded with `seed`, so every BlueTeam instance
                  (e.g. one per lambda) sees the same sequence of initial states and noises, while the
                  trajectories within one instance still differ. If True, the generator is seeded from fresh entropy.
    The global np.random state is never used or modified.
    """

    # Parameters accepting a scalar or a matrix/vector (see matrix_parameter, vector_parameter)
    var_z = matrix_parameter("var_z")
    var_y = matrix_parameter("var_y")
    r_alpha = matrix_parameter("r_alpha")
    r_beta = matrix_parameter("r_beta")
    r_v = matrix_parameter("r_v")
    t_v = matrix_parameter("t_v")
    exp_v_0 = vector_parameter("exp_v_0")
    exp_p_0 = vector_parameter("exp_p_0")
    var_v_0 = matrix_parameter("var_v_0")
    var_p_0 = matrix_parameter("var_p_0")
    cov_v_p_0 = matrix_parameter("cov_v_p_0")

    def __init__(self, n_dim = 1, var_z = 0.05, var_y = 0.05, deception_parameter = 0, terminal_time = 30,
                 random_noise = False, seed = 200):

        self.n_dim = n_dim
        n = n_dim

        # Dynamics:
        ## Initial condition x_0 ~ N(m_0, Sigma_0); deterministic (Sigma_0 = 0) as in the paper
        self.exp_v_0 = 1
        self.var_v_0 = 0
        self.exp_p_0 = 4
        self.var_p_0 = 0
        self.cov_v_p_0 = 0  # Cov(v_0, p_0)
        ## Noise:
        self.var_z = var_z
        self.var_y = var_y
        self.random_noise = random_noise
        self.rng = np.random.default_rng(None if random_noise else seed)

        # Time horizon
        self.terminal_time = terminal_time

        # Primary Task (Private)
        ## Reference trajectory bar_v_t = 1 - t/T for t = 0, ..., T, shape (terminal_time+1, n)
        self.bar_v = np.outer(1-np.linspace(0,1,self.terminal_time+1), np.ones(n))
        self.bar_v_T = self.bar_v[-1]
        ## Cost function parameters:
        self.r_alpha = 1
        self.r_beta = 10
        self.r_v = 1
        self.t_v = 1

        # Misdirection
        ## Deception parameter (checked against self.margin here and again in set_matrices)
        self.deception_parameter = deception_parameter
        self.check_deception_parameter()
        ## Misdirection pattern (s_b and s_c are set externally)
        self.s_b = []
        self.s_c = []
        self.s_d = np.zeros((self.terminal_time, n))

        # Define Matrices
        self.A = np.block([[np.eye(n), np.zeros((n, n))],[np.eye(n), np.eye(n)]])
        self.B = np.eye(2*n)
        self.E2 = np.vstack([np.zeros((n, n)), np.eye(n)])  # (2n, n), selects the position block: p = E2^T x

        self.Q = np.zeros((self.terminal_time, 2*n,2*n))
        self.N = np.zeros((self.terminal_time, 2*n,2*n))
        self.q = np.zeros((self.terminal_time, 2*n))
        self.r = np.zeros((self.terminal_time, 2*n))
        self.d = np.zeros((self.terminal_time))
        self.M = np.zeros((self.terminal_time, 2*n,2*n))
        self.H = np.zeros((self.terminal_time, 2*n,2*n))
        self.H_inv = np.zeros((self.terminal_time, 2*n,2*n))
        self.G = np.zeros((self.terminal_time, 2*n,2*n))

        self.matrix_trajectory = []
        self.mean_logLt = []

    # ---------- Helpers ----------

    @property
    def R(self):
        return scipy.linalg.block_diag(self.r_alpha, self.r_beta)

    @property
    def var_y_inv(self):
        return np.linalg.inv(self.var_y)

    @property
    def margin(self):
        """
        Largest admissible deception parameter: well-posed if I - lambda var_y^{-1/2} r_beta^{-1} var_y^{-1/2} >= 0 (Theorem 1),
        i.e. lambda <= smallest eigenvalue of var_y^{1/2} r_beta var_y^{1/2} (= r_beta*var_y when n = 1)
        """
        sqrt_var_y = symmetric_sqrt(self.var_y)
        return np.min(np.linalg.eigvalsh(sqrt_var_y@self.r_beta@sqrt_var_y))

    def check_deception_parameter(self):
        margin = self.margin
        assert 0 <= self.deception_parameter <= margin*(1 + 1e-12), f"deception_parameter must be in [0, {margin}]"

    def initial_mean(self):
        return np.concatenate([self.exp_v_0, self.exp_p_0])

    def initial_covariance(self):
        return np.block([[self.var_v_0, self.cov_v_p_0],[self.cov_v_p_0.T, self.var_p_0]])

    def noise_covariance(self):
        return scipy.linalg.block_diag(self.var_z, self.var_y)

    def misdirection_patterns(self):
        """ s_b, s_c as (terminal_time, n, n) and s_d as (terminal_time, n) arrays """
        n = self.n_dim
        return (np.asarray(self.s_b, dtype=float).reshape(-1, n, n),
                np.asarray(self.s_c, dtype=float).reshape(-1, n, n),
                np.asarray(self.s_d, dtype=float).reshape(-1, n))

    def misdirection_matrix(self, t):
        """ F_t = [F^b_t  F^c_t]^T, (2n, n), so that F_t^T x = F^b_t v + F^c_t p """
        n = self.n_dim
        return np.vstack([np.reshape(self.s_b[t], (n, n)).T, np.reshape(self.s_c[t], (n, n)).T])

    def closed_loop_matrix(self, t):
        """ A - H_t^{-1} G_t^T, the state transition under optimal control (excluding the affine term) """
        return self.A - self.H_inv[t]@self.G[t].T

    # ---------- Optimal control ----------

    def set_matrices(self):

        self.check_deception_parameter()
        self.backward_recursion()
        self.compute_C_D_matrices()
        self.compute_L_matrices()
        self.run_mean_and_covariance_dynamics_under_optimal_control()

    def induction_step(self, matrix_set_, t):

        (P, s, c) = matrix_set_  # At time t+1, coming from backward induction

        n = self.n_dim
        F = self.misdirection_matrix(t)
        f_d = np.reshape(self.s_d[t], n)
        bar_v_t = self.bar_v[t]
        lam = self.deception_parameter
        F_var_y_inv = F@self.var_y_inv  # (2n, n)

        self.Q[t] = scipy.linalg.block_diag(self.r_v, np.zeros((n, n))) + lam*F_var_y_inv@F.T
        self.N[t] = np.hstack([np.zeros((2*n, n)), -lam*F_var_y_inv])
        self.q[t] = np.concatenate([-self.r_v@bar_v_t, np.zeros(n)]) + lam*F_var_y_inv@f_d
        self.r[t] = np.concatenate([np.zeros(n), -lam*self.var_y_inv@f_d])
        self.d[t] = 0.5*bar_v_t@self.r_v@bar_v_t + 0.5*lam*f_d@self.var_y_inv@f_d

        self.M[t] = self.Q[t] + self.A.T@P@self.A
        self.H[t] = self.R + self.B.T@P@self.B
        self.G[t] = self.N[t] + self.A.T@P@self.B
        self.H_inv[t] = np.linalg.inv(self.H[t])

        r_plus_s = self.r[t] + self.B.T@s
        new_P = self.M[t] - self.G[t]@self.H_inv[t]@self.G[t].T
        new_s = self.q[t] + self.A.T@s - self.G[t]@self.H_inv[t]@r_plus_s
        new_c = self.d[t] + c - 0.5*(r_plus_s@self.H_inv[t]@r_plus_s) + 0.5*np.trace(P@self.noise_covariance())

        return (new_P, new_s, new_c)

    def backward_recursion(self):
        """
        Riccati recursion (14) with T = terminal_time: controls u_0, ..., u_{T-1}, states x_0, ..., x_T.
        matrix_trajectory[t] = (P_t, s_t, c_t) for t = 0, ..., T, with the terminal condition at t = T.
        """

        # Terminal condition
        n = self.n_dim
        P_T = scipy.linalg.block_diag(self.t_v, np.zeros((n, n)))
        s_T = np.concatenate([-self.t_v@self.bar_v_T, np.zeros(n)])
        c_T = 0.5*self.bar_v_T@self.t_v@self.bar_v_T

        matrix_trajectory = [(P_T, s_T, c_T)]

        for t in range(self.terminal_time-1, -1, -1):
            matrix_trajectory.append(self.induction_step(matrix_set_=matrix_trajectory[-1], t = t))

        self.matrix_trajectory = matrix_trajectory[::-1]

    def calculate_variable_trajectory(self):
        """ For n = 1: [P_vv, P_vp, P_pp, s_v, s_p, c] at each time """

        self.variable_trajectory = []

        for t in range(self.terminal_time+1):
            (P, s, c) = self.matrix_trajectory[t]
            self.variable_trajectory.append([P[0,0], 0.5*(P[0,1]+P[1,0]), P[1,1], s[0], s[1], c])

    def compute_control_input(self,t,state):

        (P, s, c) = self.matrix_trajectory[t+1]

        return - self.H_inv[t]@(self.G[t].T@state + self.r[t] + self.B.T@s)

    # ---------- Simulation ----------

    def run_dynamics(self, save = True):
        """ Simulates T = terminal_time steps: returns states x_0, ..., x_T (T+1) and controls u_0, ..., u_{T-1} (T) """

        if len(self.matrix_trajectory) == 0:
            self.backward_recursion()

        n = self.n_dim
        chol_initial = psd_sqrt(self.initial_covariance())
        chol_z = psd_sqrt(self.var_z)
        chol_y = psd_sqrt(self.var_y)

        state_trajectory = [self.initial_mean() + chol_initial@self.rng.standard_normal(2*n)]
        control_trajectory = []

        for t in range(self.terminal_time):

            state = state_trajectory[-1]
            control = self.compute_control_input(t, state)
            control_trajectory.append(control)

            noise_z = chol_z@self.rng.standard_normal(n)
            noise_y = chol_y@self.rng.standard_normal(n)
            noise = np.concatenate([noise_z, noise_y])

            state_trajectory.append(self.A@state + self.B@control + noise)

        if save:
            self.state_trajectory = state_trajectory
            self.control_trajectory = control_trajectory

        return state_trajectory, control_trajectory

    def calculate_primary_cost(self, state_trajectory, control_trajectory):
        """ J_primary along one trajectory (paper Eq. (2)-(3)). state_trajectory: (2n, T+1), control_trajectory: (2n, T) """

        n = self.n_dim
        R = self.R
        cost = 0
        for t in range(self.terminal_time):
            control = control_trajectory[:,t]
            v_error = state_trajectory[:n,t] - self.bar_v[t]
            cost += 0.5*control@R@control + 0.5*v_error@self.r_v@v_error

        # Terminal cost
        v_error_T = state_trajectory[:n,-1] - self.bar_v_T
        cost += 0.5*v_error_T@self.t_v@v_error_T

        return cost

    def calculate_deception_measure(self, state_trajectory, control_trajectory):
        """
        Deception measure D(lambda) = d(beta_H0, beta) / d(beta_H0, beta_H1) along one trajectory (paper Eq. (15)),
        with d(a, b) = sum_{t=0}^{T-1} ||a_t - b_t||_2^2. The controls are evaluated at the states of the given trajectory:
        beta_H0 = 0 (optimal under H0, Remark 1), beta = the applied control, beta_H1 = F^b_t v_t + F^c_t p_t + f^d_t.
        By Corollary 1, D = 0 for lambda = 0 and D = 1 for lambda = margin.
        state_trajectory: (2n, T+1), control_trajectory: (2n, T)
        """
        n = self.n_dim
        T = self.terminal_time
        s_b, s_c, s_d = self.misdirection_patterns()

        v, p = state_trajectory[:n, :T], state_trajectory[n:, :T]
        beta = control_trajectory[n:, :T]
        beta_H1 = np.einsum('tij,jt->it', s_b[:T], v) + np.einsum('tij,jt->it', s_c[:T], p) + s_d[:T].T

        return np.sum(beta**2)/np.sum(beta_H1**2)

    # ---------- Stealthiness analysis ----------

    def run_mean_and_covariance_dynamics_under_optimal_control(self):
        """
        Calculates m_k and Cov_k, where x_k follows N(m_k, Cov_k) at time k under optimal control,
        and E[log L_k] (Lemma 3), for k = 0, ..., T.
        """
        mean_states = [self.initial_mean()]
        cov_states = [self.initial_covariance()]
        mean_logLt = [0]

        for t in range(self.terminal_time):

            # prev corresponds to t, and next corresponds to t+1 in each iteration

            (P, s, c) = self.matrix_trajectory[t+1]
            F = self.misdirection_matrix(t)
            H_inv_s = self.H_inv[t]@s

            m_prev = mean_states[-1]
            cov_prev = cov_states[-1]
            second_moment_prev = cov_prev + np.outer(m_prev, m_prev)

            M = self.closed_loop_matrix(t)
            m_next = M@m_prev - H_inv_s
            cov_next = M@second_moment_prev@M.T \
                - np.outer(M@m_prev, s@self.H_inv[t]) - np.outer(H_inv_s, m_prev@M.T) \
                + np.outer(H_inv_s, s@self.H_inv[t]) + self.noise_covariance() - np.outer(m_next, m_next)
            m_logL_next = mean_logLt[-1] - 0.5*(2*(s@self.H_inv[t]@self.E2@self.var_y_inv@F.T@m_prev)
                            + np.trace((F@self.var_y_inv@F.T + 2*self.G[t]@self.H_inv[t]@self.E2@self.var_y_inv@F.T)@second_moment_prev))

            mean_states.append(m_next)
            cov_states.append(cov_next)
            mean_logLt.append(m_logL_next)

        self.mean_states = np.array(mean_states).T  # (num_states = 2n, len_traj)
        self.cov_states = np.array(cov_states).T
        self.mean_logLt = np.array(mean_logLt)

    def calculate_sampling_based_probability(self, log_U, sample_size = 100000):
        """ Monte Carlo estimate of P(log L_t >= log U) at each time, shape: (len_traj,) """

        state_trajectories = []
        for i in tqdm(range(sample_size)):
            state_trajectory, _ = self.run_dynamics(save = False)
            state_trajectories.append(state_trajectory)

        state_trajectories = np.transpose(np.array(state_trajectories), (0,2,1))  # (sample_size, 2n, len_traj)
        logLt_list = self.sequential_log_likelihood_ratio(state_trajectories)

        return np.mean(logLt_list >= log_U, axis = 0)

    def sequential_log_likelihood_ratio(self, observed_trajectory):
        """
        Cumulative log-likelihood ratio log L_t of "misdirected" vs. "nominal" dynamics (paper Eq. (8)),
        computed with the blue team's own knowledge (s_b, s_c, s_d, var_y).

        observed_trajectory: (num_traj, 2n, len_traj), states ordered as (v, p)
        Returns: (num_traj, len_traj), with log L_0 = 0
        """
        observed_trajectory = np.asarray(observed_trajectory)
        n = self.n_dim
        len_traj = observed_trajectory.shape[2]
        s_b, s_c, s_d = self.misdirection_patterns()

        v_prev = observed_trajectory[:, :n, :-1]  # (num_traj, n, len_traj-1)
        p_prev = observed_trajectory[:, n:, :-1]
        p = observed_trajectory[:, n:, 1:]

        residual = p - v_prev - p_prev
        deviation = np.einsum('tij,kjt->kit', s_b[:len_traj-1], v_prev) + np.einsum('tij,kjt->kit', s_c[:len_traj-1], p_prev) \
                    + s_d[:len_traj-1].T
        shifted = residual - deviation
        increments = 0.5*(np.einsum('kit,ij,kjt->kt', residual, self.var_y_inv, residual)
                          - np.einsum('kit,ij,kjt->kt', shifted, self.var_y_inv, shifted))

        log_L_t = np.zeros(observed_trajectory.shape[::2])
        log_L_t[:, 1:] = np.cumsum(increments, axis=1)

        return log_L_t

    def compute_C_D_matrices(self):

        T = self.terminal_time
        b = 2*self.n_dim  # state dimension
        D_matrices = np.zeros((T+1, b))  #(k,2n)  # Row T stays zero so that D_{-1} = 0
        C_matrices = np.zeros((T+1, T, b,b))  #(j,k,2n,2n)  # dim[0] is T+1 because -1 is added for j
        S_matrices = np.zeros((T, b,b))  #(k,2n,2n)

        for j in range(0, T+1):  # row index of C

            # C matrices: C_{j,j-1} = I, C_{j,k} = (A - H_k^{-1} G_k^T) C_{j,k-1}, i.e. the state transition from j to k+1
            if j < T:
                C_matrices[j+1,j,:,:] = np.eye(b)

            for k in range(j, T):  # upper triangle part of C
                if k == 0:
                    C_matrices[j,k,:,:] = self.closed_loop_matrix(k)
                else:
                    C_matrices[j,k,:,:] = self.closed_loop_matrix(k)@C_matrices[j,k-1,:,:]

        for k in range(T):
            (P, s, c) = self.matrix_trajectory[k+1]
            if k == 0:
                D_matrices[k,:] = -self.H_inv[k]@s
            else:
                D_matrices[k,:] = self.closed_loop_matrix(k)@D_matrices[k-1,:] - self.H_inv[k]@s

            F = self.misdirection_matrix(k)
            S_matrices[k,:,:] = F@self.var_y_inv@F.T + 2*self.G[k]@self.H_inv[k]@self.E2@self.var_y_inv@F.T

        self.C_matrices = C_matrices
        self.D_matrices = D_matrices
        self.S_matrices = S_matrices

    def compute_L_matrices(self):
        """
        log L_t = L0_t + L1_t^T w_t + w_t^T L2_t w_t for t = 0, ..., T,
        with w_t = (x_0, w_0, ..., w_{t-1}) the stacked initial state and noises
        """

        T = self.terminal_time
        b = 2*self.n_dim  # state dimension
        coef = 0.5
        E2_var_y_inv = self.E2@self.var_y_inv  # (2n, n)
        L0_matrices = np.zeros(T+1)
        L1_matrices = np.zeros((T+1, b*(T+1)))
        L2_matrices = np.zeros((T+1, b*(T+1), b*(T+1)))

        L0 = 0

        for t in range(1, T+1):

            (P, s, c) = self.matrix_trajectory[t]
            F = self.misdirection_matrix(t-1)

            L0 += -coef*(2*(s@self.H_inv[t-1]@E2_var_y_inv@F.T@self.D_matrices[t-2]) + self.D_matrices[t-2]@self.S_matrices[t-1]@self.D_matrices[t-2])  #D_{-1} = 0
            L0_matrices[t] = L0

            # L1 component that corresponds to k
            for k in range(-1, t):
                L1_component = np.zeros(b)
                if k >= 0:
                    F_k = self.misdirection_matrix(k)
                    L1_component += 2*E2_var_y_inv@F_k.T@self.D_matrices[k-1]   #Note: D_{-1}=0
                for j in range(k+1, t):
                    (P, s, c) = self.matrix_trajectory[j+1]
                    F_j = self.misdirection_matrix(j)
                    S_sym_D = (self.S_matrices[j]+self.S_matrices[j].T)@self.D_matrices[j-1]
                    if j == 0:
                        L1_component -= (2*F_j@E2_var_y_inv.T@self.H_inv[j]@s + S_sym_D)  #C_{0,-1} = I
                    else:
                        L1_component -= self.C_matrices[k+1,j-1].T@(2*F_j@E2_var_y_inv.T@self.H_inv[j]@s + S_sym_D)

                L1_matrices[t,b*(k+1):b*(k+2)] = coef*L1_component

            # First part of (L2)_ij
            for i in range(0, t):
                E2_F_i = E2_var_y_inv@self.misdirection_matrix(i).T  # (2n, 2n)
                if i == 0:
                    L2_matrices[t,b:2*b,0:b] += 2*coef*E2_F_i #C_{0,-1} = I
                else:
                    for j in range(0, i+1):
                        L2_matrices[t,b*(i+1):b*(i+2),b*j:b*(j+1)] += 2*coef*(E2_F_i@self.C_matrices[j,i-1])

            # Second part of (L2)_ij
            for i in range(0, t):
                for j in range(0, t):
                    for k in range(max(i,j), t):
                        if k == 0: # This means i=0 and j=0
                            L2_matrices[t,b*i:b*(i+1),b*j:b*(j+1)] -= coef*self.S_matrices[k].T #C_{0,-1} = I
                        else:
                            L2_matrices[t,b*i:b*(i+1),b*j:b*(j+1)] -= coef*self.C_matrices[i,k-1].T@self.S_matrices[k].T@self.C_matrices[j,k-1]

        self.L0_matrices = L0_matrices
        self.L1_matrices = L1_matrices
        self.L2_matrices = L2_matrices
        self.L2_sym_matrices = (L2_matrices + np.transpose(L2_matrices, axes = (0,2,1)))/2

    def get_theorem_2_bounds(self, log_U, optimize=True):
        """
        Upper bound on P(log L_t >= log U) at each time t = 0, ..., T (Theorem 2), shape: (T+1,).
        log U - E[log L_t] is split into epsilon1 + epsilon2; with optimize=True the split is chosen
        from a grid to minimize the bound, otherwise epsilon1 = 0.6*(log U - E[log L_t]).
        Times with E[log L_t] >= log U, where the theorem does not apply, get 1.
        Computed bounds are not clipped, so they can exceed 1.
        """

        # Norms in the exponents only depend on t, so compute them once
        b = 2*self.n_dim  # state dimension
        # (Sigma_w^t)^{1/2} = diag(Sigma_0^{1/2}, Sigma_w^{1/2}, ..., Sigma_w^{1/2}) since Sigma_w^t is block diagonal
        sqrt_cov_state0 = symmetric_sqrt(self.initial_covariance())
        sqrt_cov_noise = symmetric_sqrt(self.noise_covariance())
        linear_norm_sq = np.zeros(self.terminal_time+1)
        quadratic_fro_sq = np.zeros(self.terminal_time+1)
        quadratic_spec = np.zeros(self.terminal_time+1)

        for t in range(self.terminal_time+1):
            sqrt_cov_w_t = scipy.linalg.block_diag(sqrt_cov_state0, *([sqrt_cov_noise]*t))
            mean_w_t = np.zeros(b*(t+1))
            mean_w_t[:b] = self.initial_mean()

            L2_t_sym = self.L2_sym_matrices[t,:b*(t+1),:b*(t+1)]
            L1_t = self.L1_matrices[t,:b*(t+1)]
            whitened_L2 = sqrt_cov_w_t@L2_t_sym@sqrt_cov_w_t

            linear_norm_sq[t] = np.linalg.norm(sqrt_cov_w_t@(L1_t+2*L2_t_sym@mean_w_t), ord = 2)**2
            quadratic_fro_sq[t] = np.linalg.norm(whitened_L2, ord = 'fro')**2
            quadratic_spec[t] = np.linalg.norm(whitened_L2, ord = 2)

        def get_bounds(epsilon1, epsilon2):

            assert np.all(epsilon1 >= 0)
            assert np.all(epsilon2 >= 0)

            with np.errstate(divide = 'ignore', invalid = 'ignore'):  # t = 0 has zero norms -> bound 0
                exponent1 = -(epsilon1**2)/(2*linear_norm_sq)
                exponent2 = -(1/8)*np.minimum(epsilon2**2/quadratic_fro_sq, epsilon2/quadratic_spec)

            return np.exp(exponent1) + np.exp(exponent2)

        gap = log_U - self.mean_logLt
        # Theorem 2 requires log U > E[log L_t]; where it doesn't hold, there is no concentration bound,
        # so those times get the trivial bound 1. safe_gap is a placeholder there to keep the computation valid.
        applicable = gap > 0
        safe_gap = np.where(applicable, gap, 1)

        if optimize:
            # Grid of epsilon1 from 0.001 to 0.999*gap (from 0.001*gap if gap < 0.001, so that epsilon2 >= 0)
            start = np.where(safe_gap >= 0.001, 0.001, 0.001*safe_gap)
            epsilon1_candidates = np.linspace(start, 0.999*safe_gap, num=50)
            bound_candidates = np.array([get_bounds(epsilon1, safe_gap - epsilon1) for epsilon1 in epsilon1_candidates])  #(num, len_traj)
            theorem2_bounds = np.min(bound_candidates, axis = 0)
        else:
            epsilon1 = 0.6*safe_gap
            theorem2_bounds = get_bounds(epsilon1, safe_gap - epsilon1)

        theorem2_bounds[~applicable] = 1.0

        return theorem2_bounds
