import numpy as np
import scipy.linalg
from tqdm import tqdm

E2 = np.array([0,1])  # Selects the position component of the state (v, x)


class BlueTeam():
    """
    Blue team solving the LQR problem with a deception (misdirection) term, in matrix form.

    State: (v, x), control: (alpha, beta), dynamics: state_{t+1} = A state_t + B control_t + noise.
    The misdirection pattern s_b, s_c, s_d must be set before calling set_matrices().

    random_noise: if False, the process noise at each step t is drawn after np.random.seed(t+200),
                  so every trajectory (and every lambda) sees the same noise sequence.
                  If True, noise is drawn from the global RNG without reseeding.
    """

    def __init__(self, var_z = 0.05, var_y = 0.05, deception_parameter = 0, terminal_time = 30, random_noise = False):

        # Dynamics:
        ## Initial condition:
        self.exp_v_0 = 2
        self.var_v_0 = 0.1
        self.exp_x_0 = 4
        self.var_x_0 = 0.1
        self.cov_x_v_0 = 0
        ## Noise:
        self.var_z = var_z
        self.var_y = var_y
        self.random_noise = random_noise

        # Time horizon
        self.terminal_time = terminal_time

        # Primary Task (Private)
        ## Reference trajectory
        self.bar_v = 2-np.linspace(0,1,self.terminal_time)
        self.bar_v_T = self.bar_v[-1]
        ## Cost function parameters:
        self.r_alpha = 1
        self.r_beta = 10
        self.r_v = 1
        self.t_v = 1

        # Misdirection
        ## Deception parameter
        self.margin = self.r_beta*self.var_y
        assert deception_parameter <= self.margin
        self.deception_parameter = deception_parameter
        ## Misdirection pattern (s_b and s_c are set externally)
        self.s_b = []
        self.s_c = []
        self.s_d = np.zeros(self.terminal_time)

        # Define Matrices
        self.A = np.array([[1,0],[1,1]])
        self.B = np.eye(2)
        self.R = np.array([[self.r_alpha, 0],[0, self.r_beta]])

        self.Q = np.zeros((self.terminal_time, 2,2))
        self.N = np.zeros((self.terminal_time, 2,2))
        self.q = np.zeros((self.terminal_time, 2))
        self.r = np.zeros((self.terminal_time, 2))
        self.d = np.zeros((self.terminal_time))
        self.M = np.zeros((self.terminal_time, 2,2))
        self.H = np.zeros((self.terminal_time, 2,2))
        self.H_inv = np.zeros((self.terminal_time, 2,2))
        self.G = np.zeros((self.terminal_time, 2,2))

        self.matrix_trajectory = []
        self.mean_logLt = []

    # ---------- Helpers ----------

    def initial_mean(self):
        return np.array([self.exp_v_0, self.exp_x_0])

    def initial_covariance(self):
        return np.array([[self.var_v_0, self.cov_x_v_0],[self.cov_x_v_0, self.var_x_0]])

    def noise_covariance(self):
        return np.array([[self.var_z, 0],[0, self.var_y]])

    def misdirection_vector(self, t):
        return np.array([self.s_b[t], self.s_c[t]])

    def closed_loop_matrix(self, t):
        """ A - H_t^{-1} G_t^T, the state transition under optimal control (excluding the affine term) """
        return self.A - self.H_inv[t]@self.G[t].T

    # ---------- Optimal control ----------

    def set_matrices(self):

        self.backward_recursion()
        self.compute_C_D_matrices()
        self.compute_L_matrices()
        self.run_mean_and_covariance_dynamics_under_optimal_control()

    def induction_step(self, matrix_set_, t):

        (P, s, c) = matrix_set_  # At time t+1, coming from backward induction

        s_b_t = self.s_b[t]
        s_c_t = self.s_c[t]
        s_d_t = self.s_d[t]
        bar_v_t = self.bar_v[t]
        lam_over_var_y = self.deception_parameter/self.var_y

        self.Q[t] = np.array([[self.r_v + lam_over_var_y*(s_b_t**2), lam_over_var_y*(s_b_t*s_c_t)],
                              [lam_over_var_y*(s_b_t*s_c_t), lam_over_var_y*(s_c_t**2)]])
        self.N[t] = np.array([[0, -lam_over_var_y*s_b_t],[0, -lam_over_var_y*s_c_t]])
        self.q[t] = np.array([-self.r_v*bar_v_t + lam_over_var_y*(s_b_t*s_d_t), lam_over_var_y*(s_c_t*s_d_t)])
        self.r[t] = np.array([0, -lam_over_var_y*s_d_t])
        self.d[t] = 0.5*self.r_v*(bar_v_t**2) + 0.5*lam_over_var_y*(s_d_t**2)

        self.M[t] = self.Q[t] + self.A.T@P@self.A
        self.H[t] = self.R + self.B.T@P@self.B
        self.G[t] = self.N[t] + self.A.T@P@self.B
        self.H_inv[t] = np.linalg.inv(self.H[t])

        r_plus_s = self.r[t] + self.B.T@s
        new_P = self.M[t] - self.G[t]@self.H_inv[t]@self.G[t].T
        new_s = self.q[t] + self.A.T@s - self.G[t]@self.H_inv[t]@r_plus_s
        new_c = self.d[t] + c - 0.5*(r_plus_s@self.H_inv[t]@r_plus_s)

        return (new_P, new_s, new_c)

    def backward_recursion(self):

        # Terminal condition
        P_T = np.array([[self.t_v, 0],[0,0]])
        s_T = np.array([-self.t_v*self.bar_v_T, 0])
        c_T = 0.5*self.t_v*(self.bar_v_T**2)

        matrix_trajectory = [(P_T, s_T, c_T)]

        for t in range(self.terminal_time-1, -1, -1):
            matrix_trajectory.append(self.induction_step(matrix_set_=matrix_trajectory[-1], t = t))

        self.matrix_trajectory = matrix_trajectory[::-1]

    def calculate_variable_trajectory(self):

        self.variable_trajectory = []

        for t in range(self.terminal_time):
            (P, s, c) = self.matrix_trajectory[t]
            self.variable_trajectory.append([P[0,0], 0.5*(P[0,1]+P[1,0]), P[1,1], s[0], s[1], c])

    def compute_control_input(self,t,state):

        (P, s, c) = self.matrix_trajectory[t+1]

        return - self.H_inv[t]@(self.G[t].T@state + self.r[t] + self.B.T@s)

    # ---------- Simulation ----------

    def run_dynamics(self, save = True):

        if len(self.matrix_trajectory) == 0:
            self.backward_recursion()

        v_0 = np.random.normal(self.exp_v_0, self.var_v_0**0.5)
        x_0 = np.random.normal(self.exp_x_0, self.var_x_0**0.5)

        state_trajectory = [np.array([v_0, x_0])]
        control_trajectory = []

        for t in range(self.terminal_time-1):

            state = state_trajectory[-1]
            control = self.compute_control_input(t, state)
            control_trajectory.append(control)

            if not self.random_noise:
                np.random.seed(t+200)
            noise = np.array([np.random.normal(0, self.var_z**0.5), np.random.normal(0, self.var_y**0.5)])

            state_trajectory.append(self.A@state + self.B@control + noise)

        if save:
            self.state_trajectory = state_trajectory
            self.control_trajectory = control_trajectory

        return state_trajectory, control_trajectory

    def calculate_primary_cost(self, state_trajectory, control_trajectory):
        """ state_trajectory: (2, len_traj), control_trajectory: (2, len_traj-1) """

        cost = 0
        for t in range(self.terminal_time-1):
            control = control_trajectory[:,t]
            state = state_trajectory[:,t]
            cost += control@self.R@control + self.r_v/2*(state[0] - self.bar_v[t])**2

        # Terminal cost
        cost += self.t_v/2*(state_trajectory[0,-1] - self.bar_v_T)**2

        return cost

    # ---------- Stealthiness analysis ----------

    def run_mean_and_covariance_dynamics_under_optimal_control(self):
        """
        Calculates m_k and Cov_k, where x_k follows N(m_k, Cov_k) at time k under optimal control,
        and E[log L_k].
        """
        mean_states = [self.initial_mean()]
        cov_states = [self.initial_covariance()]
        mean_logLt = [0]

        for t in range(self.terminal_time-1):

            # prev corresponds to t, and next corresponds to t+1 in each iteration

            (P, s, c) = self.matrix_trajectory[t+1]
            sv = self.misdirection_vector(t)
            H_inv_s = self.H_inv[t]@s

            m_prev = mean_states[-1]
            cov_prev = cov_states[-1]
            second_moment_prev = cov_prev + np.outer(m_prev, m_prev)

            M = self.closed_loop_matrix(t)
            m_next = M@m_prev - H_inv_s
            cov_next = M@second_moment_prev@M.T \
                - np.outer(M@m_prev, s@self.H_inv[t]) - np.outer(H_inv_s, m_prev@M.T) \
                + np.outer(H_inv_s, s@self.H_inv[t]) + self.noise_covariance() - np.outer(m_next, m_next)
            m_logL_next = mean_logLt[-1] - (1/(2*self.var_y))*((2*(E2@H_inv_s)*(sv@m_prev)
                            + np.trace((np.outer(sv,sv) + 2*np.outer(self.G[t]@self.H_inv[t]@E2, sv))@second_moment_prev)))

            mean_states.append(m_next)
            cov_states.append(cov_next)
            mean_logLt.append(m_logL_next)

        self.mean_states = np.array(mean_states).T  # (num_states = 2, len_traj)
        self.cov_states = np.array(cov_states).T
        self.mean_logLt = np.array(mean_logLt)

    def calculate_sampling_based_probability(self, log_U, sample_size = 100000):
        """ Monte Carlo estimate of P(log L_t >= log U) at each time, shape: (len_traj,) """

        state_trajectories = []
        for i in tqdm(range(sample_size)):
            state_trajectory, _ = self.run_dynamics(save = False)
            state_trajectories.append(state_trajectory)

        state_trajectories = np.transpose(np.array(state_trajectories), (0,2,1))  # (sample_size, 2, len_traj)
        logLt_list = self.sequential_log_likelihood_ratio(state_trajectories)

        return np.mean(logLt_list >= log_U, axis = 0)

    def sequential_log_likelihood_ratio(self, observed_trajectory):
        """
        Cumulative log-likelihood ratio log L_t of "misdirected" vs. "nominal" dynamics,
        computed with the blue team's own knowledge (s_b, s_c, s_d, var_y).

        observed_trajectory: (num_traj, 2, len_traj), states ordered as (v, x)
        Returns: (num_traj, len_traj), with log L_0 = 0
        """
        observed_trajectory = np.asarray(observed_trajectory)
        len_traj = observed_trajectory.shape[2]

        v_prev = observed_trajectory[:, 0, :-1]
        x_prev = observed_trajectory[:, 1, :-1]
        x = observed_trajectory[:, 1, 1:]
        s_b_t = np.asarray(self.s_b)[:len_traj-1]
        s_c_t = np.asarray(self.s_c)[:len_traj-1]
        s_d_t = np.asarray(self.s_d)[:len_traj-1]

        residual = x - v_prev - x_prev
        increments = (1/(2*self.var_y))*(residual**2 - (residual - s_b_t*v_prev - s_c_t*x_prev - s_d_t)**2)

        log_L_t = np.zeros(observed_trajectory.shape[::2])
        log_L_t[:, 1:] = np.cumsum(increments, axis=1)

        return log_L_t

    def compute_C_D_matrices(self):

        T = self.terminal_time
        D_matrices = np.zeros((T+1, 2))  #(k,2)  # Row T stays zero so that D_{-1} = 0
        C_matrices = np.zeros((T+1, T, 2,2))  #(j,k,2,2)  # dim[0] is T+1 because -1 is added for j
        S_matrices = np.zeros((T, 2,2))  #(k,2,2)

        for j in range(0, T+1):  # row index of C

            # C matrices: C_{j,j-1} = I, C_{j,k} = C_{j,k-1} (A - H_k^{-1} G_k^T)
            if j < T:
                C_matrices[j+1,j,:,:] = np.eye(2)

            for k in range(j, T):  # upper triangle part of C
                if k == 0:
                    C_matrices[j,k,:,:] = self.closed_loop_matrix(k)
                else:
                    C_matrices[j,k,:,:] = C_matrices[j,k-1,:,:]@self.closed_loop_matrix(k)

        for k in range(T):
            (P, s, c) = self.matrix_trajectory[k+1]
            if k == 0:
                D_matrices[k,:] = -self.H_inv[k]@s
            else:
                D_matrices[k,:] = self.closed_loop_matrix(k)@D_matrices[k-1,:] - self.H_inv[k]@s

            sv = self.misdirection_vector(k)
            S_matrices[k,:,:] = np.outer(sv, sv) + 2*np.outer(self.G[k]@self.H_inv[k]@E2, sv)

        self.C_matrices = C_matrices
        self.D_matrices = D_matrices
        self.S_matrices = S_matrices

    def compute_L_matrices(self):
        """ log L_t = L0_t + L1_t^T w_t + w_t^T L2_t w_t, with w_t the stacked initial state and noises """

        T = self.terminal_time
        coef = 1/(2*self.var_y)
        L0_matrices = np.zeros(T)
        L1_matrices = np.zeros((T, 2*(T+1)))
        L2_matrices = np.zeros((T, 2*(T+1), 2*(T+1)))

        L0 = 0

        for t in range(1, T):

            (P, s, c) = self.matrix_trajectory[t]
            sv = self.misdirection_vector(t-1)

            L0 += -coef*(2*(E2@self.H_inv[t-1]@s)*(sv@self.D_matrices[t-2]) + self.D_matrices[t-2]@self.S_matrices[t-1]@self.D_matrices[t-2])  #D_{-1} = 0
            L0_matrices[t] = L0

            # L1 component that corresponds to k
            for k in range(-1, t):
                L1_component = np.array([0,0], np.float64)
                if k >= 0:
                    sv_k = self.misdirection_vector(k)
                    L1_component += 2*E2*(self.D_matrices[k-1]@sv_k)   #Note: D_{-1}=0
                for j in range(k+1, t):
                    (P, s, c) = self.matrix_trajectory[j+1]
                    sv_j = self.misdirection_vector(j)
                    S_sym_D = (self.S_matrices[j]+self.S_matrices[j].T)@self.D_matrices[j-1]
                    if j == 0:
                        L1_component -= (2*sv_j*(s@self.H_inv[j]@E2) + S_sym_D)  #C_{0,-1} = I
                    else:
                        L1_component -= self.C_matrices[k+1,j-1].T@(2*(np.outer(sv_j,s)@self.H_inv[j]@E2) + S_sym_D)

                L1_matrices[t,2*k+2:2*k+4] = coef*L1_component

            # First part of (L2)_ij
            for i in range(0, t):
                sv_i = self.misdirection_vector(i)
                if i == 0:
                    L2_matrices[t,2:4,0:2] += 2*coef*(np.outer(E2,sv_i)) #C_{0,-1} = I
                else:
                    for j in range(0, i+1):
                        L2_matrices[t,2*(i+1):2*(i+2),2*j:2*(j+1)] += 2*coef*(np.outer(E2,sv_i)@self.C_matrices[j,i-1])

            # Second part of (L2)_ij
            for i in range(0, t):
                for j in range(0, t):
                    for k in range(max(i,j), t):
                        if k == 0: # This means i=0 and j=0
                            L2_matrices[t,2*i:2*(i+1),2*j:2*(j+1)] -= coef*self.S_matrices[k].T #C_{0,-1} = I
                        else:
                            L2_matrices[t,2*i:2*(i+1),2*j:2*(j+1)] -= coef*self.C_matrices[i,k-1].T@self.S_matrices[k].T@self.C_matrices[j,k-1]

        self.L0_matrices = L0_matrices
        self.L1_matrices = L1_matrices
        self.L2_matrices = L2_matrices
        self.L2_sym_matrices = (L2_matrices + np.transpose(L2_matrices, axes = (0,2,1)))/2

    def get_theorem_2_bounds(self, log_U, optimize=True):
        """
        Upper bound on P(log L_t >= log U) at each time t (Theorem 2), shape: (len_traj,).
        log U - E[log L_t] is split into epsilon1 + epsilon2; with optimize=True the split is chosen
        from a grid to minimize the bound, otherwise epsilon1 = 0.6*(log U - E[log L_t]).
        Times with E[log L_t] >= log U, where the theorem does not apply, get 1.
        Computed bounds are not clipped, so they can exceed 1.
        """

        # Norms in the exponents only depend on t, so compute them once
        cov_state0 = self.initial_covariance()
        cov_noise = self.noise_covariance()
        linear_norm_sq = np.zeros(self.terminal_time)
        quadratic_fro_sq = np.zeros(self.terminal_time)
        quadratic_spec = np.zeros(self.terminal_time)

        for t in range(self.terminal_time):
            sqrt_cov_w_t = scipy.linalg.sqrtm(scipy.linalg.block_diag(cov_state0, *([cov_noise]*t)))
            mean_w_t = np.zeros(2*(t+1))
            mean_w_t[:2] = self.initial_mean()

            L2_t_sym = self.L2_sym_matrices[t,:2*(t+1),:2*(t+1)]
            L1_t = self.L1_matrices[t,:2*(t+1)]
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
