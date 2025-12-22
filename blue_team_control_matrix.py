import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

class BlueTeamMatrix():
    def __init__(self, deception_parameter = 0, terminal_time = 30):
        
        # Dynamics:
        ## Initial condition:
        self.exp_v_0 = 2
        self.var_v_0 = 0.1
        self.exp_x_0 = 4
        self.var_x_0 = 0.1
        self.cov_x_v_0 = 0
        ## Noise:
        self.var_z = 0.25
        self.var_y = 0.25
        
        # Time horizon
        self.terminal_time = terminal_time
        
        # Reference trajectory
        self.bar_v = 2-np.linspace(0,1,self.terminal_time)
        self.bar_v_T = self.bar_v[-1]
        
        # Cost function parameters:
        self.r_alpha = 1
        self.r_beta = 10
        self.r_v = 1
        self.t_v = 1
        
        # Deception parameter
        self.deception_parameter = deception_parameter
        self.s_b = 1*np.ones(self.terminal_time) #np.zeros(self.terminal_time)
        self.s_c = 1*np.ones(self.terminal_time) #0.2*np.sin(np.linspace(0, self.terminal_time,self.terminal_time))
        # self.s_c = 0.2*np.sin(np.linspace(0, self.terminal_time,self.terminal_time))
        self.s_d = 0*np.ones(self.terminal_time) #2*np.sin(np.linspace(0, self.terminal_time,self.terminal_time))
        
        self.matrix_trajectory = []
        self.Delta_trajectory = np.zeros(self.terminal_time)
        
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
        
        
        self.margin = 0.5*self.r_beta*self.var_y
        
    # def compute_C_matrix(self, j,k):
        
    #     negative_C = np.eye(2)
        
    #     for i in range(j+1,k):
            
    #         negative_C = negative_C@(self.A - self.H_inv[i]@self.G[i].T)
            
    #     return -negative_C
    
    def compute_C_D_matrices(self):
       
        D_matrices = np.zeros((self.terminal_time+1, 2))  #(k,2) 
        C_matrices = np.zeros((self.terminal_time+1, self.terminal_time, 2,2))  #(j,k,2,2)  # dim[0] is T+1 because -1 is added for j
        S_matrices = np.zeros((self.terminal_time, 2,2))  #(k,2,2)
        
        e2 = np.array([0,1]).T
        
        for j in range(0,self.terminal_time+1):  # row index of C
            
            # Build C matrices
            if j<self.terminal_time:
                C_matrices[j+1,j,:,:] = np.eye(2)
                        
            for k in range(j, self.terminal_time):  # upper triangle part of C
                if k == 0:
                    C_matrices[j,k,:,:] = (self.A - self.H_inv[k]@self.G[k].T)
                else:
                    C_matrices[j,k,:,:] = C_matrices[j,k-1,:,:]@(self.A - self.H_inv[k]@self.G[k].T)  # This gives C_{j, k}
              
            # Build D matrices      
            # For D matrices, think index j as k  # This is row index of D
            if j < self.terminal_time:
                (P, s, c) = self.matrix_trajectory[j+1]
                if j ==0:
                    D_matrices[j,:] = -self.H_inv[j]@s
                else:
                    D_matrices[j,:] = (self.A - self.H_inv[j]@self.G[j].T)@D_matrices[j-1,:] - self.H_inv[j]@s
                    
            # Build S matrices
            if (j<self.terminal_time):  # from 1 to T-1
                sv = np.array([self.s_b[j], self.s_c[j]])
                S_matrices[j,:,:] = np.outer(sv, sv.T) + 2*np.outer(self.G[j]@self.H_inv[j]@e2,sv.T)
                
        self.C_matrices = C_matrices
        self.D_matrices = D_matrices
        self.S_matrices = S_matrices
        
    def compute_L_matrices(self):
        
        L0_matrices = np.zeros(self.terminal_time)
        L1_matrices = np.zeros((self.terminal_time, 2*(self.terminal_time+1)))
        L2_matrices = np.zeros((self.terminal_time, 2*(self.terminal_time+1), 2*(self.terminal_time+1)))
        
        e2 = np.array([0,1]).T
        L0 = 0
        
        for t in range(1,self.terminal_time):
            
            (P, s, c) = self.matrix_trajectory[t]
            sv = np.array([self.s_b[t-1], self.s_c[t-1]])
            
            L0 += -(1/(2*self.var_y))*(2*(e2.T@self.H_inv[t-1]@s)*(sv.T@self.D_matrices[t-2])+ self.D_matrices[t-2].T@self.S_matrices[t-1]@self.D_matrices[t-2])  #D_{-1} = 0
            
            L0_matrices[t]=L0
                
            # L1 component that corresponds to k
            for k in range(-1, t):
                L1_component = np.array([0,0], np.float64)
                if k >= 0:
                    sv_k = np.array([self.s_b[k], self.s_c[k]])
                    L1_component += 2*e2*(self.D_matrices[k-1].T@sv_k)   # Same for all t with same k -> Make it more efficient?  #Note: D_{-1}=0
                if k<=t-2:
                    for j in range(k+1, t):
                        (P, s, c) = self.matrix_trajectory[j+1]
                        sv_j = np.array([self.s_b[j], self.s_c[j]])
                        if j == 0:
                            L1_component -= (2*sv_j*(s.T@self.H_inv[j]@e2) + (self.S_matrices[j]+self.S_matrices[j].T)@self.D_matrices[j-1])  #C_{0,-1} = I
                        else:
                            L1_component -= self.C_matrices[k+1,j-1].T@(2*(np.outer(sv_j,s.T)@self.H_inv[j]@e2) + (self.S_matrices[j]+self.S_matrices[j].T)@self.D_matrices[j-1])
                        
                L1_matrices[t,2*k+2:2*k+4] = (1/(2*self.var_y))*L1_component
                
            # First part of (L2)_ij
            for i in range(0,t):
                sv_i = np.array([self.s_b[i], self.s_c[i]])
                if i == 0: 
                    j = 0 
                    L2_matrices[t,2*(i+1):2*(i+2),2*j:2*(j+1)] += 2*(1/(2*self.var_y))*(np.outer(e2,sv_i.T)) #C_{0,-1} = I
                else:
                    for j in range(0,i+1):
                        L2_matrices[t,2*(i+1):2*(i+2),2*j:2*(j+1)] += 2*(1/(2*self.var_y))*(np.outer(e2,sv_i.T)@self.C_matrices[j,i-1])  
            
            # Second part of (L2)_ij
            for i in range(0,t):
                for j in range(0,t):
                    m = max(i,j)  # Goes from 0 to t-1
                    for k in range(m, t):
                        if k == 0: # This means i=0 and j=0
                            L2_matrices[t,2*i:2*(i+1),2*j:2*(j+1)] -= (1/(2*self.var_y))*self.S_matrices[k].T #C_{0,-1} = I
                        else:
                            L2_matrices[t,2*i:2*(i+1),2*j:2*(j+1)] -= (1/(2*self.var_y))*self.C_matrices[i,k-1].T@self.S_matrices[k].T@self.C_matrices[j,k-1]
            
            
            self.L0_matrices = L0_matrices
            self.L1_matrices = L1_matrices
            self.L2_matrices = L2_matrices
            self.L2_sym_matrices = (L2_matrices + np.transpose(L2_matrices, axes = (0,2,1)))/2
            
            
            for t in range(1,self.terminal_time):
                L2_t_sym = self.L2_sym_matrices[t,:2*(t+1),:2*(t+1)]
                print(np.linalg.cholesky(L2_t_sym))
                print(sp.linalg.sqrtm(L2_t_sym))
            #     L2 = L2_matrices[t,:2*(t+1),:2*(t+1)]
            #     L1 = L1_matrices[t,:2*(t+1)]
                
            #     # print(L2_t_sym[:,2*t:])
            #     # print(L2_t_sym[2*t:,:])
            #     print(L1[2*t:])
                
                # if np.linalg.matrix_rank(L2_t_sym) == 2*(t+1):
                #     print("O")
                # else:
                #     print("Singular",np.linalg.matrix_rank(L2_t_sym) ,L2_t_sym.shape, L2_t_sym)
            
    def get_theorem_1_bound(self, t, eta, B):
        
        t=2
        cov_state0 = np.array([[self.var_v_0, self.cov_x_v_0],[self.cov_x_v_0, self.var_x_0]])
        cov_noise = np.array([[self.var_z,0],[0,self.var_y]])
        cov_g_t = sp.linalg.block_diag(cov_state0, *([cov_noise]*t))
        L2_t_sym = self.L2_sym_matrices[t,:2*(t+1),:2*(t+1)]  # 2(t+1)*2(t+1)
        eta_margin = 1/(2*np.linalg.norm(sp.linalg.sqrtm(cov_g_t)@L2_t_sym@sp.linalg.sqrtm(cov_g_t),ord = 2))  # Take L_2 norm of a matrix
            
        print(eta_margin)
        # eta should be within (0, eta_margin)
        # L0_t = self.L0_matrices[t]
        # L1_t = self.L1_matrices[t,:2*(t+1)]
        # L2_t_sym_inv = np.linalg.inv(L2_t_sym)
        # mean_g_t = (1/2)*L2_t_sym_inv@L1_t
        # mean_g_t[0] += self.exp_v_0
        # mean_g_t[1] += self.exp_x_0
        # temp_matrix1 = mean_g_t.T@L2_t_sym@sp.linalg.sqrtm(cov_g_t)
        # temp_matrix2 = np.eye(2*(t+1)) - 2*eta*(sp.linalg.sqrtm(cov_g_t)@L2_t_sym@sp.linalg.sqrtm(cov_g_t))
        # exponent = -eta*(np.log(B) - (L0_t - (L1_t.T@L2_t_sym_inv@L1_t)/4)) + eta*(mean_g_t.T@L2_t_sym@mean_g_t) + 2*(eta**2)*(temp_matrix1@np.linalg.inv(temp_matrix2)@temp_matrix1.T)
        
        
        # probability_bound = np.exp(exponent) * (np.linalg.det(temp_matrix2)**(-0.5))
        
        # print(probability_bound)
        
    def get_theorem_2_bound(self, t, epsilon1, log_B):
        
        # Set parameter
        epsilon2 = log_B - self.mean_logLt[t] - epsilon1
        assert epsilon2>=0
                
        # Calculate required matrices        
        cov_state0 = np.array([[self.var_v_0, self.cov_x_v_0],[self.cov_x_v_0, self.var_x_0]])
        cov_noise = np.array([[self.var_z,0],[0,self.var_y]])
        cov_w_t = sp.linalg.block_diag(cov_state0, *([cov_noise]*t))
        sqrt_cov_w_t = sp.linalg.sqrtm(cov_w_t)
        mean_w_t = np.zeros(2*(t+1))
        mean_w_t[0] = self.exp_v_0
        mean_w_t[1] = self.exp_x_0
        
        L2_t_sym = self.L2_sym_matrices[t,:2*(t+1),:2*(t+1)]
        L1_t = self.L1_matrices[t,:2*(t+1)]
        
        # Calculate expoenent
        exponent1 = -(epsilon1**2)/(2*np.linalg.norm(sqrt_cov_w_t@(L1_t+2*L2_t_sym@mean_w_t),ord = 2)**2)
        exponent2_1 = (epsilon2**2)/(np.linalg.norm(sqrt_cov_w_t@L2_t_sym@sqrt_cov_w_t, ord = 'fro')**2)
        exponent2_2 = (epsilon2)/(np.linalg.norm(sqrt_cov_w_t@L2_t_sym@sqrt_cov_w_t, ord = 2))
        exponent2 = -(1/8)*min(exponent2_1, exponent2_2)
        
        probability_bound = np.exp(exponent1) + np.exp(exponent2)
        
        return probability_bound
        
    def run_dynamics(self):
        
        v_0 = np.random.normal(self.exp_v_0, self.var_v_0**0.5)
        x_0 = np.random.normal(self.exp_x_0, self.var_x_0**0.5)
        
        state_0 = np.array([v_0, x_0]).T
        
        state_trajectory = [state_0]
        control_trajectory = []
        
        for t in range(self.terminal_time-1):
            
            state = state_trajectory[-1]
            control = self.compute_control_input(t, state)
            
            control_trajectory.append(control)
            
            noise = np.array([np.random.normal(0, self.var_z**0.5), np.random.normal(0, self.var_y**0.5)]).T
            
            state_ = self.A@state + self.B@control + noise
            
            state_trajectory.append(state_)
            
        self.state_trajectory = state_trajectory
        self.control_trajectory = control_trajectory
            
    def compute_control_input(self,t,state):
        
        (P, s, c) = self.matrix_trajectory[t+1]
        
        control = - np.linalg.inv(self.R + P).dot(self.G[t].T@state + self.r[t]+ s)
        
        return control
    
    def run_mean_and_covariance_dynamics_under_optimal_control(self):
        
        """
        Calculates m_k and Cov_k, where x_k follows N(m_k, Cov_k) at time k under optimal control
        """
        # Initial distribution
        m0 = np.array([self.exp_v_0, self.exp_x_0])
        cov0 = np.array([[self.var_v_0, self.cov_x_v_0],[self.cov_x_v_0, self.var_x_0]])
        logL0 = 0
        
        mean_states = [m0]
        cov_states = [cov0]
        mean_logLt = [logL0]
        
        e2 = np.array([0,1]).T
                
        for t in range(self.terminal_time-1):
            
            # prev corresponds to t, and next corresponds to t+1 in each iteration
            
            (P, s, c) = self.matrix_trajectory[t+1]
            sv = np.array([self.s_b[t], self.s_c[t]])
            
            m_prev = mean_states[-1]
            cov_prev = cov_states[-1]
            m_logL_prev = mean_logLt[-1]
            
            M = self.A - self.H_inv[t]@self.G[t].T
            m_next = (M)@m_prev - self.H_inv[t]@s
            cov_next = (M)@(cov_prev + np.outer(m_prev,m_prev.T))@(M.T) \
                - np.outer((M)@m_prev,(s.T@self.H_inv[t])) - np.outer((self.H_inv[t]@s),(m_prev.T@(M.T)))\
                + np.outer((self.H_inv[t]@s),(s.T@self.H_inv[t])) + np.array([[self.var_z, 0],[0, self.var_y]]) - np.outer(m_next, m_next.T)
            m_logL_next = m_logL_prev - (1/(2*self.var_y))*((2*(e2.T@self.H_inv[t]@s)*(sv.T@m_prev)+ np.trace((np.outer(sv,sv.T)+2*np.outer(self.G[t]@self.H_inv[t]@e2,sv.T))@(cov_prev + np.outer(m_prev, m_prev.T)))))
                
            mean_states.append(m_next)
            cov_states.append(cov_next)
            mean_logLt.append(m_logL_next)
            
        self.mean_states = np.array(mean_states).T  # (num_states = 2, len_traj)
        self.cov_states = np.array(cov_states).T  
        self.mean_logLt = np.array(mean_logLt)
        
        
    
    def calculate_primary_cost(self, state_trajectory, control_trajectory):
        
        cost = 0
        for t in range(self.terminal_time-1):
            control = control_trajectory[:,t]
            state = state_trajectory[:,t]
            cost += control@self.R@control +self.r_v/2*(state[0] - self.bar_v[t])**2
            
        # Terminal cost
        cost += self.t_v/2*(state_trajectory[:,-1][0] - self.bar_v_T)**2
        
        return cost
        
        
    def induction_step(self, matrix_set_, t):
        
        (P, s, c) = matrix_set_  # At time t+1, coming from backward induction
        
        s_b_t = self.s_b[t]
        s_c_t = self.s_c[t]
        s_d_t = self.s_d[t]
        bar_v_t = self.bar_v[t]
        
        self.Q[t] = np.array([[self.r_v+ self.deception_parameter*(s_b_t**2)/self.var_y,self.deception_parameter*(s_b_t*s_c_t)/self.var_y],
                       [self.deception_parameter*(s_b_t*s_c_t)/self.var_y,self.deception_parameter*(s_c_t**2)/self.var_y]])
        self.N[t] = np.array([[0,-self.deception_parameter*(s_b_t)/self.var_y],[0,-self.deception_parameter*(s_c_t)/self.var_y]])
        self.q[t] = np.array([-self.r_v*bar_v_t + self.deception_parameter*(s_b_t*s_d_t)/self.var_y, self.deception_parameter*(s_c_t*s_d_t)/self.var_y]).T
        self.r[t] = np.array([0, -self.deception_parameter/self.var_y*s_d_t])
        self.d[t] = 0.5*self.r_v*(bar_v_t**2) + 0.5*self.deception_parameter/self.var_y*(s_d_t**2)

        self.M[t] = self.Q[t] + self.A.T@P@self.A
        self.H[t] = self.R + self.B.T@P@self.B
        self.G[t] = self.N[t] + self.A.T@P@self.B
        self.H_inv[t] = np.linalg.inv(self.H[t])
        
        new_P = self.M[t] - self.G[t]@self.H_inv[t]@self.G[t].T
        new_s = self.q[t] + self.A.T@s - self.G[t]@self.H_inv[t]@(self.r[t]+self.B.T@s)
        new_c = self.d[t] + c - 0.5*((self.r[t]+self.B.T@s).T@self.H_inv[t]@(self.r[t]+self.B.T@s))
        
                
        matrix_set = (new_P, new_s, new_c)
        
        return matrix_set
    
    def calculate_discrete_dynamics(self):
        
        # Terminal condition
        P_T = np.array([[self.t_v, 0],[0,0]])
        s_T = np.array([-self.t_v*self.bar_v_T, 0]).T
        c_T = 0.5*self.t_v*(self.bar_v_T**2)
        
        matrix_trajectory = [(P_T, s_T, c_T)]
        
        for t in range(self.terminal_time-1, -1, -1): 
            
            matrix_set_ = matrix_trajectory[-1]
            matrix_set = self.induction_step(matrix_set_=matrix_set_, t = t)
            matrix_trajectory.append(matrix_set)
            
        self.matrix_trajectory = matrix_trajectory[::-1]
        
    def calculate_variable_trajectory(self):
        
        self.variable_trajectory = []
        
        for t in range(self.terminal_time):
            
            (P, s, c) = self.matrix_trajectory[t]
            
            variable_set = [P[0,0], 0.5*(P[0,1]+P[1,0]), P[1,1], s[0], s[1], c]
            
            self.variable_trajectory.append(variable_set)
            

class RedTeamSHT():
    def __init__(self, blue_team):
        # Time horizon
        self.terminal_time = blue_team.terminal_time
        
        # Blue team's dynamics info
        self.var_z = blue_team.var_z
        self.var_y = blue_team.var_y
        
        # Blue team's deception info
        self.s_b = blue_team.s_b
        self.s_c = blue_team.s_c
        self.s_d = blue_team.s_d
        
        # Probability of error
        self.alpha = 0.01
        self.beta = 0.01
        
        # Testing parameters
        self.A = (1-self.beta)/self.alpha
        self.B = self.beta/(1-self.alpha)
        
    def calculate_L_T(self, observed_trajectory):
        
        log_L_T = 0
        # Shape of observed trajectory = (num_traj, 2, len_traj)
        
        for i in range(observed_trajectory.shape[0]):
            v_prev = observed_trajectory[i,0,0]
            x_prev = observed_trajectory[i,1,0]
            for t in range(1,observed_trajectory.shape[2]):
                v = observed_trajectory[i,0,t]
                x = observed_trajectory[i,1,t]
                s_b_t = self.s_b[t-1]
                s_c_t = self.s_c[t-1]
                s_d_t = self.s_d[t-1]
                
                log_L_T  += (1/(2*self.var_y))*((x - v_prev - x_prev)**2 - (x - v_prev - x_prev - s_b_t*v_prev - s_c_t*x_prev - s_d_t)**2)
                
                v_prev = v
                x_prev = x
                
        return log_L_T/observed_trajectory.shape[0]

    def sequential_hypothesis_test(self, observed_trajectory):
        
        # Shape of observed trajectory = (num_traj, 2, len_traj)
        
        num_traj = observed_trajectory.shape[0]
        len_traj = observed_trajectory.shape[2]
        
        sequential_L_T = np.zeros((num_traj, len_traj))
        
        for i in range(num_traj):
            v_prev = observed_trajectory[i,0,0]
            x_prev = observed_trajectory[i,1,0]
            
            for t in range(1, len_traj):
                v = observed_trajectory[i,0,t]
                x = observed_trajectory[i,1,t]
                s_b_t = self.s_b[t-1]
                s_c_t = self.s_c[t-1]
                s_d_t = self.s_d[t-1]
                
                sequential_L_T[i,t] = sequential_L_T[i,t-1] + (1/(2*self.var_y))*((x - v_prev - x_prev)**2 - (x - v_prev - x_prev - s_b_t*v_prev - s_c_t*x_prev - s_d_t)**2)
                
                v_prev = v
                x_prev = x
        
        return sequential_L_T

if __name__ == '__main__':
    
    lambda_list = [0.25] #np.arange(0,0.51,0.05)
    num_traj = 30
    len_traj =20
    
    # fig, axs = plt.subplots(2,2)
    
    red_team_L_T_list = []
    primary_cost_list =[]
    mean_velocity_distance_list = []
    
    for l in range(len(lambda_list)):
        state_traj_list = []
        control_traj_list = []
        deceptive_beta_list = []
        velocity_distance_list = []

        logLt_list = []
        
        for j in range(num_traj):
            blue_team = BlueTeamMatrix(deception_parameter = lambda_list[l], terminal_time=len_traj)
            blue_team.calculate_discrete_dynamics()
            blue_team.run_dynamics()
            blue_team.calculate_variable_trajectory()
            blue_team.run_mean_and_covariance_dynamics_under_optimal_control()
            blue_team.compute_C_D_matrices()
            blue_team.compute_L_matrices()
        
            state_trajectory = np.array(blue_team.state_trajectory).T
            control_trajectory = np.array(blue_team.control_trajectory).T
            
            state_traj_list.append(state_trajectory)
            control_traj_list.append(control_trajectory)
            
            velocity_distance = np.sum((state_trajectory[0,:] - blue_team.bar_v)@(state_trajectory[0,:] - blue_team.bar_v))
            velocity_distance_list.append(velocity_distance)
            
            deceptive_beta = blue_team.s_b*state_trajectory[0,:] + blue_team.s_c*state_trajectory[1,:] + blue_team.s_d
            deceptive_beta_list.append(deceptive_beta)
            
            # Check L matrices     
            cov_state0 = np.array([[blue_team.var_v_0, blue_team.cov_x_v_0],[blue_team.cov_x_v_0, blue_team.var_x_0]])
            cov_noise = np.array([[blue_team.var_z,0],[0,blue_team.var_y]])
            cov_w = sp.linalg.block_diag(cov_state0, *([cov_noise]*len_traj))
            mean_w = np.zeros(2*(len_traj+1))
            mean_w[0] = blue_team.exp_v_0
            mean_w[1] = blue_team.exp_x_0
            w = np.random.multivariate_normal(mean_w, cov_w)
            logLt = []
            for t in range(len_traj):
                logLt.append(blue_team.L0_matrices[t] + blue_team.L1_matrices[t]@w + w.T@blue_team.L2_sym_matrices[t]@w)
            logLt_list.append(logLt)
            
        
        # print(np.array(state_traj_list).shape) : (num_traj, num_states = 2, len_traj)
        mean_state_traj = np.mean(np.array(state_traj_list), axis = 0)    # This is estimated m*_k's (Mean of x_k's)
        cov_state_traj = np.zeros((2,2,len_traj))  # This is estimated \Sigma*_k's (Covariance matrix of x_k's)
        for k in range(len_traj):
            cov_state_traj[:,:,k] = np.cov(np.array(state_traj_list)[:,:,k], rowvar=False)
        mean_control_traj = np.mean(np.array(control_traj_list), axis = 0)
        
        mean_velocity_distance_list.append(np.mean(np.array(velocity_distance_list)))
        
        
        # plt.plot(mean_state_traj[0,1:], label = "estimated mean v")
        # plt.plot(blue_team.mean_states[0,1:],'--', label = "calculated mean v")
        # # print(blue_team.C_matrices[0,:,:].shape)
        # # print(np.dot(blue_team.C_matrices[0,:,:],np.array([2,4])).shape)
        # # print(blue_team.D_matrices.shape)
        # plt.plot((blue_team.D_matrices + np.dot(blue_team.C_matrices[0,:,:],np.array([2,4])))[:,0],':', label = "D0")
        # plt.plot(mean_state_traj[1,1:], label = "estimated mean x")
        # plt.plot(blue_team.mean_states[1,1:], '--', label = "calculated mean x")
        # plt.plot((blue_team.D_matrices + np.dot(blue_team.C_matrices[0,:,:],np.array([2,4])))[:,1],':', label = "D1")
        # plt.legend()
        # plt.show()
        # plt.plot(cov_state_traj[0,0,:], label = "estimated var_v")
        # plt.plot(blue_team.cov_states[0,0,:], '--', label = "calculated var_v")
        # plt.plot(cov_state_traj[0,1,:], label = "estimated cov_vx")
        # plt.plot(blue_team.cov_states[0,1,:], '--', label = "calculated cov_vx")
        # plt.legend()
        # plt.show()
        
        # plt.plot(cov_state_traj[1,1,:], label = "estimated var_x")
        # plt.plot(blue_team.cov_states[1,1,:], '--', label = "calculated var_x")
        # plt.legend()
        # plt.show()
        
        # x_id = int(l/2)
        # y_id = int(l%2)
        # axs[x_id,y_id].plot(state_trajectory[0,:], label = "v(velocity)")
        # # axs[x_id,y_id].plot(state_trajectory[1,:], label = "x(position)")
        # axs[x_id,y_id].plot(control_trajectory[0,:], label = "alpha(velocity control)")
        # axs[x_id,y_id].plot(control_trajectory[1,:], label = "beta(position control)")
        # axs[x_id,y_id].plot(blue_team.s_c*state_trajectory[1,:], label = "deceptive_beta")
        
        # axs[x_id,y_id].plot(blue_team.bar_v, label = "reference v")
        # axs[x_id,y_id].plot(mean_state_traj[0,:], label = "v(velocity)")
        # axs[x_id,y_id].plot(mean_state_traj[1,:], label = "x(position)")
        # axs[x_id,y_id].plot(mean_control_traj[0,:], label = "alpha(velocity control)")
        # axs[x_id,y_id].plot(mean_control_traj[1,:], label = "beta(position control)")
        # axs[x_id,y_id].plot(np.mean(deceptive_beta_list, axis = 0), label = "deceptive_beta")
        
        # axs[x_id,y_id].set_title("Deception parameter:"+ str(lambda_list[l]))
        
        # cmap = plt.get_cmap('plasma')
        # if l == 0:
        #     axs[0,0].plot(blue_team.bar_v,'--', label = "reference v", c = 'black')
        #     # axs[0,0].set_xlabel("Time, t")
        #     axs[0,0].set_ylabel("Velocity, v")
        #     # axs[0,1].set_xlabel("Time, t")
        #     axs[0,1].set_ylabel("Position, x")
        #     axs[1,0].set_xlabel("Time, t")
        #     axs[1,0].set_ylabel(r"Velocity Control, $\alpha$")
        #     axs[1,1].set_xlabel("Time, t")
        #     axs[1,1].set_ylabel(r"Position Control, $\beta$")
        # axs[0,0].plot(mean_state_traj[0,:], label = "$\lambda$="+str(np.round(lambda_list[l],2)), color = cmap(l/len(lambda_list)))
        # axs[0,1].plot(mean_state_traj[1,:], label = "$\lambda$="+str(np.round(lambda_list[l],2)), color = cmap(l/len(lambda_list)))
        # axs[1,0].plot(mean_control_traj[0,:], label = "$\lambda$="+str(np.round(lambda_list[l],2)), color = cmap(l/len(lambda_list)))
        # axs[1,1].plot(mean_control_traj[1,:], label = "$\lambda$="+str(np.round(lambda_list[l],2)), color = cmap(l/len(lambda_list)))
    
        
    #     # Red Team's SHT
        red_team = RedTeamSHT(blue_team)
        
        red_team_observed_trajectory = np.array(state_traj_list) #(num_traj,2, len_traj)
    #     # print(red_team_observed_trajectory.shape)
        red_team_L_T = red_team.calculate_L_T(red_team_observed_trajectory)
    #     red_team_L_T_list.append(red_team_L_T)
        
        print(red_team_L_T)
        
    #     primary_cost = blue_team.calculate_primary_cost(mean_state_traj, mean_control_traj)
    #     primary_cost_list.append(primary_cost)
        
        red_team_sequential_L_T = red_team.sequential_hypothesis_test(red_team_observed_trajectory) #(num_traj, len_traj)  # Samples of Log(L_t)
        
        plt.plot(np.mean(red_team_sequential_L_T, axis=0), label = "estimated mean logLt")
        plt.plot(blue_team.mean_logLt, '--', label = "calculated mean logLt")
        print(np.array(logLt_list).shape)
        plt.plot(np.mean(np.array(logLt_list), axis = 0), label = "from L matrices")
        plt.legend()
        plt.show()
        
        
        # log_A = np.log(red_team.A)*np.ones(red_team_sequential_L_T.shape[1])
        # log_B = np.log(red_team.B)*np.ones(red_team_sequential_L_T.shape[1])
        
        theorem2_bounds = []
        for epsilon1 in np.linspace(0.1,np.log(red_team.A)-0.1,num=10):
            theorem2_bound = []
            for t in range(len_traj):
                theorem2_bound.append(blue_team.get_theorem_2_bound(t,epsilon1,np.log(red_team.A)))
            theorem2_bounds.append(theorem2_bound)
            plt.plot(theorem2_bound, label = epsilon1)
        theorem2_bounds = np.array(theorem2_bounds)
        plt.legend()
        plt.show()
                
                
    #     # axs[x_idy_id].plot(log_B, '--', c='gray')
        
    #     # plt.plot(np.mean(red_team_sequential_L_T, axis=0), label = "$\lambda$="+str(np.round(lambda_list[l],2)), color = cmap(l/len(lambda_list)))
    #     # plt.plot(log_A, '--', c='gray')
    #     # plt.plot(log_B, '--', c='gray')
    #     # plt.xlabel("Time, t")
    #     # plt.ylabel("$\log L_t$")
    #     # plt.title("Red Team's Sequential Hypothesis Test")
        
    # plt.plot(lambda_list, mean_velocity_distance_list, color = cmap(0))
    # plt.xlabel("$\lambda$")
    # plt.ylabel("Distance, $d$")
    # plt.title("Mean distance between observed and reference trajectory")
        
    # plt.legend()
    # # fig.suptitle("State and Control Input Trajectories")
    # # fig.suptitle("Red Team's Sequential Hypothesis Test")
    # # plt.savefig("state_control_traj.png")
    # plt.savefig("sht.png")
    # plt.show()
    
    # plt.plot(np.log(lambda_list), red_team_L_T_list)
    # plt.title("Result of Red Team's SHT")
    # plt.xlabel("$\log\lambda$")
    # plt.ylabel("$\log L_T$")
    # plt.show()
    # plt.plot(np.log(lambda_list), primary_cost_list)
    # plt.show()
    # plt.plot(np.log(lambda_list), primary_cost_list - np.array(lambda_list)*red_team_L_T_list)
    # plt.show()
    
    # fig, axs = plt.subplots(2,3)
    
    # for l in range(len(lambda_list)):
    
    #     blue_team = BlueTeamMatrix(deception_parameter = lambda_list[l])
    #     blue_team.calculate_discrete_dynamics()
    #     blue_team.run_dynamics()
    #     blue_team.calculate_variable_trajectory()
        
    #     # print(np.array(blue_team.matrix_trajectory).shape)
        
    #     variable_trajectory = np.array(blue_team.variable_trajectory).T
        
    #     variable_name = ["mu", "eta", "rho", "gamma", "theta", "xi"]
            
    #     x_id = int(l/3)
    #     y_id = int(l%3)
        
    #     for i in range(6):
    #         axs[x_id,y_id].plot(variable_trajectory[i,:], label = variable_name[i])
            
    #     axs[x_id,y_id].set_title("Deception parameter:"+ str(lambda_list[l]))
        
    # plt.legend()
    # fig.suptitle("Internal Variable Trajectories")
    # plt.savefig("variables_traj.png")
    # plt.show()
    
    # blue_team = BlueTeamMatrix(deception_parameter = 0, terminal_time=len_traj)
    # blue_team.calculate_discrete_dynamics()
    # blue_team.run_dynamics()
    # blue_team.calculate_variable_trajectory()

    # non_deceptive_state_trajectory = np.array(blue_team.state_trajectory).T
    # non_deceptive_control_trajectory = np.array(blue_team.control_trajectory).T
    # # deceptive_beta = blue_team.s_b*state_trajectory[0,:] + blue_team.s_c*state_trajectory[1,:] + blue_team.s_d
    
    # blue_team = BlueTeamMatrix(deception_parameter = 0.5, terminal_time=len_traj)
    # blue_team.calculate_discrete_dynamics()
    # blue_team.run_dynamics()
    # blue_team.calculate_variable_trajectory()

    # state_trajectory = np.array(blue_team.state_trajectory).T
    # control_trajectory = np.array(blue_team.control_trajectory).T
    # deceptive_beta = blue_team.s_b*state_trajectory[0,:] + blue_team.s_c*state_trajectory[1,:] + blue_team.s_d
    
    # # plt.plot(blue_team.bar_v, label = "reference v")
    # # plt.plot(state_trajectory[0,:], label = "v(velocity)")
    # # plt.plot(state_trajectory[1,:], label = "x(position)")
    # plt.plot(control_trajectory[0,:], label = "alpha(velocity control)")
    # plt.plot(control_trajectory[1,:], label = "beta(position control)")
    # plt.plot(non_deceptive_control_trajectory[0,:], label = "non_deceptive_alpha(velocity control)")
    # plt.plot(non_deceptive_control_trajectory[1,:], label = "non_deceptive_beta(position control)")
    # plt.plot(deceptive_beta,'--', label = "deceptive_beta")
    # plt.legend()
    # plt.savefig("test.png")
    # plt.show()
    
    # print(control_trajectory[1,:])
    # print(deceptive_beta)