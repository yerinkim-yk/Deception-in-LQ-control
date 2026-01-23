import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm

class BlueTeamMatrix():
    """
    Attributes:
    
    Functions:
    
    """
    
    def __init__(self, var_z = 0.05, var_y = 0.05, deception_parameter = 0, terminal_time = 30):
        
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
        ## Misdirection Pattern
        self.s_b = [] # 0.2*np.ones(self.terminal_time) #0.2*np.sin(np.linspace(0, self.terminal_time,self.terminal_time))
        self.s_c = [] #-0.2*np.ones(self.terminal_time) #0.2*np.sin(np.linspace(0, self.terminal_time,self.terminal_time))
        self.s_d = 0*np.ones(self.terminal_time) #2*np.sin(np.linspace(0, self.terminal_time,self.terminal_time))
        
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
    
    def backward_recursion(self):
        
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
        
    
    def compute_control_input(self,t,state):
        
        (P, s, c) = self.matrix_trajectory[t+1]
        
        control = - np.linalg.inv(self.R + P).dot(self.G[t].T@state + self.r[t]+ s)
        
        return control
        
    def run_dynamics(self, save = True):
        
        if len(self.matrix_trajectory) == 0:
            self.backward_recursion()
            
            
        v_0 = np.random.normal(self.exp_v_0, self.var_v_0**0.5)
        x_0 = np.random.normal(self.exp_x_0, self.var_x_0**0.5)
        
        state_0 = np.array([v_0, x_0]).T
        
        state_trajectory = [state_0]
        control_trajectory = []
        
        for t in range(self.terminal_time-1):
            
            state = state_trajectory[-1]
            control = self.compute_control_input(t, state)
            
            control_trajectory.append(control)
            
            np.random.seed(t+200)
            noise = np.array([np.random.normal(0, self.var_z**0.5), np.random.normal(0, self.var_y**0.5)]).T
            
            state_ = self.A@state + self.B@control + noise
            
            state_trajectory.append(state_)
            
        if save:
            self.state_trajectory = state_trajectory
            self.control_trajectory = control_trajectory
        
        return state_trajectory, control_trajectory
            
      
    def calculate_primary_cost(self, state_trajectory, control_trajectory):
        
        cost = 0
        for t in range(self.terminal_time-1):
            control = control_trajectory[:,t]
            state = state_trajectory[:,t]
            cost += control@self.R@control +self.r_v/2*(state[0] - self.bar_v[t])**2
            
        # Terminal cost
        cost += self.t_v/2*(state_trajectory[:,-1][0] - self.bar_v_T)**2
        
        return cost
    
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
           
    def calculate_sampling_based_probability(self, log_A, sample_size = 100000):
        
        def calculate_L_t(state_trajectory):
        
            log_L_t = np.zeros(self.terminal_time)
            state_trajectory = np.array(state_trajectory)
            # Shape of state trajectory = (len_traj, 2)
            
            v_prev = state_trajectory[0,0]
            x_prev = state_trajectory[0,1]
            for t in range(1,self.terminal_time):
                v = state_trajectory[t,0]
                x = state_trajectory[t,1]
                s_b_t = self.s_b[t-1]
                s_c_t = self.s_c[t-1]
                s_d_t = self.s_d[t-1]
                
                log_L_t[t]  = log_L_t[t-1] + (1/(2*self.var_y))*((x - v_prev - x_prev)**2 - (x - v_prev - x_prev - s_b_t*v_prev - s_c_t*x_prev - s_d_t)**2)
                
                v_prev = v
                x_prev = x
                    
            return log_L_t
        
        logLt_list = []
                
        for i in tqdm(range(sample_size)):
            state_trajectory, _ = self.run_dynamics(save = False)
            logLt = calculate_L_t(state_trajectory)
            logLt_list.append(logLt)
            
        logLt_list = np.array(logLt_list)  # (sample_size, len_traj)
        
        detection = np.where(logLt_list>=log_A, 1, 0)
        detection_probability = np.sum(detection, axis = 0)/sample_size
        
        print(np.sum(detection))
        
        return detection_probability # Probability at each time, shape: (len_traj)
    
    
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
    
        
    def get_theorem_2_bounds(self, log_A, optimize=True):
        
    
        def get_bound_at_single_time(t, epsilon1, epsilon2):
            
            assert epsilon1>=0
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
        
        if optimize:
            bound_candidates = []
            for i, epsilon1 in enumerate(np.linspace(0.001*np.ones(self.mean_logLt.shape),0.999*np.maximum(0,log_A- self.mean_logLt),num=50)):
                theorem2_bound = []
                for t in range(self.terminal_time):
                    epsilon2 = log_A - self.mean_logLt[t] - epsilon1[t]
                    bound = get_bound_at_single_time(t,epsilon1[t],epsilon2)
                    theorem2_bound.append(bound)
                bound_candidates.append(theorem2_bound)
            bound_candidates = np.array(bound_candidates) #(num, len_traj)
            theorem2_bounds = np.min(bound_candidates, axis = 0)
        else:
            epsilon1 = np.maximum(0,np.log(self.A)- self.mean_logLt)*0.6
            epsilon2 = log_A - self.mean_logLt[t] - epsilon1
            theorem2_bounds = []
            for t in range(self.terminal_time):
                bound = get_bound_at_single_time(t,epsilon1,epsilon2)
                theorem2_bounds.append(bound)
        
        return theorem2_bounds
            

class RedTeamSHT():
    def __init__(self, blue_team, var_z = 0.05, var_y = 0.05, alpha = 0.05, beta=0.05):
        
        # Blue team's dynamics info
        self.var_z = var_z
        self.var_y = var_y
        
        # Time horizon
        self.terminal_time = blue_team.terminal_time
        
        # Blue team's deception info
        self.s_b = blue_team.s_b
        self.s_c = blue_team.s_c
        self.s_d = blue_team.s_d
        
        # Probability of error
        self.alpha = alpha
        self.beta = beta
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
        
        sequential_L_t = np.zeros((num_traj, len_traj))
        
        for i in range(num_traj):
            v_prev = observed_trajectory[i,0,0]
            x_prev = observed_trajectory[i,1,0]
            
            for t in range(1, len_traj):
                v = observed_trajectory[i,0,t]
                x = observed_trajectory[i,1,t]
                s_b_t = self.s_b[t-1]
                s_c_t = self.s_c[t-1]
                s_d_t = self.s_d[t-1]
                
                sequential_L_t[i,t] = sequential_L_t[i,t-1] + (1/(2*self.var_y))*((x - v_prev - x_prev)**2 - (x - v_prev - x_prev - s_b_t*v_prev - s_c_t*x_prev - s_d_t)**2)
                
                v_prev = v
                x_prev = x
        
        return sequential_L_t

if __name__ == '__main__':
    
    # Overall settings
    var_z = 0.05
    var_y = 0.05
    lambda_list = np.arange(0.0,0.51,0.1)
    num_traj = 1
    len_traj = 21
    s_b_value = 0.5
    s_c_value = -0.1
    alpha_error = 0.01
    beta_error = 0.01
    log_A = np.log((1-beta_error)/alpha_error)
    sample_size = 20000
    
    # Run everything before plot # Only Blue team side for now
    def run_experiments_with_fixed_lambda(num_traj, len_traj, s_b_value, s_c_value, deception_parameter, 
                                       misdirection_b = "standard", misdirection_c = "standard"):
        
        state_traj_list = []
        control_traj_list = []
        primary_cost_list = []
        
        blue_team = BlueTeamMatrix(var_z = var_z, 
                                    var_y = var_y, 
                                    deception_parameter = deception_parameter, 
                                    terminal_time=len_traj)
        
        if misdirection_b == "standard":
            blue_team.s_b = s_b_value*np.ones(len_traj)
        elif misdirection_b == "negative":
            blue_team.s_b = -s_b_value*np.ones(len_traj)
        elif misdirection_b == "sinusoidal":
            blue_team.s_b = s_b_value*np.sin(np.linspace(0, len_traj, len_traj))
        else:
            blue_team.s_b = s_b_value*np.ones(len_traj)
        
        if misdirection_c == "standard":
            blue_team.s_c = s_c_value*np.ones(len_traj)
        elif misdirection_c == "negative":
            blue_team.s_c = -s_c_value*np.ones(len_traj)
        elif misdirection_c == "sinusoidal":
            blue_team.s_c = s_c_value*np.sin(np.linspace(0, len_traj, len_traj))
        else:
            blue_team.s_c = s_c_value*np.ones(len_traj)
        
        blue_team.set_matrices()
        
        red_team = RedTeamSHT(blue_team, var_z = var_z, var_y = var_y, alpha = alpha_error, beta=beta_error)
        
        for j in tqdm(range(num_traj)):
            
            blue_team.run_dynamics()
            
            state_trajectory = np.array(blue_team.state_trajectory).T
            control_trajectory = np.array(blue_team.control_trajectory).T
            state_traj_list.append(state_trajectory)
            control_traj_list.append(control_trajectory)
                        
            primary_cost = blue_team.calculate_primary_cost(state_trajectory, control_trajectory)
            primary_cost_list.append(primary_cost)
            
        sht_results = red_team.sequential_hypothesis_test(np.array(state_traj_list))
        
        mean_state_traj = np.mean(np.array(state_traj_list), axis = 0)    # This is estimated m*_k's (Mean of x_k's)
        # cov_state_traj = np.zeros((2,2,len_traj))  # This is estimated \Sigma*_k's (Covariance matrix of x_k's)
        # for k in range(len_traj):
        #     cov_state_traj[:,:,k] = np.cov(np.array(state_traj_list)[:,:,k], rowvar=False)
        mean_control_traj = np.mean(np.array(control_traj_list), axis = 0)
        mean_primary_cost = np.mean(np.array(primary_cost_list))
        mean_log_L_t = blue_team.mean_logLt
        
        
        # bound_for_detection_probabilty = blue_team.get_theorem_2_bounds(log_A = log_A, optimize = True)
        # approximated_detection_probability = blue_team.calculate_sampling_based_probability(log_A = log_A, sample_size=sample_size)
        
        result = {
            "state_trajectories": np.array(state_traj_list),
            "control_trajectories": np.array(control_traj_list),
            "mean_state_trajectory": mean_state_traj, #(len_traj,2)
            "mean_control_trajectory": mean_control_traj, #(len_traj,2)
            "mean_primary_cost": mean_primary_cost, # ()
            "mean_log_L_t" : mean_log_L_t, #(len_traj,)
            "SHT_results": sht_results,
            # "approximated_probability" : approximated_detection_probability, #(len_traj,)
            # "bound_for_probability": bound_for_detection_probabilty #(len_traj,)
        }
                        
        return result
        
    
    
    # Introducing Misdirection
    
    def plot_trajectories_with_lambda(results_with_lambda):
        
        fig, axs = plt.subplots(2,2)
        lambda_list = results_with_lambda.keys()
        
        for i, l in enumerate(lambda_list):
            
            mean_state_traj = results_with_lambda[l]["mean_state_trajectory"]
            mean_control_traj = results_with_lambda[l]["mean_control_trajectory"]
            
            if i == 0:
                # axs[0,0].plot(blue_team.bar_v,'--', label = "reference v", c = 'black')
                # axs[0,0].set_xlabel("Time, t")
                axs[0,0].set_ylabel("Velocity, v")
                # axs[0,1].set_xlabel("Time, t")
                axs[0,1].set_ylabel("Position, x")
                axs[1,0].set_xlabel("Time, t")
                axs[1,0].set_ylabel(r"Velocity Control, $\alpha$")
                axs[1,1].set_xlabel("Time, t")
                axs[1,1].set_ylabel(r"Position Control, $\beta$")
            axs[0,0].plot(mean_state_traj[0,:], label = "$\lambda$="+str(l), color = cmap(i/(1.25*len(lambda_list)-1)))
            axs[0,1].plot(mean_state_traj[1,:], label = "$\lambda$="+str(l), color = cmap(i/(1.25*len(lambda_list)-1)))
            axs[1,0].plot(mean_control_traj[0,:], label = "$\lambda$="+str(l), color = cmap(i/(1.25*len(lambda_list)-1)))
            axs[1,1].plot(mean_control_traj[1,:], label = "$\lambda$="+str(l), color = cmap(i/(1.25*len(lambda_list)-1)))
        
        fig.suptitle("Trajectories with Misdirection")
        plt.legend()
        plt.show()
        
            
    def plot_all_trajectories_with_lambda(results_with_lambda):
        
        fig, axs = plt.subplots(2,2)
        lambda_list = results_with_lambda.keys()
        
        for i, l in enumerate(lambda_list):
            
            state_traj = results_with_lambda[l]["state_trajectories"]
            control_traj = results_with_lambda[l]["control_trajectories"]
            
            if i == 0:
                # axs[0,0].plot(blue_team.bar_v,'--', label = "reference v", c = 'black')
                # axs[0,0].set_xlabel("Time, t")
                axs[0,0].set_ylabel("Velocity, $v$")
                # axs[0,1].set_xlabel("Time, t")
                axs[0,1].set_ylabel("Position, $x$")
                axs[1,0].set_xlabel("Time, $t$")
                axs[1,0].set_ylabel(r"Velocity Control, $\alpha$")
                axs[1,1].set_xlabel("Time, $t$")
                axs[1,1].set_ylabel(r"Position Control, $\beta$")
            for j in range(state_traj.shape[0]):
                axs[0,0].plot(state_traj[j,0,:], label = "$\lambda$="+str(l), color = cmap(i/(1.25*len(lambda_list)-1)))
                axs[0,1].plot(state_traj[j,1,:], label = "$\lambda$="+str(l), color = cmap(i/(1.25*len(lambda_list)-1)))
                axs[1,0].plot(control_traj[j,0,:], label = "$\lambda$="+str(l), color = cmap(i/(1.25*len(lambda_list)-1)))
                axs[1,1].plot(control_traj[j,1,:], label = "$\lambda$="+str(l), color = cmap(i/(1.25*len(lambda_list)-1)))
        
        fig.suptitle("Trajectories with Misdirection")
        plt.legend()
        plt.show()
        
    def plot_primary_cost_with_lambda(results_with_lambda):
        
        primary_cost_list =[]
        lambda_list = list(results_with_lambda.keys())
        
        for i, l in enumerate(lambda_list):
            primary_cost_list.append(results_with_lambda[l]["mean_primary_cost"])
            
        print(lambda_list)
            
        ### Bar Plot ###
        fig, ax = plt.subplots()
        bar_container = ax.bar(lambda_list, primary_cost_list, width=0.05, color = cmap(0))
        ax.set(ylabel='pints sold', title='Gelato sales by flavor')
        ax.bar_label(bar_container, fmt='{:,.1f}')
            
        ### Line Plot ###
        # plt.plot(lambda_list, primary_cost_list, marker = '^', color = cmap(0))  
        # for i in range(len(lambda_list)):
        #     plt.text(lambda_list[i], primary_cost_list[i] + 0.3, f'${primary_cost_list[i]:.1f}$', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel("Deception parameter, $\lambda$")
        plt.ylabel("Primary cost")
        plt.title("Primary Cost")
        plt.show()
    
    def plot_E_log_L_T_with_lambda(results_with_lambda, log_A):
        
        E_log_L_T_list =[]
        lambda_list = results_with_lambda.keys()
        
        for i, l in enumerate(lambda_list):
            E_log_L_T_list.append(results_with_lambda[l]["mean_log_L_t"][-1])
            
        plt.plot(lambda_list, E_log_L_T_list, c = cmap(0))  
        plt.plot(lambda_list, log_A*np.ones(len(lambda_list)), '--', c = 'gray')
        plt.xlabel("Deception parameter, $\lambda$")
        plt.ylabel("$E[\log L^*_T]$")
        plt.title("Expected value of $\log L^*_T$")
        plt.show()
    
    
    def plot_sht_results_with_lambda(results_with_lambda, log_A):
        
        lambda_list = results_with_lambda.keys()
        
        for i, l in enumerate(lambda_list):
            
            sht_results = results_with_lambda[l]["SHT_results"]
            mean_sht_results = np.mean(sht_results, axis = 0)
            
            plt.plot(mean_sht_results, label = "$\lambda$="+str(l), color = cmap(i/(1.25*len(lambda_list)-1)))
            
        plt.plot(log_A*np.ones(mean_sht_results.shape[0]), '--', c = 'gray', label = '$\log A$')
        plt.xlabel("Time, $t$")
        plt.xticks(range(0,len_traj,2))
        plt.ylabel("$\log L^*_t$")
        plt.title("$\log L^*_t$ with Observed Trajectory")
        plt.legend()
        plt.show()
    
    # Stealthyness of Misdirection
    
    def plot_E_log_L_t_with_lambda(results_with_lambda, log_A):
        
        lambda_list = results_with_lambda.keys()
        
        for i, l in enumerate(lambda_list):
            
            plt.plot(results_with_lambda[l]["mean_log_L_t"], label = "$\lambda$="+str(l), color = cmap(i/(1.25*len(lambda_list)-1)))
            
        plt.plot(log_A*np.ones(len_traj), '--', c = 'gray', label = '$\log A$')
        plt.xlabel("Time, $t$")
        plt.xticks(range(0,len_traj,2))
        plt.ylabel("$E[\log L^*_t]$")
        plt.title("Expected value of $\log L^*_t$")
        plt.legend()
        plt.show()
    
    
    def plot_approximated_probability_with_lambda(results_with_lambda):
        
        lambda_list = results_with_lambda.keys()
        
        for i, l in enumerate(lambda_list):
            
            plt.plot(results_with_lambda[l]["approximated_probability"], label = "$\lambda$="+str(l), color = cmap(i/(1.25*len(lambda_list)-1)))
            
        plt.xlabel("Time, $t$")
        plt.xticks(range(0,len_traj,2))
        plt.ylabel("$Approximated probability")
        plt.title("Approximated Probability of Detection")
        plt.legend()
        plt.show()
    
    def plot_probability_bound_with_lambda(results_with_lambda):
        
        lambda_list = results_with_lambda.keys()
        
        for i, l in enumerate(lambda_list):
            
            plt.plot(results_with_lambda[l]["bound_for_probability"], label = "$\lambda$="+str(l), color = cmap(i/(1.25*len(lambda_list)-1)))
            
        plt.xlabel("Time, $t$")
        plt.xticks(range(0,len_traj,2))
        plt.ylabel("Probability bound")
        plt.title("Upper Bound for Probability of Detection")
        plt.legend()
        plt.show()
    
    def compare_primary_cost_and_stealthiness_with_lambda():
        pass
    
    
    results_with_lambda = {}
    
    for l in lambda_list:
        
        l = np.round(l,3)
        result = run_experiments_with_fixed_lambda(num_traj, len_traj, s_b_value, s_c_value, deception_parameter = l)
        results_with_lambda[l] = result
        
    cmap = plt.get_cmap('plasma')
    # plot_trajectories_with_lambda(results_with_lambda)
    # plot_primary_cost_with_lambda(results_with_lambda)
    plot_sht_results_with_lambda(results_with_lambda, log_A)
    # plot_E_log_L_T_with_lambda(results_with_lambda, log_A)
    # plot_E_log_L_t_with_lambda(results_with_lambda, log_A)
    # plot_approximated_probability_with_lambda(results_with_lambda)
    # plot_probability_bound_with_lambda(results_with_lambda)
    
        
    # fig, axs = plt.subplots(2,2)
    
    # for k  in ["positive","negative","sinusoidal"]:
        
    #     results = run_experiments_with_fixed_lambda(num_traj, len_traj, s_b_value, s_c_value, deception_parameter = 0.1, misdirection_c=k, misdirection_b=k)
        
    #     mean_state_traj = results["mean_state_trajectory"]
    #     mean_control_traj = results["mean_control_trajectory"]
            
    #     if k == "positive":
    #         # axs[0,0].plot(blue_team.bar_v,'--', label = "reference v", c = 'black')
    #         # axs[0,0].set_xlabel("Time, t")
    #         axs[0,0].set_ylabel("Velocity, $v$")
    #         # axs[0,1].set_xlabel("Time, t")
    #         axs[0,1].set_ylabel("Position, $x$")
    #         axs[1,0].set_xlabel("Time, t")
    #         axs[1,0].set_ylabel(r"Velocity Control, $\alpha$")
    #         axs[1,1].set_xlabel("Time, t")
    #         axs[1,1].set_ylabel(r"Position Control, $\beta$")
    #     axs[0,0].plot(mean_state_traj[0,:], label = str(k))
    #     axs[0,1].plot(mean_state_traj[1,:], label = str(k))
    #     axs[1,0].plot(mean_control_traj[0,:], label = str(k))
    #     axs[1,1].plot(mean_control_traj[1,:], label = str(k))
        
    # fig.suptitle("Trajectories with Misdirection")
    # plt.legend()
    # plt.show()