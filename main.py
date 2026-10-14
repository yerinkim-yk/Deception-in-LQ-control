import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from blue_team_original import BlueTeam, score_confidence_interval
from red_team_original import RedTeam


# Overall settings
n_dim = 1  # Dimension of v and p (state x = (v, p) is 2*n_dim dimensional)
var_z = 0.05  # Scalar (meaning var_z*I) or (n_dim, n_dim) covariance matrix
var_y = 0.05  # Scalar (meaning var_y*I) or (n_dim, n_dim) covariance matrix
lambda_list = np.arange(0.0,0.51,0.1)
num_traj = 1
terminal_time = 20  # T: controls u_0, ..., u_{T-1}
len_traj = terminal_time + 1  # Number of states x_0, ..., x_T
# Deception pattern matrices F^b, F^c, each (n_dim, n_dim); replace with any matrices
s_b_matrix = 0.5*np.eye(n_dim)
s_c_matrix = -0.1*np.eye(n_dim)
alpha_error = 0.01
beta_error = 0.01
log_U = np.log((1-beta_error)/alpha_error)
sample_size = 200
significance = 0.05  # c in Lemma 2: (1-c) confidence intervals for the sampled detection probabilities
random_noise = False  # False: fixed seed, every lambda sees the same noise sequence (trajectories still differ); True: fresh noise each run

cmap = plt.get_cmap('plasma')


# ---------- Experiments ----------

def make_misdirection_pattern(kind, matrix, length):
    """
    Time-varying pattern built from an (n, n) matrix, returns (length, n, n).
    kind: "standard" (constant), "negative" (negated constant), or "sinusoidal". Unknown kinds fall back to "standard".
    """
    matrix = np.asarray(matrix, dtype=float)

    if kind == "negative":
        return -matrix*np.ones((length, 1, 1))
    if kind == "sinusoidal":
        return np.sin(np.arange(length))[:, None, None]*matrix  # sin(t), t = 0, 1, ..., length-1
    return matrix*np.ones((length, 1, 1))


def run_experiments_with_fixed_lambda(num_traj, terminal_time, s_b_matrix, s_c_matrix, deception_parameter,
                                      misdirection_b = "standard", misdirection_c = "standard",
                                      compute_probabilities = False):

    blue_team = BlueTeam(n_dim = n_dim,
                         var_z = var_z,
                         var_y = var_y,
                         deception_parameter = deception_parameter,
                         terminal_time = terminal_time,
                         random_noise = random_noise)
    blue_team.s_b = make_misdirection_pattern(misdirection_b, s_b_matrix, terminal_time)
    blue_team.s_c = make_misdirection_pattern(misdirection_c, s_c_matrix, terminal_time)
    blue_team.set_matrices()

    red_team = RedTeam(terminal_time = terminal_time, s_b = blue_team.s_b, s_c = blue_team.s_c, s_d = blue_team.s_d,
                       n_dim = n_dim, var_z = var_z, var_y = var_y, alpha = alpha_error, beta = beta_error)

    state_traj_list = []
    control_traj_list = []
    primary_cost_list = []
    deception_measure_list = []

    for j in tqdm(range(num_traj)):

        blue_team.run_dynamics()

        state_trajectory = np.array(blue_team.state_trajectory).T
        control_trajectory = np.array(blue_team.control_trajectory).T
        state_traj_list.append(state_trajectory)
        control_traj_list.append(control_trajectory)
        primary_cost_list.append(blue_team.calculate_primary_cost(state_trajectory, control_trajectory))
        deception_measure_list.append(blue_team.calculate_deception_measure(state_trajectory, control_trajectory))

    state_trajectories = np.array(state_traj_list)
    control_trajectories = np.array(control_traj_list)

    result = {
        "state_trajectories": state_trajectories,
        "control_trajectories": control_trajectories,
        "mean_state_trajectory": np.mean(state_trajectories, axis = 0),  # (2n, len_traj), estimated m*_k's (Mean of x_k's)
        "mean_control_trajectory": np.mean(control_trajectories, axis = 0),  # (2n, len_traj-1) = (2n, T)
        "mean_primary_cost": np.mean(np.array(primary_cost_list)),  # ()
        "mean_deception_measure": np.mean(np.array(deception_measure_list)),  # (), D(lambda), Eq. (15)
        "mean_log_L_t": blue_team.mean_logLt,  # (len_traj,)
        "SHT_results": red_team.sequential_hypothesis_test(state_trajectories),  # (num_traj, len_traj)
    }

    if compute_probabilities:
        result["approximated_probability"] = blue_team.calculate_sampling_based_probability(log_U = log_U, sample_size = sample_size)  # (len_traj,)
        result["approximated_probability_CI"] = np.array(score_confidence_interval(result["approximated_probability"], sample_size, significance))  # (2, len_traj): lower, upper
        result["bound_for_probability"] = blue_team.get_theorem_2_bounds(log_U = log_U, optimize = True)  # (len_traj,)

    return result


def run_experiments_over_lambda(lambda_list, **kwargs):

    results_with_lambda = {}
    for l in lambda_list:
        l = np.round(l,3)
        results_with_lambda[l] = run_experiments_with_fixed_lambda(num_traj, terminal_time, s_b_matrix, s_c_matrix,
                                                                   deception_parameter = l, **kwargs)
    return results_with_lambda


# ---------- Plot helpers ----------

def lambda_color(cmap, i, num_lambda):
    return cmap(i/(1.25*num_lambda-1))


def lambda_label(l):
    return r"$\lambda$=" + str(l)


def create_trajectory_axes(component, n_dim):
    """ 2x2 axes for (v, p, alpha, beta) of one component; labels carry the component index when n_dim > 1 """

    sub = f"_{{{component+1}}}" if n_dim > 1 else ""
    fig, axs = plt.subplots(2,2)
    axs[0,0].set_ylabel(f"Velocity, $v{sub}$")
    axs[0,1].set_ylabel(f"Position, $p{sub}$")
    axs[1,0].set_xlabel("Time, $t$")
    axs[1,0].set_ylabel(rf"Velocity Control, $\alpha{sub}$")
    axs[1,1].set_xlabel("Time, $t$")
    axs[1,1].set_ylabel(rf"Position Control, $\beta{sub}$")

    return fig, axs


def trajectory_title(component, n_dim):
    return "Trajectories with Misdirection" + (f" (component {component+1})" if n_dim > 1 else "")


def plot_state_and_control(axs, state_traj, control_traj, component, **plot_kwargs):
    """ state_traj: (2n, len_traj) = (v, p), control_traj: (2n, len_traj-1) = (alpha, beta); plots one component """

    n = state_traj.shape[0]//2
    axs[0,0].plot(state_traj[component,:], **plot_kwargs)
    axs[0,1].plot(state_traj[n+component,:], **plot_kwargs)
    axs[1,0].plot(control_traj[component,:], **plot_kwargs)
    axs[1,1].plot(control_traj[n+component,:], **plot_kwargs)


def plot_over_time_with_lambda(results_with_lambda, key, cmap, len_traj, ylabel, title, log_U = None, band_key = None):
    """
    Line plot of results_with_lambda[l][key] (shape (len_traj,)) over time, one line per lambda.
    band_key: optional key of a (2, len_traj) array (lower, upper) shaded around each line.
    """

    lambda_list = list(results_with_lambda.keys())

    for i, l in enumerate(lambda_list):
        color = lambda_color(cmap, i, len(lambda_list))
        plt.plot(results_with_lambda[l][key], label = lambda_label(l), color = color)
        if band_key is not None:
            lower, upper = results_with_lambda[l][band_key]
            plt.fill_between(range(len(lower)), lower, upper, color = color, alpha = 0.3)

    if log_U is not None:
        plt.plot(log_U*np.ones(len_traj), '--', c = 'gray', label = r'$\log U$')
    plt.xlabel("Time, $t$")
    plt.xticks(range(0,len_traj,2))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


# ---------- Introducing misdirection ----------

def plot_trajectories_with_lambda(results_with_lambda, cmap):
    """ One figure per state component """

    lambda_list = list(results_with_lambda.keys())
    n = results_with_lambda[lambda_list[0]]["mean_state_trajectory"].shape[0]//2

    for component in range(n):
        fig, axs = create_trajectory_axes(component, n)

        for i, l in enumerate(lambda_list):
            plot_state_and_control(axs, results_with_lambda[l]["mean_state_trajectory"], results_with_lambda[l]["mean_control_trajectory"],
                                   component, label = lambda_label(l), color = lambda_color(cmap, i, len(lambda_list)))

        fig.suptitle(trajectory_title(component, n))
        plt.legend()
        plt.show()


def plot_all_trajectories_with_lambda(results_with_lambda, cmap):
    """ One figure per state component """

    lambda_list = list(results_with_lambda.keys())
    n = results_with_lambda[lambda_list[0]]["mean_state_trajectory"].shape[0]//2

    for component in range(n):
        fig, axs = create_trajectory_axes(component, n)

        for i, l in enumerate(lambda_list):
            state_traj = results_with_lambda[l]["state_trajectories"]
            control_traj = results_with_lambda[l]["control_trajectories"]
            for j in range(state_traj.shape[0]):
                plot_state_and_control(axs, state_traj[j], control_traj[j], component,
                                       label = lambda_label(l) if j == 0 else None,  # One legend entry per lambda
                                       color = lambda_color(cmap, i, len(lambda_list)))

        fig.suptitle(trajectory_title(component, n))
        plt.legend()
        plt.show()


def plot_trajectories_with_misdirection_patterns(deception_parameter = 0.1, patterns = ("standard", "negative", "sinusoidal")):
    """ One figure per state component """

    results = {k: run_experiments_with_fixed_lambda(num_traj, terminal_time, s_b_matrix, s_c_matrix, deception_parameter = deception_parameter,
                                                    misdirection_b = k, misdirection_c = k)
               for k in patterns}

    for component in range(n_dim):
        fig, axs = create_trajectory_axes(component, n_dim)

        for k in patterns:
            plot_state_and_control(axs, results[k]["mean_state_trajectory"], results[k]["mean_control_trajectory"], component, label = k)

        fig.suptitle(trajectory_title(component, n_dim))
        plt.legend()
        plt.show()


def plot_primary_cost_with_lambda(results_with_lambda, cmap):

    lambda_list = list(results_with_lambda.keys())
    primary_cost_list = [results_with_lambda[l]["mean_primary_cost"] for l in lambda_list]

    fig, ax = plt.subplots()
    bar_container = ax.bar(lambda_list, primary_cost_list, width=0.05, color = cmap(0))
    ax.bar_label(bar_container, fmt='{:,.1f}')
    ax.set_xlabel(r"Deception parameter, $\lambda$")
    ax.set_ylabel("Primary cost")
    ax.set_title("Primary Cost")
    plt.show()


def plot_deception_measure_and_primary_cost_with_lambda(results_with_lambda, cmap = None):
    """
    Grouped bars of the deception measure D(lambda) (Eq. (15), left axis, navy) and the primary cost
    (right axis, orange) against lambda, as in the paper's Fig. 1b. cmap is unused (kept for a uniform signature).
    """

    lambda_list = list(results_with_lambda.keys())
    deception_measure_list = [results_with_lambda[l]["mean_deception_measure"] for l in lambda_list]
    primary_cost_list = [results_with_lambda[l]["mean_primary_cost"] for l in lambda_list]

    x = np.arange(len(lambda_list))
    width = 0.35

    fig, ax = plt.subplots()
    ax_cost = ax.twinx()
    bars_D = ax.bar(x - width/2, deception_measure_list, width, color = 'navy', label = "Deception measure")
    bars_cost = ax_cost.bar(x + width/2, primary_cost_list, width, color = 'orange', label = "Primary cost")
    ax.bar_label(bars_D, fmt = '{:.1f}')
    ax_cost.bar_label(bars_cost, fmt = '{:.0f}')

    ax.set_xticks(x, [f"{l:.1f}" for l in lambda_list])
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\mathcal{D}(\lambda)$")
    ax_cost.set_ylabel("Primary cost")
    ax.set_ylim(0, 1.1)
    ax_cost.set_ylim(0, 1.15*max(primary_cost_list))
    ax.legend(handles = [bars_D, bars_cost], loc = "upper left")
    plt.show()


def plot_E_log_L_T_with_lambda(results_with_lambda, log_U, cmap):

    lambda_list = list(results_with_lambda.keys())
    E_log_L_T_list = [results_with_lambda[l]["mean_log_L_t"][-1] for l in lambda_list]

    plt.plot(lambda_list, E_log_L_T_list, c = cmap(0))
    plt.plot(lambda_list, log_U*np.ones(len(lambda_list)), '--', c = 'gray')
    plt.xlabel(r"Deception parameter, $\lambda$")
    plt.ylabel(r"$E[\log L^*_T]$")
    plt.title(r"Expected value of $\log L^*_T$")
    plt.show()


def plot_sht_results_with_lambda(results_with_lambda, log_U, cmap, len_traj):

    mean_sht_results = {l: {"mean_SHT_results": np.mean(result["SHT_results"], axis = 0)}
                        for l, result in results_with_lambda.items()}
    plot_over_time_with_lambda(mean_sht_results, "mean_SHT_results", cmap, len_traj,
                               ylabel = r"$\log L^*_t$", title = r"$\log L^*_t$ with Observed Trajectory", log_U = log_U)


# ---------- Stealthiness of misdirection ----------

def plot_E_log_L_t_with_lambda(results_with_lambda, log_U, cmap, len_traj):

    plot_over_time_with_lambda(results_with_lambda, "mean_log_L_t", cmap, len_traj,
                               ylabel = r"$E[\log L^*_t]$", title = r"Expected value of $\log L^*_t$", log_U = log_U)


def plot_approximated_probability_with_lambda(results_with_lambda, cmap, len_traj):
    """ Requires results computed with compute_probabilities=True; shaded areas are the Lemma 2 confidence intervals """

    plot_over_time_with_lambda(results_with_lambda, "approximated_probability", cmap, len_traj,
                               ylabel = "Approximated probability", title = "Approximated Probability of Detection",
                               band_key = "approximated_probability_CI")


def plot_probability_bound_with_lambda(results_with_lambda, cmap, len_traj):
    """ Requires results computed with compute_probabilities=True """

    plot_over_time_with_lambda(results_with_lambda, "bound_for_probability", cmap, len_traj,
                               ylabel = "Probability bound", title = "Upper Bound for Probability of Detection")


def compare_primary_cost_and_stealthiness_with_lambda():
    pass


if __name__ == '__main__':

    results_with_lambda = run_experiments_over_lambda(lambda_list, compute_probabilities = True)

    plot_trajectories_with_lambda(results_with_lambda, cmap)
    plot_all_trajectories_with_lambda(results_with_lambda, cmap)
    plot_primary_cost_with_lambda(results_with_lambda, cmap)
    plot_deception_measure_and_primary_cost_with_lambda(results_with_lambda, cmap)
    plot_sht_results_with_lambda(results_with_lambda, log_U, cmap, len_traj)
    plot_E_log_L_T_with_lambda(results_with_lambda, log_U, cmap)
    plot_E_log_L_t_with_lambda(results_with_lambda, log_U, cmap, len_traj)
    plot_approximated_probability_with_lambda(results_with_lambda, cmap, len_traj)  # needs compute_probabilities=True
    plot_probability_bound_with_lambda(results_with_lambda, cmap, len_traj)  # needs compute_probabilities=True
    plot_trajectories_with_misdirection_patterns()
