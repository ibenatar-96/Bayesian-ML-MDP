import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import jax
import numpy as np
import itertools
import time

import utils

# numpyro.set_platform("gpu")
numpyro.set_host_device_count(4)

model_beta_parameters = {}
func_numeric_mapping = {}


# TODO: Change the model_parameters value to probability dist.
# TODO : The values of model_parameters need to be the value of the dist (for example [alpha,beta] for Beta Dist.)

# When inferring the AI Agent, and before sending the model_parameters, we need to sample
# from the dist and send those sampled values to the model_parameters of the AI Agent.
# What we are expecting to see is an increase in the acc. rewards.
# Order is -> 1. Run Model, 2. Gather logs / acc. reward, 3. Update Posterior (model_parameters) and repeat 1.,2.,3...

def parse_obs(obs_file):
    """
    Parses the Observation File, returns a List of tuples,
    where each tuple is composed of (State, (Function, Action Parameter), Next State)
    for example:
    State: [-1, 1, 0, -1, 0, -1, 1, 0, 0] where -1 indicates 'X' on the Board, 1 indicates 'O', and 0 indicates None (empty cell)
    (Function, Action Parameter): (0, 5) -> 0 is the number assigned for the function 'ai_mark', mapping is located in func_numeric_mapping
    Next State: [-1, 1, 0, -1, 1, -1, 1, 0, 0].

    0 is mapped to 'ai_mark' function in global func_numeric_mapping variable. This mapping exists because NumPyro
    MCMC expects numeric values when running inference.

    This Function also adds initial 'alpha' and 'beta' parameters to the global model_beta_parameters for every NEW
    func_action and action_param it encounters.
    for example - model_beta_parameters = {
    (0, 5): {'alpha': 1, 'beta': 1},
    (0, 1): {'alpha': 1, 'beta': 1} ... }
    """

    #Converts states from: [['X', 'O', None], ['X', None, 'X'], ['O', None, None]] to [-1, 1, 0, -1, 0, -1, 1, 0, 0]
    #Meaning, creating a minimal state representation, where 'X' is marked a -1, 'O' as 1, and None as 0.
    def minimal_state(state):
        minimal_repr = []
        for i in range(len(state)):
            for j in range(len(state[i])):
                if state[i][j] == 'X':
                    minimal_repr.append(-1)
                elif state[i][j] == 'O':
                    minimal_repr.append(1)
                else:
                    minimal_repr.append(0)
        return minimal_repr

    obs_list = []
    func_numeric_count = 0
    global model_beta_parameters, func_numeric_mapping
    with open(obs_file, "r") as obs_log:
        for episode in obs_log:
            episode = episode.strip()  # Remove leading/trailing whitespaces and newlines
            episode_list = eval(episode)  # Convert the line string to a list
            for (state, (func_name, action_param), next_state) in episode_list:
                if func_name in utils.IGNORE_ACTIONS:
                    continue
                if func_name in func_numeric_mapping.keys():
                    func_numeric = func_numeric_mapping[func_name]
                else:
                    func_numeric = func_numeric_count
                    func_numeric_mapping[func_name] = func_numeric_count
                    func_numeric_count += 1
                obs_list.append((minimal_state(state), (func_numeric, action_param), minimal_state(next_state)))
                if not (func_numeric, action_param) in model_beta_parameters:
                    model_beta_parameters[(func_numeric, action_param)] = {'alpha': 1, 'beta': 1}
    model_beta_parameters = dict(sorted(model_beta_parameters.items()))
    return obs_list


def fill_post_model_params(posterior, prior):
    posterior = {action: prior[action] for action in prior.keys() if action not in posterior}


def ai_model(obs=None):
    """
    (Tic-Tac-Toe) AI Agent model with NumPyro.
    p is a dictionary mapping (function, action parameter) -> sample from Beta distribution using (alpha, beta).
    obs = [
    ([-1,1,0,...], (0,3), [-1,1,1,...]),
    ([-1,1,...,0], (0,9), [-1,1,...,1]),
    ]
    """
    # p ~ Beta(alpha, beta)
    global model_beta_parameters
    p = {}
    for key in model_beta_parameters.keys():
        alpha = model_beta_parameters[key]['alpha']
        beta = model_beta_parameters[key]['beta']
        p[key] = numpyro.sample(f"p{str(key)}", dist.Beta(alpha, beta))
    # with numpyro.plate(f"obs", size=len(model_beta_parameters.keys())):
    #     for key in model_beta_parameters.keys():
    #         success = [1 if entry[0] != entry[2] else 0 for entry in obs if entry[1] == key] if obs is not None else None
    #         if success is not None:
    #             print(f"key: {key}, success: {success}")
    #             success = jax.numpy.array(success)
    #         numpyro.sample(f"o{key}", dist.Bernoulli(p[key]), obs=success)
    # p ~ Beta(alpha, beta)
    # if obs is not None:
        for i in range(len(obs)):
            s, a, snext = obs[i]
            p_i = p[a]
            success = jax.numpy.array([1 if s != snext else 0])
            numpyro.sample(f"success{i}", dist.Bernoulli(p_i), obs=success)


def prior_predictive():
    """
    Evaluates Prior Predictive from the Prior Dist.
    Expecting to see mass in the bar where observation and imaginations are aligned.
    """
    prior_predi = numpyro.infer.Predictive(ai_model, num_samples=10000)
    prior_samples = prior_predi(jax.random.PRNGKey(int(time.time() * 1E6)))
    if utils.PLOT:
        global model_beta_parameters
        num_mapping = len(model_beta_parameters.keys())
        num_rows, num_cols = largest_divisors(num_mapping)
        assert num_rows * num_cols == num_mapping
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))
        plt.subplots_adjust(hspace=0.5)
        plt.title("prior predictive")
        for i in range(num_rows):
            for j in range(num_cols):
                key = list(model_beta_parameters.keys())[i*3 + j]
                o_key = f"o{str(key)}"
                x_titles = ["Success", "Fail"]
                axs[i, j].set_title(o_key)
                axs[i, j].set_xlim(0, 1)
                success_counts = np.sum(prior_samples[o_key] == 1, axis=0)
                fail_counts = np.sum(prior_samples[o_key] == 0, axis=0)
                normalized_vector = [np.mean(success_counts), np.mean(fail_counts)]
                sum_normalized = np.sum(normalized_vector)
                normalized_vector /= sum_normalized
                axs[i, j].bar(x_titles, normalized_vector)
        plt.show()
    return prior_predi


def inference(obs):
    """
    Runs Inference using MCMC.
    """
    nuts_kernel = numpyro.infer.NUTS(ai_model)
    mcmc = numpyro.infer.MCMC(
        nuts_kernel,
        num_warmup=500,
        num_chains=4,
        num_samples=5000)
    mcmc.run(jax.random.PRNGKey(int(time.time() * 1E6)), obs=obs)
    mcmc.print_summary()
    return mcmc


def largest_divisors(x):
    # Initialize variables to store the two largest divisors
    largest_divisor1 = 1
    largest_divisor2 = x

    # Find the two largest divisors of x
    for i in range(2, int(x ** 0.5) + 1):
        if x % i == 0:
            largest_divisor1 = i
            largest_divisor2 = x // i

    return largest_divisor2, largest_divisor1


def posterior(mcmc, obs):
    if utils.PLOT:
        global model_beta_parameters
        num_mapping = len(model_beta_parameters.keys())
        num_rows, num_cols = largest_divisors(num_mapping)
        assert num_rows * num_cols == num_mapping
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))
        plt.subplots_adjust(hspace=0.5)
        plt.title("posterior")
        for i in range(num_rows):
            for j in range(num_cols):
                key = list(model_beta_parameters.keys())[i*3 + j]
                p_key = f"p{str(key)}"
                x_titles = ["Success", "Fail"]
                axs[i, j].set_title(p_key)
                axs[i, j].set_xlim(0, 1)
                success_counts = np.sum(mcmc.get_samples()[p_key] == 1, axis=0)
                fail_counts = np.sum(mcmc.get_samples()[p_key] == 0, axis=0)
                normalized_vector = [np.mean(success_counts), np.mean(fail_counts)]
                sum_normalized = np.sum(normalized_vector)
                normalized_vector /= sum_normalized
                axs[i, j].bar(x_titles, normalized_vector)

                # p_key = f"p{str(key)}"
                # axs[i, j].set_title(p_key)
                # axs[i, j].set_xlabel("p")
                # axs[i, j].hist(mcmc.get_samples()[p_key], density=True, bins='auto')
        plt.show()


def posterior_predictive(obs, mcmc):
    """
    Evaluates Posterior Predictive from the Posterior Dist.
    In basic words, what we are most expecting to see. (Probability to see each observation given our posterior dist.)
    """
    posterior_predi = numpyro.infer.Predictive(ai_model, posterior_samples=mcmc.get_samples())
    posterior_samples = posterior_predi(jax.random.PRNGKey(int(time.time() * 1E6)))
    if utils.PLOT:
        num_rows, num_cols = largest_divisors(len(obs.keys()))
        assert num_rows * num_cols == len(obs.keys())
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))
        plt.subplots_adjust(hspace=0.5)
        plt.title("posterior predictive")
        obs_list = list(obs.items())
        for i in range(num_rows):
            for j in range(num_cols):
                key, action_obs = obs_list[i*num_cols + j]
                o_key = f"o{str(key)}"
                axs[i, j].set_title(o_key)
                axs[i, j].set_xlim(-1, len(action_obs) + 1)
                axs[i, j].hist([sum(o) for o in posterior_samples[o_key]], density=True, bins=len(action_obs) * 2 + 1,
                               label="imaginations")
                axs[i, j].axvline(sum(action_obs), color="red", lw=2, label="observation")
                axs[i, j].legend()
        plt.show()
    return posterior_predi


def summarize_posterior(mcmc, obs):
    """
    Summarizes Posterior, displays attributes such as mean, standard deviation, quantiles.
    """
    distribution_stats = {}
    if utils.PLOT:
        num_rows, num_cols = largest_divisors(len(obs.keys()))
        assert num_rows * num_cols == len(obs.keys())
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))
        plt.subplots_adjust(hspace=0.5)
        obs_list = list(obs.items())
        for i in range(num_rows):
            for j in range(num_cols):
                key, action_obs = obs_list[i*num_cols + j]
                p_key = f"p{str(key)}"
                p = mcmc.get_samples()[p_key]
                p_mean = p.mean()
                p_stddev = p.std()
                distribution_stats[key] = {'mean': p_mean, 'stddev': p_stddev}
                quantiles = [0, 0.025, 0.25, 0.5, 0.75, 0.975, 1]
                pq = np.quantile(p, quantiles)
                print(f"stat\t{p_key}\n-------------")
                print(f"mean\t{p_mean:.3f}")
                print(f"stddev\t{p_stddev:.3f}")
                for q in range(len(quantiles)):
                    print(f"{quantiles[q] * 100:3.0f}%\t{pq[q]:.3f}")
                print("\n")
                height, _, _ = axs[i, j].hist(p, histtype="step", lw=2, bins="auto", label="posterior")
                axs[i, j].set_title(f"mean={p_mean:.3f}, stddev={p_stddev:.3f}")
                axs[i, j].axvline(p_mean, ls="dashed", color="red", label="mean")
                axs[i, j].fill_betweenx([0, height.max()], pq[1], pq[-2],
                                        color="red", alpha=0.1, label=f"{(quantiles[-2] - quantiles[1]) * 100:.0f}%")
                axs[i, j].fill_betweenx([0, height.max()], pq[2], pq[-3],
                                        color="red", alpha=0.2, label=f"{(quantiles[-3] - quantiles[2]) * 100:.0f}%")
        plt.show()
    return distribution_stats


def ML(obs_file, prior_model_parameters=None):
    """
    Using the Logs in observations.log,
    """
    obs_list = parse_obs(obs_file)
    posterior_model_parameters = {}
    global model_beta_parameters
    prior_predi = prior_predictive()
    mcmc = inference(obs_list)
    posterior(mcmc, obs_list)
    posterior_predi = posterior_predictive(obs_list, mcmc)
    # p_value(obs, posterior_samples)
    dist_stats = summarize_posterior(mcmc, obs_list)
    for key in obs_list:
        p_key = f"p{str(key)}"
        sample = np.random.choice(mcmc.get_samples()[p_key])
        print(f"sample: {sample}")
        p_mean = dist_stats[key]['mean']
        p_stddev = dist_stats[key]['stddev']
        posterior_model_parameters[key] = sample
        model_beta_parameters[key]['alpha'] = p_mean * (
                ((p_mean * (1 - p_mean)) / p_stddev) - 1)
        model_beta_parameters[key]['beta'] = (1 - p_mean) * (
                ((p_mean * (1 - p_mean)) / p_stddev) - 1)
    fill_post_model_params(posterior_model_parameters, prior_model_parameters)
    print(f"posterior model parameters: {posterior_model_parameters}")
    return posterior_model_parameters
