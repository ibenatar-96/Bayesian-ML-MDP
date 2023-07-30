import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import jax
import numpy
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
    Parses the Observation File, returns a Dictionary where: Key: (Function Number, Action Parameter), Value: List of
    observations (list composed of 1's and 0's - 1 indicate that State != Next State, and 0 else).
    Notice that it ignores actions that are in utils.IGNORE_ACTIONS. In our case - 'opponent_mark' is ignored.
    For example: obs_map = {
    (0, 5): [1, 1, 0, 0, 0, 0, 0, 1],
    (0, 1): [1, 1, 1, 1],
    (0, 8): [1, 1, 1, 1, 1, 1],
    (0, 4): [1, 1, 1, 1, 1],
    (0, 7): [1, 1, 1, 1, 1, 1, 1],
    (0, 3): [1, 1, 1, 1, 1],
    (0, 6): [1, 1, 1, 1, 1, 1, 1],
    (0, 2): [1, 1, 1, 1, 1, 1, 1, 1],
    (0, 9): [1, 1, 1]}

    Where 0 is mapped to 'ai_mark' function in global func_numeric_mapping variable. This mapping exists because NumPyro
    MCMC expects numeric values when running inference.

    This Function also adds initial 'alpha' and 'beta' parameters to the global model_beta_parameters for every new
    func_action and action_param it encounters.
    for example - model_beta_parameters = {
    (0, 5): {'alpha': 1, 'beta': 1},
    (0, 1): {'alpha': 1, 'beta': 1} ... }
    """
    func_numeric_count = 0
    obs_map = {}
    obs_list = []
    with open(obs_file, "r") as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespaces and newlines
            line_list = eval(line)  # Convert the line string to a list
            obs_list.append(line_list)
    global model_beta_parameters, func_numeric_mapping
    for episode in obs_list:
        for (state, (func_name, action_param), next_state) in episode:
            if func_name in utils.IGNORE_ACTIONS:
                continue
            if func_name in func_numeric_mapping.keys():
                func_numeric = func_numeric_mapping[func_name]
            else:
                func_numeric = func_numeric_count
                func_numeric_mapping[func_name] = func_numeric_count
                func_numeric_count += 1
            obs_map.setdefault((func_numeric, action_param), []).append(1) if state != next_state \
                else obs_map.setdefault((func_numeric, action_param), []).append(0)
            if not (func_numeric, action_param) in model_beta_parameters:
                model_beta_parameters[(func_numeric, action_param)] = {'alpha': 1, 'beta': 1}

    sorted_dict = dict(sorted(obs_map.items(), key=lambda item: item[0]))
    return sorted_dict


def fill_post_model_params(posterior, prior):
    posterior = {action: prior[action] for action in prior.keys() if action not in posterior}


def ai_model(obs=None, nobs=None):
    """
    (Tic-Tac-Toe) AI Agent model with NumPyro.
    p is a dictionary mapping (function, action parameter) -> sample from Beta distribution using (alpha, beta).
    obs = {
    (0, 5): [1, 1, 0, 0, 0, 0, 0, 1],
    (0, 1): [1, 1, 1, 1],
    (0, 8): [1, 1, 1, 1, 1, 1],
    (0, 4): [1, 1, 1, 1, 1],
    (0, 7): [1, 1, 1, 1, 1, 1, 1],
    (0, 3): [1, 1, 1, 1, 1],
    (0, 6): [1, 1, 1, 1, 1, 1, 1],
    (0, 2): [1, 1, 1, 1, 1, 1, 1, 1],
    (0, 9): [1, 1, 1]}
    """
    # p ~ Beta(alpha, beta)
    global model_beta_parameters
    p = {}
    for key in model_beta_parameters.keys():
        alpha = model_beta_parameters[key]['alpha']
        beta = model_beta_parameters[key]['beta']
        p[key] = numpyro.sample(f"p{str(key)}", dist.Beta(alpha, beta))
    for key, p_i in p.items():
        if obs is not None:
            action_obs = obs[key]
            action_obs = jax.numpy.array(action_obs)
            nobs_i = len(action_obs)
        else:
            action_obs = None
            nobs = nobs
            nobs_i = nobs[key]
        with numpyro.plate(f"obs{str(key)}", size=nobs_i):
            numpyro.sample(f"o{str(key)}", dist.Bernoulli(p_i), obs=action_obs)


def prior_predictive(obs):
    """
    Evaluates Prior Predictive from the Prior Dist.
    Expecting to see mass in the bar where observation and imaginations are aligned.
    """
    prior_predi = numpyro.infer.Predictive(ai_model, num_samples=10000)
    prior_samples = prior_predi(jax.random.PRNGKey(int(time.time() * 1E6)),
                                nobs={key: len(value) for (key, value) in obs.items()})
    if utils.PLOT:
        num_rows, num_cols = largest_divisors(len(obs.keys()))
        assert num_rows * num_cols == len(obs.keys())
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))
        plt.subplots_adjust(hspace=0.5)
        plt.title("prior predictive")
        obs_list = list(obs.items())
        for i in range(num_rows):
            for j in range(num_cols):
                key, action_obs = obs_list[i*num_cols + j]
                o_key = f"o{str(key)}"
                axs[i, j].set_title(o_key)
                axs[i, j].set_xlim(-1, len(action_obs) + 1)
                axs[i, j].hist([sum(o) for o in prior_samples[o_key]], density=True, bins=len(action_obs) * 2 + 1,
                               label="imaginations")
                axs[i, j].axvline(sum(action_obs), color="red", lw=2, label="observation")
                axs[i, j].legend()
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
        num_rows, num_cols = largest_divisors(len(obs.keys()))
        assert num_rows * num_cols == len(obs.keys())
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))
        plt.subplots_adjust(hspace=0.5)
        plt.title("posterior")
        obs_list = list(obs.items())
        for i in range(num_rows):
            for j in range(num_cols):
                key, action_obs = obs_list[i*num_cols + j]
                p_key = f"p{str(key)}"
                axs[i, j].set_title(p_key)
                axs[i, j].set_xlabel("p")
                axs[i, j].hist(mcmc.get_samples()[p_key], density=True, bins='auto')
        plt.show()


def posterior_predictive(obs, mcmc):
    """
    Evaluates Posterior Predictive from the Posterior Dist.
    In basic words, what we are most expecting to see. (Probability to see each observation given our posterior dist.)
    """
    posterior_predi = numpyro.infer.Predictive(ai_model, posterior_samples=mcmc.get_samples())
    posterior_samples = posterior_predi(jax.random.PRNGKey(int(time.time() * 1E6)),
                                        nobs={key: len(value) for (key, value) in obs.items()})
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


# def p_value(obs, posterior_samples):
#     p_value = sum(posterior_samples['o'].sum(axis=1) >= sum(obs)) / len(posterior_samples['o'])
#     print(f"p_value = {p_value:.3f}")


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
                pq = numpy.quantile(p, quantiles)
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
    obs_map = parse_obs(obs_file)
    posterior_model_parameters = {}
    global model_beta_parameters
    prior_predi = prior_predictive(obs_map)
    mcmc = inference(obs_map)
    posterior(mcmc, obs_map)
    posterior_predi = posterior_predictive(obs_map, mcmc)
    # p_value(obs, posterior_samples)
    dist_stats = summarize_posterior(mcmc, obs_map)
    for key, _ in obs_map.items():
        p_key = f"p{str(key)}"
        sample = numpy.random.choice(mcmc.get_samples()[p_key])
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
