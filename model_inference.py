import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import jax
import numpy
import scipy
import time

import utils

gpu_available = jax.cuda.device_count() > 0

if gpu_available:
    # Use GPU for inference
    numpyro.set_platform("gpu")
else:
    # Use CPU for inference
    numpyro.set_platform("cpu")

model_beta_parameters = {}


# TODO: Change the model_parameters value to probability dist.
# TODO : The values of model_parameters need to be the value of the dist (for example [alpha,beta] for Beta Dist.)

# When inferring the AI Agent, and before sending the model_parameters, we need to sample
# from the dist and send those sampled values to the model_parameters of the AI Agent.
# What we are expecting to see is an increase in the acc. rewards.
# Order is -> 1. Run Model, 2. Gather logs / acc. reward, 3. Update Posterior (model_parameters) and repeat 1.,2.,3...

def parse_obs(obs_file):
    """
    Parses the Observation File, returns a Dictionary where: Key: (Function Name, Action Parameter), Value: List of
    observations (list composed of 1's and 0's - 1 indicate that State != Next State, and 0 else).
    Notice that it ignores actions that are in utils.IGNORE_ACTIONS. In our case - 'opponent_mark' is ignored.
    For example: obs_map = {
    ('ai_mark', 5): [1, 1, 0, 0, 0, 0, 0, 1],
    ('ai_mark', 1): [1, 1, 1, 1],
    ('ai_mark', 8): [1, 1, 1, 1, 1, 1],
    ('ai_mark', 4): [1, 1, 1, 1, 1],
    ('ai_mark', 7): [1, 1, 1, 1, 1, 1, 1],
    ('ai_mark', 3): [1, 1, 1, 1, 1],
    ('ai_mark', 6): [1, 1, 1, 1, 1, 1, 1],
    ('ai_mark', 2): [1, 1, 1, 1, 1, 1, 1, 1],
    ('ai_mark', 9): [1, 1, 1]}

    This Function also adds initial 'alpha' and 'beta' parameters to the global model_beta_parameters for every new func_action and action_param it encounters.
    for example - model_beta_parameters = {
    ('ai_mark', 5): {'alpha': 1, 'beta': 1},
    ('ai_mark', 1): {'alpha': 1, 'beta': 1} ... }
    """
    obs_map = {}
    obs_list = []
    with open(obs_file, "r") as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespaces and newlines
            line_list = eval(line)  # Convert the line string to a list
            obs_list.append(line_list)
    global model_beta_parameters
    for episode in obs_list:
        for (state, (func_name, action_param), next_state) in episode:
            if func_name in utils.IGNORE_ACTIONS:
                continue
            obs_map.setdefault((func_name, action_param), []).append(1) if state != next_state \
                else obs_map.setdefault((func_name, action_param), []).append(0)
            if not (func_name, action_param) in model_beta_parameters:
                model_beta_parameters[(func_name, action_param)] = {'alpha': 1, 'beta': 1}
    return obs_map


def fill_post_model_params(posterior, prior):
    for action in prior.keys():
        if action not in posterior:
            posterior[action] = prior[action]


def ai_model(alpha, beta, obs=None, nobs=utils.INIT_OBSERVATIONS_LEN):
    """
    (Tic-Tac-Toe) AI Agent model with NumPyro.
    """
    if obs is not None:  # for prior predictive
        nobs = len(obs)
    # p ~ Beta(1, 1)
    p = numpyro.sample("p", dist.Beta(alpha, beta))
    with numpyro.plate("obs", len(obs) if obs is not None else nobs):
        # o ~ Bernoulli(p)
        numpyro.sample("o", dist.Bernoulli(p), obs=obs)


def prior_predictive(obs, alpha, beta):
    """
    Evaluates Prior Predictive from the Prior Dist.
    Expecting to see mass in the bar where observation and imaginations are aligned.
    """
    prior_predi = numpyro.infer.Predictive(ai_model, num_samples=10000)
    prior_samples = prior_predi(jax.random.PRNGKey(int(time.time() * 1E6)), alpha=alpha, beta=beta)
    if utils.PLOT:
        plt.figure(figsize=(10, 3))
        plt.xlim(-1, len(obs) + 1)
        plt.hist([sum(o) for o in prior_samples['o']], density=True, bins=len(obs) * 2 + 1,
                 label="imaginations")
        plt.axvline(sum(obs), color="red", lw=2, label="observation")
        plt.title("prior predictive")
        plt.legend()
        plt.show()
    return prior_predi


def inference(obs, alpha, beta):
    """
    Runs Inference using MCMC.
    """
    nuts_kernel = numpyro.infer.NUTS(ai_model)
    mcmc = numpyro.infer.MCMC(
        nuts_kernel,
        num_warmup=500,
        num_chains=4,
        num_samples=5000)
    mcmc.run(jax.random.PRNGKey(int(time.time() * 1E6)), alpha=alpha, beta=beta, obs=jax.numpy.array(obs))
    mcmc.print_summary()
    return mcmc


def posterior(mcmc):
    if utils.PLOT:
        plt.figure(figsize=(10, 3))
        plt.title("posterior")
        plt.xlabel("p")
        plt.hist(mcmc.get_samples()['p'], density=True, bins="auto", label="approximate")
        plt.show()


def posterior_predictive(obs, mcmc, alpha, beta):
    """
    Evaluates Posterior Predictive from the Posterior Dist.
    In basic words, what we are most expecting to see. (Probability to see each observation given our posterior dist.)
    """
    posterior_predi = numpyro.infer.Predictive(ai_model, posterior_samples=mcmc.get_samples())
    posterior_samples = posterior_predi(jax.random.PRNGKey(int(time.time() * 1E6)), alpha=alpha, beta=beta)
    if utils.PLOT:
        plt.figure(figsize=(10, 3))
        plt.xlim(-1, len(obs) + 1)
        plt.hist([sum(o) for o in posterior_samples['o']], density=True, bins=len(obs) * 2 + 1,
                 label="imaginations")
        plt.axvline(sum(obs), color="red", lw=2, label="observation")
        plt.title("posterior predictive")
        plt.legend()
        plt.show()
    return posterior_predi


# def p_value(obs, posterior_samples):
#     p_value = sum(posterior_samples['o'].sum(axis=1) >= sum(obs)) / len(posterior_samples['o'])
#     print(f"p_value = {p_value:.3f}")


def summarize_posterior(mcmc):
    """
    Summarizes Posterior, displays attributes such as mean, standard deviation, quantiles.
    """
    p = mcmc.get_samples()["p"]
    p_mean = p.mean()
    p_stddev = p.std()
    quantiles = [0, 0.025, 0.25, 0.5, 0.75, 0.975, 1]
    pq = numpy.quantile(p, quantiles)
    print(f"stat\tp\n-------------")
    print(f"mean\t{p_mean:.3f}")
    print(f"stddev\t{p_stddev:.3f}")
    for i in range(len(quantiles)):
        print(f"{quantiles[i] * 100:3.0f}%\t{pq[i]:.3f}")
    if utils.PLOT:
        plt.figure(figsize=(10, 3))
        plt.xlabel("p")
        height, _, _ = plt.hist(p, histtype="step", lw=2, bins="auto", label="posterior")
        plt.title(f"mean={p_mean:.3f}, stddev={p_stddev:.3f}")
        plt.axvline(p_mean, ls="dashed", color="red", label="mean")
        plt.fill_betweenx([0, height.max()], pq[1], pq[-2],
                          color="red", alpha=0.1, label=f"{(quantiles[-2] - quantiles[1]) * 100:.0f}%")
        plt.fill_betweenx([0, height.max()], pq[2], pq[-3],
                          color="red", alpha=0.2, label=f"{(quantiles[-3] - quantiles[2]) * 100:.0f}%")
        plt.legend()
        plt.show()
    return p_mean, p_stddev


def ML(obs_file, prior_model_parameters=None):
    """
    Using the Logs in observations.log,
    """
    obs_map = parse_obs(obs_file)
    posterior_model_parameters = {}
    global model_beta_parameters
    for (func_name, action_parameter), obs in obs_map.items():
        if func_name in utils.IGNORE_ACTIONS:
            print(f"Ignoring function: {func_name}")
            continue
        if len(obs) < 10:
            print(f"Not enough observations for: {func_name, action_parameter}")
            continue
        print(f"Function: {func_name}, Action Parameter: {action_parameter}, Obs: {obs}")
        alpha = model_beta_parameters[(func_name, action_parameter)]['alpha']
        beta = model_beta_parameters[(func_name, action_parameter)]['beta']
        print(f"alpha: {alpha}, beta: {beta}")
        prior_predi = prior_predictive(obs, alpha, beta)
        mcmc = inference(obs, alpha, beta)
        posterior(mcmc)
        posterior_predi = posterior_predictive(obs, mcmc, alpha, beta)
        # p_value(obs, posterior_samples)
        p_mean, p_stddev = summarize_posterior(mcmc)
        sample = numpy.random.choice(mcmc.get_samples()["p"])
        print(f"sample: {sample}\n")
        posterior_model_parameters[(func_name, action_parameter)] = sample
        model_beta_parameters[(func_name, action_parameter)]['alpha'] = p_mean * (
                    ((p_mean * (1 - p_mean)) / p_stddev) - 1)
        model_beta_parameters[(func_name, action_parameter)]['beta'] = (1 - p_mean) * (
                    ((p_mean * (1 - p_mean)) / p_stddev) - 1)
    fill_post_model_params(posterior_model_parameters, prior_model_parameters)
    print(f"posterior model parameters: {posterior_model_parameters}")
    return posterior_model_parameters
