# TODO: Change the model_parameters value to probability dist.
# TODO : The values of model_parameters need to be the value of the dist (for example [alpha,beta] for Beta Dist.)

# When inferring the AI Agent, and before sending the model_parameters, we need to sample
# from the dist and send those sampled values to the model_parameters of the AI Agent.
# What we are expecting to see is an increase in the acc. rewards.
# Order is -> 1. Run Model, 2. Gather logs / acc. reward, 3. Update Posterior (model_parameters) and repeat 1.,2.,3...
