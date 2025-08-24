# ToDos:

* Refactor

  * refactor evalute()
    * ~~only use normal env instead of vec env~~
      * ~~Bad idea because vec norm~~
    * ~~use only a single env, the DummyVecEn~~
      * ~~Bad idea because evaluate() may evaluate policies in multiple environments~~
    * ~~use original design, vec_env for both train and evaluation; only use vec_norm for RL policies~~
      * ~~Should read the codebase first; forget a lot of things~~
        * ~~vecnorm only normalizes observations and reward, not actions~~
        * ~~vecnorm should be used for both training and evaluation~~
        * expert policies shoud use raw observation for convenience; the imitation learning phase can use unnormalized observations for expert policies and keep the normalized observations for training
          * for off-policy methods, the train method of model will sample raw observations and rewards from buffer and applies normalization if vecnorm is available
          * for on-policy methods, the BC seems to use the transitions directly, so we may need to use normalized observations and rewards
  * ~~fix expert policies~~
    * ~~wall following~~
    * pure pursuit
    * lattice planner
  * write some tests
    * use reward to check the functionality of expert policies and RL policies
      * cmd to run tests: python3 -m pytest rl/test.py
      * cmd to run main script for evaluation: python3 -m rl.main --use_il=false --num_envs=1 --use_dr=false --num_param_cmbs=1 --eval=true --num_eval_episodes=10 --render_in_eval=false --plot_in_eval=false --algorithm=wall_follow
  * avoid too flexible code
    * use type annotation to limit the type used
  * add more logs to increase the understanding of the codebase
  * maybe we should stop using vecnorm and use some basic division to limit the value scale

* Diffusion policy uses receding horizon control that reminds me of MPC. Maybe we can use that too
  * Implement an MPC first
  * Use action sequence prediction in RL

* bidirectional RNN -> give it a try

* Get a good policy (pure pursuit, MPC, MPPI) on the real car & use that policy as the expert policy. Maybe RL fine tuning is a bad idea because it trains a policy only working well in sim and increase the sim-to-real gap. The quality of the policy purely depends on the quality of the simulation.

* use MPC to replace the dynamic part and use the rl policy as the planner

* use model-based rl and use the model’s error to measure the accuracy of the dynamic part. Another advantage of model-based rl is that we can use real-world data to fine tune the dynamic model.

* Modify the command delay model. The min delay is one step
* Reduce observation noise
* Action noise
* improve sample efficiency (such as using model based rl) and then fine tune the model in the reality
* Nueral ODE
* multiple expert policy for IL
* scenario generation (random or AI)
* use a special loss to force the network to learn different behaviors in different envs
* Use another sim to use acc and steering angle as the output
* change the output from v & s to a & sv?
* Tune the params in ROS to get real-time mode
* train the agent directly in the sim with ROS instead of using command delay? -> Use particle filter instead of accurate localization in sim

# Dones:

* Trained SAC agent achieves 16 s/lap in levine in simulation. When deployed, it achieves 30 s/lap. It exceeds our expectation because it didn't go through any fine tuning on the real car.
* Introducing imitation learning with pure pursuit and wall following. Pure pursuit performs way better than wall following. Using IL on pure pursuit before IL leads to best lap time (7s/lap for levine) so far. However, the trained rl agent cannot run smoothly at all when deployed because its overfitting in the simulation. The rl agent tries very large acceleration and starts drifting immediately.
* To mitigate the sim2real gap in the high-speed scenario, I introduce the domain randomization (DR) because its widely usage and effectiveness. We introduce disturbations on the mass, inertial, corner stiffness, observation noise, and command delay.

  * Due to the MDP property of gym, I have to use a crafted stratgy to emulate command delay. When an actions is computed, it gets used a random varible number of times. The varible ranges from 0~2. When a command is used 0 time, it means to mimic the behavior that observations are usually more frequent than the actions.
  * RL agent is very sensitive to command delay, especially in high-speed scenario. It's kind of surprising. But let's say the speed is 5 m/s, the command frequency is 25 so the interval is 0.04 second, then the command delay will cause the vehicle keeps going 20 cm along the original direction, which is huge in a narrow track. Hence, I have to maintain the probablity of command delay very low.
  * After using DR, the performance of rl agent is very bad in simulation. The highest speed is around 2 m/s, the average speed is 1.5 m/s, and the speed is lower than 1 m/s when turning around. However, it can be deployed successfully. The behavior of the agent on the real car is similar to in the simulation. Though we don't know the sucess of deployment is caused by DR or just lower speed.
  * Reducing the number of parameter set used from 24 to 2 only increases the reward by a small amount. Reducing the the disturbed params from 12 to 1 has the same effect.
* The rl agent tries to keep different environments happy when using DR. To improve its performance when DR is turned on, contextual reinforcement learning (cRL) is introduced.

  * The effectiveness of contextual RL is proved in some papers.
  * Some trials of using VAE to reconstruct state transitions (state, action, next state) proves that trajectories sampled from different environments have different distributions. This means that we can encode the environment-related information into the observation so that the rl agent can take different actions in different environments.
  * To encode trajectories into meaningful environment representations, a subtask is introduced. The subtask is predicting the environment parameters given trajectories. I use LSTM to do that because it can process varible-length trajectories and the hidden states may be useful. The cons of using LSTM is the GPU memory consumption is relatively high when training.
  * Before integrating LSTM into rl training pipeline, the real environment parameters are included into the observations. However, the reward stays the same as before. Turning off IL leads to even worse reward. Maybe we should use different expert polices for different environments so that the rl agent can learn to adapt to different environments.
* Feature extractor

  * feature extractor breaks the training. However, reduce the size of network in feature extractor restore the training.
  * Several different architectures, including MLP, residual connection, FiLM layer, transformer, MoE, are tried but no obvious difference is observed. Residual connection does improve the training efficiency by making the optimization easier. Let's use MoE as default because it aligns with the inductive bias of cRL.
* Recurrent policy

  * PPO performs way better than SAC in the environment without DR. Potential influencial factors: gradient clipping leads to stable training, smaller network causes easier optimization, different hyper parameters...
  * Recurrent policy works way better than normal policies, even in domain randomized environments.
* IL
* Check the qos for the subscriber
* Clip the acceleration by clipping the next velocity
* Move to foxglove if needed
* Increase the progress reward
* Remove the punishment for control signal in reward
* Remove the lidar mask in training
* Create a docker
