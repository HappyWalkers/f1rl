# F1RL

## Todo List
* only disturb one param
* transformer
* Auxiliary prediction head that reconstructs the physical parameters accelerates utilisation of context.
* Analyse gradients – verify that ∂π/∂μ is non‑zero; if it vanishes, the architecture or loss needs adjustment.
* recurrent policy (RecurrentPPO in sb3‑contrib)
* multiple expert policy for IL

## Experiment record:

* Trained SAC agent achieves 16 s/lap in levine in simulation. When deployed, it achieves 30 s/lap. It exceeds our expectation because it didn't go through any fine tuning on the real car.

* Introducing imitation learning with pure pursuit and wall following. Pure pursuit performs way better than wall following. Using IL on pure pursuit before IL leads to best lap time (7s/lap for levine) so far. However, the trained rl agent cannot run smoothly at all when deployed because its overfitting in the simulation. The rl agent tries very large acceleration and starts drifting immediately.

* To mitigate the sim2real gap in the high-speed scenario, I introduce the domain randomization because its widely usage and effectiveness. We introduce disturbations on the mass, inertial, corner stiffness, observation noise, and command delay.
    * Due to the MDP property of gym, I have to use a crafted stratgy to emulate command delay. When an actions is computed, it gets used a random varible number of times. The varible ranges from 0~2. When a command is used 0 time, it means to mimic the behavior that observations are usually more frequent than the actions.
    * RL agent is very sensitive to command delay, especially in high-speed scenario. It's kind of surprising. But let's say the speed is 5 m/s, the command frequency is 25 so the interval is 0.04 second, then the command delay will cause the vehicle keeps going 20 cm along the original direction, which is huge in a narrow track. Hence, I have to maintain the probablity of command delay very low.
    * After using DR, the performance of rl agent is very bad in simulation. The highest speed is around 2 m/s, the average speed is 1.5 m/s, and the speed is lower than 1 m/s when turning around. However, it can be deployed successfully. The behavior of the agent on the real car is similar to in the simulation. Though we don't know the sucess of deployment is caused by DR or just lower speed.

* The rl agent tries to keep different environments happy when using DR. To improve its performance when DR is turned on, contextual reinforcement learning is introduced. 
    * The effectiveness of contextual RL is proved in some papers.
    * Some trials of using VAE to reconstruct state transitions (state, action, next state) proves that trajectories sampled from different environments have different distributions. This means that we can encode the environment-related information into the observation so that the rl agent can take different actions in different environments.
    * To encode trajectories into meaningful environment representations, a subtask is introduced. The subtask is predicting the environment parameters given trajectories. I use LSTM to do that because it can process varible-length trajectories and the hidden states may be useful. The cons of using LSTM is the GPU memory consumption is relatively high when training. 
    * Before integrating LSTM into rl training pipeline, the real environment parameters are included into the observations. However, the reward stays the same as before. Turning off IL leads to even worse reward. Maybe we should use different expert polices for different environments so that the rl agent can learn to adapt to different environments.
    * feature extractor breaks the training. However, reduce the size of network in feature extractor restore the training. 