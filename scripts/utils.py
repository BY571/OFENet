

def timer(start,end, train_type="Training"):
    """ Helper to print training time """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\n{} Time:  {:0>2}:{:0>2}:{:05.2f}".format(train_type, int(hours),int(minutes),seconds))

def fill_buffer(agent, env, samples=1000):
    collected_samples = 0
    print("HERE3")
    state_size = env.observation_space.shape[0]
    print("HERE4")
    state = env.reset() 
    print("HERE5")
    state = state.reshape((1, state_size))
    for i in range(samples):
            
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        next_state = next_state.reshape((1, state_size))
        agent.memory.add(state, action, reward, next_state, done)
        collected_samples += 1
        state = next_state
        if done:
            state = env.reset()
            state = state.reshape((1, state_size))
    print("Adding random samples to buffer done! Buffer size: ", agent.memory.__len__())
                
def pretrain_ofenet(agent, epochs, writer, target_dim):
    for ep in range(epochs):
        # ---------------------------- update OFENet ---------------------------- #
        ofenet_loss = agent.ofenet.train_ofenet(agent.memory.sample())
        writer.add_scalar("OFENet-pretrainig-loss", ofenet_loss, ep)
    return agent

def get_target_dim(env_name):
    TARGET_DIM_DICT = {
        "AntBulletEnv-v0": 27, # originally 28
        "HalfCheetahBulletEnv-v0": 17, # originally 26
        "Walker2dBulletEnv-v0": 17,
        "HopperBulletEnv-v0": 11, # originally 15
        "ReacherBulletEnv-v0": 11, # originally 9
        "HumanoidBulletEnv-v0": 292, # originally 44
        "Pendulum-v0": 3,
        "LunarLanderContinuous-v2": 8
    }
    return TARGET_DIM_DICT[env_name]