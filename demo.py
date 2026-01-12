from chopsticks import chopsticks_v0

env = chopsticks_v0.env()
seed = None
env.reset(seed=seed)
env.render()

for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()

    if terminated or truncated:
        action = None
    else:
        if obs:
            action_mask = obs["action_mask"]
            action = env.action_space(agent).sample(action_mask)
        else:
            action = env.action_space(agent).sample()
    # print(action, env.unwrapped.game_state.visited)
    env.step(action)
    env.render()
env.close()
