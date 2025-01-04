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
        mask = obs["action_mask"]
        action = env.action_space(agent).sample(mask)
    # print(action, env.unwrapped.game_state.visited)
    env.step(action)
    env.render()
env.close()
