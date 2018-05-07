import gym

env = gym.make("MsPacman-v0")
state = env.reset()

for i in range(1000):
	action = env.action_space.sample()
	obs, rew, done, info = env.step(action)
	env.render()
	if done:
		break
# necessary to prevent import error
env.close()
print("done")