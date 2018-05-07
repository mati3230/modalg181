import retro
	
def start_env():
	env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act2')
	env.reset()

	for i in range(10000):
		action = env.action_space.sample()
		obs, rew, done, info = env.step(action)
		env.render()
		if done:
			break
	# necessary to prevent import error
	env.close()
	print("done")
	
start_env()
'''
movie = retro.Movie('SonicTheHedgehog-Genesis-GreenHillZone.Act1-0000.bk2')
movie.step()

env = retro.make(game=movie.get_game(), state=retro.STATE_NONE, use_restricted_actions=retro.ACTIONS_ALL)
env.initial_state = movie.get_state()
env.reset()

while movie.step():
	keys = []
	for i in range(env.NUM_BUTTONS):
		keys.append(movie.get_key(i))
	_obs, _rew, _done, _info = env.step(keys)
	env.render()
'''