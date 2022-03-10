import random
from env.Maze import GoalTextMaze
import numpy as np
from utils.Params import ParamsLoader
import matplotlib.pyplot as plt
import IPython.terminal.debugger as Debug


# define a random policy collector
class RandomPolicy(object):
    """
        Collect the optimal episodes from multiple environment
    """

    def __init__(self, env_list, configs):
        super(RandomPolicy, self).__init__()
        self.env_list = env_list
        self.sample_episode_num = configs['sample_num']
        self.rnd_seed = configs["rnd_seed"]
        self.results_data = []
        self.configs = configs

    def sample_episodes(self, configs):
        """
        This function is used to sample episodes from one maze
        The return value is a list of [maze id, maze map, episodes]
        for each episode, we have [obs_0, a_1, obs_1, a_t, ..., ]
        for each obs, we have [depth obs, loc obs]
        """
        # sample multiple episodes
        episodes = []

        # create the maze and the map
        env_maze = configs['maze']

        # get the actions
        action_name = ['turn_left', 'turn_right', 'forward', 'backward']
        action_list = env_maze.agent_action_space

        # start collecting episodes
        for i in range(configs['episode_num']):
            # sample the start and goal locations
            obs = env_maze.reset()
            # get the optimal trajectory using random policy

            # compose the obs
            obs = np.array(obs['observation'])
            loc_obs = np.array([env_maze.agent.pos[0],
                                env_maze.agent.pos[2],
                                np.sin(env_maze.agent.dir),
                                np.cos(env_maze.agent.dir)])
            obs = [obs, loc_obs]
            episode = [obs]
            # start collecting one episode
            for t in range(configs['maze'].max_episode_steps):
                # sample one action
                action = random.sample(action_list, 1)[0]
                # step
                next_obs, reward, done, _ = env_maze.step(action)
                # store the trajectory
                episode.append(action_name[action_list.index(action)])

                # compose the next obs
                obs = np.array(next_obs['observation'])
                loc_obs = np.array([env_maze.agent.pos[0],
                                    env_maze.agent.pos[2],
                                    np.sin(env_maze.agent.dir),
                                    np.cos(env_maze.agent.dir)])
                next_obs = [obs, loc_obs]
                episode.append(next_obs)

            # save the episode
            print(f"Process maze id {configs['id']}: {i + 1} episode.")
            episodes.append(episode)
        env_maze.close()
        # put the data to the queue
        # return {'id': configs['id'], 'map': configs['map'].tolist(), 'episodes': episodes}
        return [configs['map'], episodes]

    def run(self):
        # construct the configurations for each maze
        for idx in range(len(self.env_list)):
            configs = {'id': idx,
                       'maze': self.env_list[idx][0],
                       'map': self.env_list[idx][1],
                       'episode_num': self.configs["sample_num"]
                       }

            # run mazes one by one
            res = self.sample_episodes(configs)

            # save the results
            self.save_results(idx, res)

    # save function
    def save_results(self, idx, results):
        np.save(f'./data/maze_{self.configs["maze_size"]}_{self.configs["obs_name"]}_id_{idx}.npy', results)


def load_maze(m_id, configs):
    maze_file = configs["maze_path"] + f"maze_{configs['maze_size']}_{m_id}.txt"
    # load the map
    with open(maze_file, 'r') as f_in:
        lines = f_in.readlines()
    f_in.close()

    maze_map = np.array([[int(float(d)) for d in l.rstrip().split(',')] for l in lines])
    maze_map = np.where(maze_map == 2, 1, maze_map)

    # create the maze from text
    maze = GoalTextMaze(text_file=maze_file,
                        room_size=configs['room_size'],
                        wall_size=configs['wall_size'],
                        obs_name=configs['obs_name'],
                        max_episode_steps=configs['max_episode_steps'],
                        rnd_init=configs['rnd_init'],
                        rnd_goal=configs['rnd_goal'],
                        agent_rnd_spawn=configs['agent_rnd_spawn'],
                        goal_rnd_spawn=configs['goal_rnd_spawn'],
                        action_space=configs['action_space'],
                        obs_height=configs['obs_height'],
                        obs_width=configs['obs_width'])

    # generate the ground truth map
    rows, cols = maze_map.shape
    gt_map_arr = np.zeros((configs['room_size'] * rows, configs['room_size'] * cols))
    for r in range(gt_map_arr.shape[0]):
        for c in range(gt_map_arr.shape[1]):
            gt_map_arr[r, c] = maze_map[r // configs['room_size'], c // configs['room_size']]

    return [maze, gt_map_arr]


if __name__ == "__main__":
    # parse the input arguments
    run_configs = ParamsLoader("./Params/run_demo_params.json").params_data

    # store the maze list
    maze_list = []
    for i in range(run_configs['maze_num']):
        maze_env = load_maze(i, run_configs)
        maze_list.append(maze_env)

    # create the random collector
    myCollector = RandomPolicy(maze_list, run_configs)
    myCollector.run()





