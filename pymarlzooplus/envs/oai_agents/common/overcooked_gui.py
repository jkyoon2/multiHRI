import json
import numpy as np
import pandas as pd
import pygame
from pygame import K_UP, K_LEFT, K_RIGHT, K_DOWN, K_SPACE, K_s
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE
import matplotlib
import time

matplotlib.use('TkAgg')

import os
from os import environ, name

import pathlib
USING_WINDOWS = (name == 'nt')
# Windows path

if USING_WINDOWS:
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath


# Lab streaming layer
from pylsl import local_clock

# Used to activate game window at game start for immediate game play
if USING_WINDOWS:
    import pygetwindow as gw

from oai_agents.agents.hrl import HierarchicalRL
# from oai_agents.agents import Manager
from oai_agents.common.subtasks import facing
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv
from oai_agents.gym_environments.worker_env import OvercookedSubtaskGymEnv
from overcooked_ai_py.mdp.overcooked_mdp import Direction, Action
# from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer, roboto_path
# from scripts.train_agents import get_bc_and_human_proxy

class OvercookedGUI:
    """Class to run an Overcooked Gridworld game, leaving one of the agents as fixed.
    Useful for debugging. Most of the code from http://pygametutorials.wikidot.com/tutorials-basic."""

    def __init__(self, args, layout_name=None, agent=None, teammates=None, p_idx=0, horizon=400,
                 trial_id=None, user_id=None, stream=None, outlet=None, fps=5, gif_name='gif'):
        self.x = None
        self._running = True
        self._display_surf = None
        self.args = args
        self.layout_name = layout_name or 'asymmetric_advantages'

        self.use_subtask_env = False
        if self.use_subtask_env:
            kwargs = {'single_subtask_id': 10, 'args': args, 'is_eval_env': True}
            self.env = OvercookedSubtaskGymEnv(**p_kwargs, **kwargs)
        else:
            self.env = OvercookedGymEnv(layout_name=self.layout_name, args=args, ret_completed_subtasks=False,
                                        is_eval_env=True, horizon=horizon, learner_type='originaler',

                                        )
        self.agent = agent
        self.p_idx = p_idx
        self.env.set_teammates(teammates)
        self.env.reset(p_idx=self.p_idx)
        if self.agent != 'human':
            self.agent.set_encoding_params(self.p_idx, self.args.horizon, env=self.env, is_haha=isinstance(self.agent, HierarchicalRL), tune_subtasks=False)
            self.env.encoding_fn = self.agent.encoding_fn

        for t_idx, teammate in enumerate(self.env.teammates):
            teammate.set_encoding_params(t_idx+1, self.args.horizon, env=self.env, is_haha=isinstance(teammate, HierarchicalRL), tune_subtasks=True)

        self.teammate_names= [n.name for n in self.env.teammates]
        self.deterministic = False
        self.env.deterministic = self.deterministic


        self.grid_shape = self.env.grid_shape
        self.trial_id = trial_id
        self.user_id = user_id
        self.fps = fps

        self.score = 0
        self.curr_tick = 1
        self.num_collisions = 0
        self.human_action = None
        self.data_path = args.base_dir / args.data_path
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.tile_size = 150

        self.info_stream = stream
        self.outlet = outlet
        # Currently unused, but keeping in case we need it in the future.
        self.collect_trajectory = True
        self.trajectory = {
            'positions': [],
            'actions': [],
            'observations': []
        }

        self.gif_name = gif_name
        if not os.path.exists(f'data/screenshots/{self.gif_name}'):
            os.makedirs(f'data/screenshots/{self.gif_name}')

    def start_screen(self):
        pygame.init()
        surface = StateVisualizer(tile_size=self.tile_size).render_state(self.env.state,
                                                                         grid=self.env.env.mdp.terrain_mtx,
                                                                         hud_data={"timestep": 1, "score": 0})

        self.surface_size = surface.get_size()
        self.x, self.y = (1920 - self.surface_size[0]) // 2, (1080 - self.surface_size[1]) // 2
        self.grid_shape = self.env.mdp.shape
        self.hud_size = self.surface_size[1] - (self.grid_shape[1] * self.tile_size)
        environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.x, self.y)

        self.window = pygame.display.set_mode(self.surface_size, HWSURFACE | DOUBLEBUF | RESIZABLE)

        pygame.font.init()
        start_font = pygame.font.SysFont(roboto_path, 75)
        text = start_font.render('Press Enter to Start', True, (255, 255, 255))
        start_surface = pygame.Surface(self.surface_size)
        start_surface.fill((155, 101, 0))
        text_x, text_y = (self.surface_size[0] - text.get_size()[0]) // 2, (self.surface_size[1] - text.get_size()[1]) // 2
        start_surface.blit(text, (text_x, text_y))

        self.window.blit(start_surface, (0, 0))
        pygame.display.flip()

        if USING_WINDOWS:
            win = gw.getWindowsWithTitle('pygame window')[0]
            win.activate()

        start_screen = False
        while start_screen:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    start_screen = False


    def on_init(self):
        surface = StateVisualizer(tile_size=self.tile_size).render_state(self.env.state,
                                                                         grid=self.env.env.mdp.terrain_mtx,
                                                                         hud_data={"timestep": 1, "score": self.score})
        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        self._running = True

        pygame.image.save(self.window, f"data/screenshots/{self.gif_name}/-1.png")
        # exit(0)

        if USING_WINDOWS:
            win = gw.getWindowsWithTitle('pygame window')[0]
            win.activate()

    def on_event(self, event):
        if event.type == pygame.KEYDOWN:
            pressed_key = event.dict['key']
            action = None

            if pressed_key == K_UP:
                action = Direction.NORTH
            elif pressed_key == K_RIGHT:
                action = Direction.EAST
            elif pressed_key == K_DOWN:
                action = Direction.SOUTH
            elif pressed_key == K_LEFT:
                action = Direction.WEST
            elif pressed_key == K_SPACE:
                action = Action.INTERACT
            elif pressed_key == K_s:
                action = Action.STAY
            else:
                action = Action.STAY
            self.human_action = Action.ACTION_TO_INDEX[action]

        if event.type == pygame.QUIT:
            self._running = False

    def step_env(self, agent_action):
        prev_state = self.env.state

        tile_in_front = facing(self.env.env.mdp.terrain_mtx, self.env.state.players[self.p_idx])
        prev_obj = self.env.state.players[self.p_idx].held_object.name if self.env.state.players[
            self.p_idx].held_object else None

        obs, reward, done, info = self.env.step(agent_action)

        curr_obj = self.env.state.players[self.p_idx].held_object.name if self.env.state.players[
            self.p_idx].held_object else None

        collision = self.env.mdp.prev_step_was_collision
        if collision:
            self.num_collisions += 1

        # Log data
        curr_reward = sum(info['sparse_r_by_agent'])
        self.score += curr_reward
        transition = {
            "state": json.dumps(prev_state.to_dict()),
            "joint_action": json.dumps(self.env.get_joint_action()),
            # TODO get teammate action from env to create joint_action json.dumps(joint_action.item()),
            "reward": curr_reward,
            "time_left": max((self.env.env.horizon - self.curr_tick) / self.fps, 0),
            "score": self.score,
            "time_elapsed": self.curr_tick / self.fps,
            "cur_gameloop": self.curr_tick,
            "layout": self.env.env.mdp.terrain_mtx,
            "layout_name": self.layout_name,
            "trial_id": self.trial_id,
            "user_id": self.user_id,
            "dimension": (self.x, self.y, self.surface_size, self.tile_size, self.grid_shape, self.hud_size),
            "Unix_timestamp": time.time(),
            "LSL_timestamp": local_clock(),
            # TEAMMATE and POP(TODO): uncommment it and replace teammate_name by teammate_names
            # "agent": self.teammate_name,
            "p_idx": self.p_idx,
            "collision": collision,
            "num_collisions": self.num_collisions
        }
        trans_str = json.dumps(transition)
        if self.outlet is not None:
            self.outlet.push_sample([trans_str])

        if self.collect_trajectory:
            player_positions = [p.position for p in self.env.state.players]
            obs_copy = {k: np.copy(v) for k, v in obs.items()}

            self.trajectory['positions'].append(player_positions)
            self.trajectory['actions'].append(self.env.get_joint_action())
            self.trajectory['observations'].append(obs_copy)

        return done

    def on_render(self, pidx=None):
        surface = StateVisualizer(tile_size=self.tile_size).render_state(self.env.state,
                                                                         grid=self.env.env.mdp.terrain_mtx,
                                                                         hud_data={"timestep": self.curr_tick,
                                                                                   "score": self.score})
        # self.window = pygame.display.set_mode(surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE)
        self.window.blit(surface, (0, 0))
        pygame.display.flip()

        # Save screenshot
        pygame.image.save(self.window, f"data/screenshots/{self.gif_name}/{self.curr_tick}.png")

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        self.start_screen()
        self.on_init()
        sleep_time = 1000 // (self.fps or 5)

        on_reset = True
        while (self._running):
            if self.agent == 'human':

                while self.human_action is None:
                    for event in pygame.event.get():
                        self.on_event(event)
                    pygame.event.pump()

                action = self.human_action if self.human_action is not None else Action.ACTION_TO_INDEX[Action.STAY]
            else:
                obs = self.env.get_obs(self.env.p_idx, on_reset=False)
                action = self.agent.predict(obs, state=self.env.state, deterministic=self.deterministic)[0]
                # pygame.time.wait(sleep_time)

            done = self.step_env(action)

            self.human_action = None
            if True or self.curr_tick < 200:
                pygame.time.wait(sleep_time)
            else:
                pygame.time.wait(1000)
            self.on_render()
            self.curr_tick += 1

            if done:
                self._running = False

        self.on_cleanup()
        print(f'Trial finished in {self.curr_tick} steps with total reward {self.score}')


    def save_trajectory(self, data_path):
        df = pd.DataFrame(self.trajectory)
        df.to_pickle(data_path / f'{self.layout_name}.{self.trial_id}.pickle')
