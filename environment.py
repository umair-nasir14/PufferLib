#import gym

import numpy as np
import os
import json
import time
import imageio
from PIL import Image, ImageDraw, ImageFont
from rembg import remove
import random
import io
import functools
import shimmy

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.environment
import pufferlib.postprocess
import pufferlib.utils

#

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

class MechEnv(gym.Env):
    def __init__(self, walkable_tiles, tiles_without_char, tiles, str_map_without_chars, str_map, interactive_object_tiles, enemy_tiles, render_mode="rgb_array"):
        super(MechEnv, self).__init__()
        
        self.map_str_without_chars = str_map_without_chars.strip().split('\n')
        self.map_str = str_map.strip().split('\n')
        self.map = [list(row) for row in self.map_str]
        self.map_without_chars = [list(row) for row in self.map_str_without_chars]

        self.tile_size = 16
        self.char_tile_size = 16
        self.tiles = tiles
        self.tiles_without_char = tiles_without_char
        self.action_space = spaces.Discrete(6)  # Up, down, left, right, pick, hit
        self.char_set = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'O': 4, '@': 5, '#': 6, '&': 7}
        self.char_to_int = lambda c: self.char_set.get(c, 0)

        max_width = max(len(row) for row in self.map_str)
        self.observation_space = spaces.Box(
            low=0, 
            high=1,
            shape=(len(self.char_set), len(self.map_str), max_width),  # Use len(self.char_set) for channels
            dtype=np.int32
        )
        
        self.default_walkable_tile = "A"
        self.reward = 0
        self.walkable_tiles = walkable_tiles
        self.interactive_object_tiles = interactive_object_tiles
        self.enemy_tiles = enemy_tiles
        self.picked_objects = []
        self.picked_object = False
        self.npc_tiles = ["&"]
        self.enemy_hit = False
        self.player_health = 100
        self.enemy_health = 100
        self.current_score = 0
        self.explored_tiles = set()
        self.frames = [] 
        self.video_buffer = None
        self.render_mode = render_mode
        self.reset()

    def reset(self, seed=None):
        self.map = [list(row) for row in self.map_str]
        self.map_without_chars = [list(row) for row in self.map_str_without_chars]
        self.grid_width = max(len(row) for row in self.map)
        self.default_walkable_tile = "A"
        self.grid_height = len(self.map)
        self.player_pos = self.find_player_position()
        self.current_tile = self.map_without_chars[self.player_pos[0]][self.player_pos[1]]
        self.picked_object = False
        self.enemy_hit = False
        self.reward = 0
        self.frames = [] 
        self.video_buffer = None
        state = self.get_state()["map"]
        return state, {"success": 0, "action_taken": 0}

    def get_state(self):
        max_width = max(len(row) for row in self.map)
        int_map = [
            [self.char_to_int(char) for char in row] + [0] * (max_width - len(row))
            for row in self.map
        ]
        int_map_array = np.array(int_map, dtype=np.int32)
        one_hot_map = np.eye(len(self.char_set), dtype=np.int32)[int_map_array]
        int_map_with_channel = np.expand_dims(one_hot_map, axis=0)
        int_map_with_channel = np.transpose(int_map_with_channel, (0, 3, 1, 2))
        state = {
            "map": int_map_with_channel,#self.map,
            "player_pos": self.player_pos,
            "player_current_tile": self.current_tile,
            "npc_positions": [(x, y) for y, row in enumerate(self.map) for x, tile in enumerate(row) if tile in self.npc_tiles],
            "enemy_positions": [(x, y) for y, row in enumerate(self.map) for x, tile in enumerate(row) if tile in self.enemy_tiles],
            "interactive_object_positions": [(x, y) for y, row in enumerate(self.map) for x, tile in enumerate(row) if tile in self.interactive_object_tiles],
            "picked_objects": self.picked_objects,
            "current_score": self.current_score,
            "player_health": self.player_health,
            "enemy_health": self.enemy_health,
            "explored_tiles": list(self.explored_tiles)
        }
        
        return state
    def step(self, actions):
        self.reward = 0
        self.enemy_hit = False
        done = False
        if actions < 4:  # Movement actions
            self.reward += self.move_player(actions)
        elif actions == 4:  # Pick action
            self.reward += self.pick_object() 
        elif actions == 5:  # Hit action
            self.reward += self.hit_enemy()
            done = self.enemy_hit
        info = {"success": 1 if self.enemy_hit else 0, "action_taken": actions}
        state = self.get_state()["map"]
        return state, self.reward, done, False, info
    
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        env_img = Image.new('RGBA', (len(self.map[0]) * self.tile_size, len(self.map) * self.tile_size))

        # 1st layer: Default walkable tile
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                tile_img = self.tiles[self.default_walkable_tile].resize((self.tile_size, self.tile_size))
                env_img.paste(tile_img, (j * self.tile_size, i * self.tile_size), tile_img)
        
        # 2nd layer: Map without characters
        for i, row in enumerate(self.map_without_chars):
            for j, tile in enumerate(row):
                if tile in self.tiles and tile != self.default_walkable_tile:
                    tile_img = self.tiles[tile].resize((self.tile_size, self.tile_size))
                    env_img.paste(tile_img, (j * self.tile_size, i * self.tile_size), tile_img)
        
        # 3rd layer: Characters and objects
        for i, row in enumerate(self.map):
            for j, tile in enumerate(row):
                if tile in self.tiles and tile not in self.walkable_tiles:
                    if tile.isalpha():
                        tile_img = self.tiles[tile].resize((self.tile_size, self.tile_size))
                    else:
                        tile_img = self.tiles[tile].resize((self.char_tile_size, self.char_tile_size))
                        # Center the character in the tile
                        x_offset = (self.tile_size - self.char_tile_size) // 2
                        y_offset = (self.tile_size - self.char_tile_size) // 2
                        env_img.paste(tile_img, (j * self.tile_size + x_offset, i * self.tile_size + y_offset), tile_img)
        
        draw = ImageDraw.Draw(env_img)
        font = ImageFont.load_default()
        text = f"Objects Picked: {len(self.picked_objects)}"
        draw.text((10, env_img.size[1] - 20), text, (255, 255, 255), font=font)
        
        frame = np.array(env_img.convert('RGB'))
        self.frames.append(frame)
        return frame
    def _get_video(self, fps=10):
        if not self.frames:
            return None
        
        if self.video_buffer is None:
            self.video_buffer = io.BytesIO()
            with imageio.get_writer(self.video_buffer, format='mp4', fps=fps) as writer:
                for frame in self.frames:
                    writer.append_data(frame)
            self.video_buffer.seek(0)
        
        return self.video_buffer
    def get_video(self):
        return self._get_video()
    
    def get_player_position(self):
        return self.player_pos
    def move_player(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # Up, Down, Left, Right
        dx, dy = moves[action]
        new_row, new_col = self.player_pos[0] + dx, self.player_pos[1] + dy
        reward = 0
        if 0 <= new_row < self.grid_height and 0 <= new_col < self.grid_width:
            new_tile = self.map[new_row][new_col]
            if new_tile in self.walkable_tiles:
                self.update_player_position(new_row, new_col)
                if new_tile == "C": #or new_tile == "D":
                    reward += 1
                #else:
                #    reward -= 0.1
        return reward
        
    def pick_object(self):
        reward = 0
        self.picked_object = False
        #if not self.picked_object:
        adjacent_positions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        for dx, dy in adjacent_positions:
            x, y = self.player_pos
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height:
                target_tile = self.map[new_y][new_x]
                if target_tile in self.interactive_object_tiles:
                    #print("Picked an object!")
                    self.map[new_y][new_x] = self.default_walkable_tile 
                    reward = 0.5
                    self.picked_object = True
                    break 
        return reward
    def hit_enemy(self):
        reward = 0
        self.enemy_hit = False
        adjacent_positions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        for dx, dy in adjacent_positions:
            x, y = self.player_pos
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height:
                target_tile = self.map[new_y][new_x]
                if target_tile in self.enemy_tiles: 
                    #print("Hit an enemy!")
                    if self.picked_object:
                        reward = 5
                        self.enemy_hit = True
                        #self.success_rate += 1
                        self.map[new_y][new_x] = self.default_walkable_tile 
                    else:
                        reward = -1
                    #reward = 10
                    #self.enemy_hit = True
                    
                    break
        return reward
    
    def update_player_position(self, new_row, new_col):
        old_row, old_col = self.player_pos
        self.map[old_row][old_col] = self.current_tile
        self.current_tile = self.map_without_chars[new_row][new_col]
        self.map[new_row][new_col] = '@'
        self.player_pos = (new_row, new_col)
    
    def find_player_position(self):
        for i, row in enumerate(self.map):
            for j, tile in enumerate(row):
                if tile == '@':
                    return (i, j)
        return None
    
    def clone(self):

        new_env = MechEnv(
            walkable_tiles=self.walkable_tiles,
            tiles_without_char=self.tiles_without_char,
            tiles=self.tiles,
            str_map_without_chars='\n'.join(self.map_str_without_chars),
            str_map='\n'.join(self.map_str),
            interactive_object_tiles=self.interactive_object_tiles,
            enemy_tiles=self.enemy_tiles
        )

        new_env.map = [row[:] for row in self.map]
        new_env.map_without_chars = [row[:] for row in self.map_without_chars]
        new_env.player_pos = self.player_pos
        new_env.current_tile = self.current_tile
        new_env.picked_objects = self.picked_objects.copy()
        new_env.enemy_hit = self.enemy_hit
        new_env.player_health = self.player_health
        new_env.enemy_health = self.enemy_health
        new_env.current_score = self.current_score
        new_env.explored_tiles = self.explored_tiles.copy()
        new_env.char_to_int = self.char_to_int.copy()

        return new_env
    
    #def is_terminal(self):
    #    if self.picked_object:
    #        if self.enemy_hit:
    #            return True
    #    return False
    
    def close(self):
        self.map = None
        self.map_without_chars = None
        self.tiles = None
        self.tiles_without_char = None
        self.picked_objects.clear()
        self.explored_tiles.clear()
        self.player_pos = None
        self.current_tile = None
        self.enemy_hit = False
        self.picked_object = False
        self.player_health = 100
        self.enemy_health = 100
        self.current_score = 0
        self.frames = [] 
        self.video_buffer = None

def register_env():
    register(
        id='MechEnv-v0',
        entry_point='environment:MechEnv',
        max_episode_steps=1000,
    )

#def env_creator(name='MechEnv-v0'):
#    return functools.partial(make, name)

def env_creator(name='MechEnv-v1'):
    def create_env():
       return make(name)
    return create_env

#def env_creator(name='MechEnv-v0'):
#    return make(name)

def make(name):

    #register_env()

    str_world = """BBBBBBBBBBB
BAAAAAAAAAB
BAAAAAAAAAB
BCCAAAA@AAB
B#CAOAAAAAB
BBBBBBBBBBB"""
    str_map_wo_chars = """BBBBBBBBBBB
BAAAAAAAAAB
BAAAAAAAAAB
BCCCCCAAAAB
BACCOCAAAAB
BBBBBBBBBBB"""

    walkables = ['A']
    interactive_object_tiles = ['O']
    player_tile = '@'
    enemy_tiles = ["#"]
    npc_tiles = ["&"]
    env_image = dict()

    folder_path = r"C:/Users/DELL/Projects/Research/game_mech_desc/world_tileset_data"


    env_image["A"] = Image.open(r"C:/Users/DELL/Projects/Research/game_mech_desc/world_tileset_data/td_world_floor_grass_c.png").convert("RGBA")
    env_image["B"] = Image.open(r"C:/Users/DELL/Projects/Research/game_mech_desc/world_tileset_data/td_world_wall_stone_h_a.png").convert("RGBA")
    env_image["C"] = Image.open(r"C:/Users/DELL/Projects/Research/game_mech_desc/world_tileset_data/td_world_floor_grass_c.png").convert("RGBA")
    #env_image["D"] = Image.open(r"C:/Users/DELL/Projects/Research/game_mech_desc/world_tileset_data/td_world_floor_grass_c.png").convert("RGBA")
    env_image["O"] = Image.open(r"C:/Users/DELL/Projects/Research/game_mech_desc/world_tileset_data/td_world_chest.png").convert("RGBA")
    env_image["@"] = Image.open(r"C:/Users/DELL/Projects/Research/game_mech_desc/character_sprite_data/td_monsters_archer_d1.png").convert("RGBA")
    env_image["#"] = Image.open(r"C:/Users/DELL/Projects/Research/game_mech_desc/character_sprite_data/td_monsters_witch_d1.png").convert("RGBA")
    env_image["&"] = Image.open(r"C:/Users/DELL/Projects/Research/game_mech_desc/character_sprite_data/td_monsters_goblin_captain_d1.png").convert("RGBA")

    #env = gym.make(name,
    #               walkable_tiles=walkables, 
    #              tiles_without_char=str_map_wo_chars, 
    #              tiles=env_image, 
    #              str_map_without_chars=str_map_wo_chars, 
    #              str_map=str_world, 
    #              interactive_object_tiles=interactive_object_tiles, 
    #              enemy_tiles=enemy_tiles)
    
    
    
    env = MechEnv(walkable_tiles=walkables, 
                  tiles_without_char=str_map_wo_chars, 
                  tiles=env_image, 
                  str_map_without_chars=str_map_wo_chars, 
                  str_map=str_world, 
                  interactive_object_tiles=interactive_object_tiles, 
                  enemy_tiles=enemy_tiles,
                  render_mode="rgb_array")
    env.reset = pufferlib.utils.silence_warnings(env.reset)
    #env = shimmy.GymV21CompatibilityV0(env=env)
    env = GymCompatibilityWrapper(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env)

class GymCompatibilityWrapper(gym.Wrapper):
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, reward, done, truncated, info
        elif len(result) == 4:
            obs, reward, done, info = result
            truncated = False 
            return obs, reward, done, truncated, info
        else:
            raise ValueError(f"Unexpected number of values returned from step(): {len(result)}")

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            return obs, info
        else:
            return result, {}
        
    def render(self, mode=None):
        if mode is None:
            mode = self.render_mode
        return self.env.render()



