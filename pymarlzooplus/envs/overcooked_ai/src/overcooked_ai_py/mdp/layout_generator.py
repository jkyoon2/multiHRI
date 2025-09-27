import copy
import random
from collections import defaultdict, deque

import numpy as np

from pymarlzooplus.envs.overcooked_ai.src.overcooked_ai_py.mdp.actions import Action, Direction
from pymarlzooplus.envs.overcooked_ai.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Recipe
from pymarlzooplus.envs.overcooked_ai.src.overcooked_ai_py.utils import rnd_int_uniform, rnd_uniform

EMPTY = " "
COUNTER = "X"
ONION_DISPENSER = "O"
TOMATO_DISPENSER = "T"
POT = "P"
DISH_DISPENSER = "D"
SERVING_LOC = "S"

CODE_TO_TYPE = {
    0: EMPTY,
    1: COUNTER,
    2: ONION_DISPENSER,
    3: TOMATO_DISPENSER,
    4: POT,
    5: DISH_DISPENSER,
    6: SERVING_LOC,
}
TYPE_TO_CODE = {v: k for k, v in CODE_TO_TYPE.items()}

DIRECTION_PRIORITY = ["right", "bottom", "left", "top"]
OPPOSITE_DIRECTIONS = {
    "left": "right",
    "right": "left",
    "top": "bottom",
    "bottom": "top",
}
DIRECTION_VECTORS = {
    "left": (-1, 0),
    "right": (1, 0),
    "top": (0, -1),
    "bottom": (0, 1),
}


def mdp_fn_random_choice(mdp_fn_choices):
    assert type(mdp_fn_choices) is list and len(mdp_fn_choices) > 0
    return random.choice(mdp_fn_choices)


"""
size_bounds: (min_layout_size, max_layout_size)
prop_empty: (min, max) proportion of empty space in generated layout
prop_feats: (min, max) proportion of counters with features on them
"""

DEFAULT_MDP_GEN_PARAMS = {
    "inner_shape": (5, 4),
    "prop_empty": 0.95,
    "prop_feats": 0.1,
    "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
    "recipe_values": [20],
    "recipe_times": [20],
    "display": False,
}


def DEFAILT_PARAMS_SCHEDULE_FN(outside_information):
    mdp_default_gen_params = {
        "inner_shape": (5, 4),
        "prop_empty": 0.95,
        "prop_feats": 0.1,
        "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
        "recipe_values": [20],
        "recipe_times": [20],
        "display": False,
    }
    return mdp_default_gen_params


class MDPParamsGenerator(object):
    def __init__(self, params_schedule_fn):
        """
        params_schedule_fn (callable): the function to produce a set of mdp_params for a specific layout
        """
        assert callable(
            params_schedule_fn
        ), "params scheduling function must be a callable"
        self.params_schedule_fn = params_schedule_fn

    @staticmethod
    def from_fixed_param(mdp_params_always):
        # s naive schedule function that always return the same set of parameter
        naive_schedule_fn = lambda _ignored: mdp_params_always
        return MDPParamsGenerator(naive_schedule_fn)

    def generate(self, outside_information={}):
        """
        generate a set of mdp_params that can be used to generate a mdp
        outside_information (dict): passing in outside information
        """
        assert type(outside_information) is dict
        mdp_params = self.params_schedule_fn(outside_information)
        return mdp_params


DEFAULT_FEATURE_TYPES = (
    POT,
    ONION_DISPENSER,
    DISH_DISPENSER,
    SERVING_LOC,
)  # NOTE: TOMATO_DISPENSER is disabled by default


class LayoutGenerator(object):
    # NOTE: This class hasn't been tested extensively.

    def __init__(self, mdp_params_generator, outer_shape=(5, 4)):
        """
        Defines a layout generator that will return OvercoookedGridworld instances
        using mdp_params_generator
        """
        self.mdp_params_generator = mdp_params_generator
        self.outer_shape = outer_shape

    @staticmethod
    def mdp_gen_fn_from_dict(
        mdp_params, outer_shape=None, mdp_params_schedule_fn=None
    ):
        """
        mdp_params: one set of fixed mdp parameter used by the enviroment
        outer_shape: outer shape of the environment
        mdp_params_schedule_fn: the schedule for varying mdp params
        """
        # if outer_shape is not defined, we have to be using one of the default layouts from the names' bank
        if outer_shape is None:
            assert type(mdp_params) is dict and "layout_name" in mdp_params
            mdp = OvercookedGridworld.from_layout_name(**mdp_params)
            mdp_fn = lambda _ignored: mdp
        else:
            # there is no schedule, we are using the same set of mdp_params all the time
            if mdp_params_schedule_fn is None:
                assert mdp_params is not None
                mdp_pg = MDPParamsGenerator.from_fixed_param(
                    mdp_params_always=mdp_params
                )
            else:
                assert mdp_params is None, (
                    "please remove the mdp_params from the variable, "
                    "because mdp_params_schedule_fn exist and we will "
                    "always use the schedule_fn if it exist"
                )
                mdp_pg = MDPParamsGenerator(
                    params_schedule_fn=mdp_params_schedule_fn
                )
            lg = LayoutGenerator(mdp_pg, outer_shape)
            mdp_fn = lg.generate_padded_mdp
        return mdp_fn

    def generate_padded_mdp(self, outside_information={}):
        """
        Return a PADDED MDP with mdp params specified in self.mdp_params
        """
        mdp_gen_params = self.mdp_params_generator.generate(
            outside_information
        )

        outer_shape = self.outer_shape
        if (
            "layout_name" in mdp_gen_params.keys()
            and mdp_gen_params["layout_name"] is not None
        ):
            mdp = OvercookedGridworld.from_layout_name(**mdp_gen_params)
            mdp_generator_fn = lambda: self.padded_mdp(mdp)
        else:
            # Check if this is forced coordination mode
            if mdp_gen_params.get("coordination_mode") == "forced":
                # For forced coordination, different parameters are required
                required_keys = [
                    "n_sets",
                    "inner_shape",
                    "prop_empty",
                    "feature_distribution",
                    "player_distribution",
                    "display",
                ]
                if not mdp_gen_params.get("generate_all_orders"):
                    required_keys.append("start_all_orders")
            else:
                # Default optional coordination mode
                required_keys = [
                    "inner_shape",
                    "prop_empty",
                    "prop_feats",
                    "display",
                ]
                if not mdp_gen_params.get("generate_all_orders"):
                    required_keys.append("start_all_orders")
            
            missing_keys = [
                k for k in required_keys if k not in mdp_gen_params.keys()
            ]
            if len(missing_keys) != 0:
                print("missing keys dict", mdp_gen_params)
            assert (
                len(missing_keys) == 0
            ), "These keys were missing from the mdp_params: {}".format(
                missing_keys
            )
            # Handle different inner_shape formats
            if mdp_gen_params.get("coordination_mode") == "forced":
                # For forced coordination, inner_shape is a list of shapes
                inner_shapes = mdp_gen_params["inner_shape"]
                for i, inner_shape in enumerate(inner_shapes):
                    assert (
                        inner_shape[0] <= outer_shape[0]
                        and inner_shape[1] <= outer_shape[1]
                    ), f"inner_shape[{i}] {inner_shape} cannot fit into the outer_shape {outer_shape}"
            else:
                # For optional coordination, inner_shape is a single shape
                inner_shape = mdp_gen_params["inner_shape"]
                assert (
                    inner_shape[0] <= outer_shape[0]
                    and inner_shape[1] <= outer_shape[1]
                ), "inner_shape cannot fit into the outershap"
            layout_generator = LayoutGenerator(
                self.mdp_params_generator, outer_shape=self.outer_shape
            )

            if "feature_types" not in mdp_gen_params:
                mdp_gen_params["feature_types"] = DEFAULT_FEATURE_TYPES

            mdp_generator_fn = lambda: layout_generator.make_new_layout(
                mdp_gen_params
            )
        return mdp_generator_fn()

    @staticmethod
    def create_base_params(mdp_gen_params):
        assert mdp_gen_params.get("start_all_orders") or mdp_gen_params.get(
            "generate_all_orders"
        )
        mdp_gen_params = LayoutGenerator.add_generated_mdp_params_orders(
            mdp_gen_params
        )
        recipe_params = {
            "start_all_orders": mdp_gen_params["start_all_orders"]
        }
        if mdp_gen_params.get("start_bonus_orders"):
            recipe_params["start_bonus_orders"] = mdp_gen_params[
                "start_bonus_orders"
            ]
        if "recipe_values" in mdp_gen_params:
            recipe_params["recipe_values"] = mdp_gen_params["recipe_values"]
        if "recipe_times" in mdp_gen_params:
            recipe_params["recipe_times"] = mdp_gen_params["recipe_times"]
        return recipe_params

    @staticmethod
    def add_generated_mdp_params_orders(mdp_params):
        """
        adds generated parameters (i.e., generated orders) to mdp_params,
        returns onchanged copy of mdp_params when there is no "generate_all_orders" and "generate_bonus_orders" keys inside mdp_params
        """
        mdp_params = copy.deepcopy(mdp_params)
        if mdp_params.get("generate_all_orders"):
            all_orders_kwargs = copy.deepcopy(
                mdp_params["generate_all_orders"]
            )

            if all_orders_kwargs.get("recipes"):
                all_orders_kwargs["recipes"] = [
                    Recipe.from_dict(r) for r in all_orders_kwargs["recipes"]
                ]

            all_recipes = Recipe.generate_random_recipes(**all_orders_kwargs)
            mdp_params["start_all_orders"] = [r.to_dict() for r in all_recipes]
        else:
            Recipe.configure({})
            all_recipes = Recipe.ALL_RECIPES

        if mdp_params.get("generate_bonus_orders"):
            bonus_orders_kwargs = copy.deepcopy(
                mdp_params["generate_bonus_orders"]
            )

            if not bonus_orders_kwargs.get("recipes"):
                bonus_orders_kwargs["recipes"] = all_recipes

            bonus_recipes = Recipe.generate_random_recipes(
                **bonus_orders_kwargs
            )
            mdp_params["start_bonus_orders"] = [
                r.to_dict() for r in bonus_recipes
            ]
        return mdp_params

    def padded_mdp(self, mdp, display=False):
        """Returns a padded MDP from an MDP"""
        grid = Grid.from_mdp(mdp)
        padded_grid = self.embed_grid(grid)

        start_positions = self.get_random_starting_positions(padded_grid)
        mdp_grid = self.padded_grid_to_layout_grid(
            padded_grid, start_positions, display=display
        )
        return OvercookedGridworld.from_grid(mdp_grid)

    def make_new_layout(self, mdp_gen_params):
        # Check if forced coordination mode is requested
        if mdp_gen_params.get("coordination_mode") == "forced":
            return self.generate_forced_coordination_layout(mdp_gen_params)
        else:
            # Default optional coordination mode
            return self.make_disjoint_sets_layout(
                inner_shape=mdp_gen_params["inner_shape"],
                prop_empty=mdp_gen_params["prop_empty"],
                prop_features=mdp_gen_params["prop_feats"],
                base_param=LayoutGenerator.create_base_params(mdp_gen_params),
                feature_types=mdp_gen_params["feature_types"],
                display=mdp_gen_params["display"],
            )

    def generate_forced_coordination_layout(self, mdp_gen_params):
        """
        Forced Coordination 모드를 위한 맵을 생성합니다.
        - 여러 개의 분리된 방(inner_shape 리스트)을 생성
        - 방 사이에 공유 카운터 공간을 보장하며 배치
        - feature_distribution에 따라 시설을 전략적으로 배치
        - player_distribution에 따라 플레이어를 전략적으로 배치
        """
        # 1. 파라미터 유효성 검사
        n_sets = mdp_gen_params['n_sets']
        assert len(mdp_gen_params['inner_shape']) == n_sets
        assert len(mdp_gen_params['feature_distribution']) == n_sets
        assert len(mdp_gen_params['player_distribution']) == n_sets

        # 2. 각 방(Room)을 개별적으로 생성
        rooms = []
        for i in range(n_sets):
            room_shape = mdp_gen_params['inner_shape'][i]
            # prop_empty는 각 방에 대해 동일하게 적용하거나, 파라미터로 따로 받을 수 있음
            room_grid = self._create_single_room(room_shape, mdp_gen_params.get("prop_empty", 0.7))
            rooms.append(room_grid)

        # 3. 전체 맵 그리드에 방들을 '공유 카운터'를 보장하며 배치
        outer_grid = Grid(self.outer_shape)
        room_placements = self._place_rooms_with_shared_boundary(
            outer_grid, rooms, mdp_gen_params.get('shared_counter_prop', 0.1)
        )

        # 4. 'feature_distribution'에 따라 시설 배치
        self._place_features_strategically(
            outer_grid, room_placements, mdp_gen_params['feature_distribution']
        )

        # 5. 'player_distribution'에 따라 플레이어 배치
        start_positions = self._place_players_strategically(
            outer_grid, room_placements, mdp_gen_params['player_distribution']
        )
        
        # 6. 최종 MDP 객체 생성 및 반환
        base_param = LayoutGenerator.create_base_params(mdp_gen_params)
        mdp_grid = self.padded_grid_to_layout_grid(outer_grid, start_positions, display=mdp_gen_params.get("display", False))
        return OvercookedGridworld.from_grid(mdp_grid, base_param)

    def _create_single_room(self, shape, prop_empty):
        """
        기존의 dig_space_with_disjoint_sets 로직을 재사용해서 단일 방 하나를 생성합니다.
        이 방 자체는 완전히 연결되어야 합니다 (num_sets=1).
        """
        grid = Grid(shape)
        # 이 방 자체는 완전히 연결되어야 함 (num_sets=1)
        self.dig_space_with_disjoint_sets(grid, prop_empty) 
        return grid

    def _place_rooms_with_shared_boundary(self, outer_grid, rooms, shared_counter_prop):
        """
        생성된 여러 방을 전체 맵에 배치합니다.
        방과 방 사이에 벽이 있다는 것을 보장하고, 
        아이템 교환이 가능한 공유 카운터가 특정 비율 이상이 되도록 보장합니다.
        """
        room_placements = {}
        current_x = 1  # 시작 위치 (벽 고려)
        
        for i, room in enumerate(rooms):
            room_shape = room.shape
            room_width, room_height = room_shape[0], room_shape[1]
            
            # 방을 outer_grid에 배치
            start_x = current_x
            start_y = 1  # 상단 벽 고려
            
            # 방의 내용을 outer_grid에 복사
            for x in range(room_width):
                for y in range(room_height):
                    outer_x = start_x + x
                    outer_y = start_y + y
                    if outer_x < outer_grid.shape[0] - 1 and outer_y < outer_grid.shape[1] - 1:
                        outer_grid.mtx[outer_x][outer_y] = room.mtx[x][y]
            
            # 방 배치 정보 저장
            room_placements[i] = {
                'start_pos': (start_x, start_y),
                'end_pos': (start_x + room_width - 1, start_y + room_height - 1),
                'shape': room_shape
            }
            
            # 다음 방을 위한 위치 업데이트 (방 + 벽 + 공유 공간)
            current_x += room_width + 2  # 방 너비 + 벽 2칸
        
        # 공유 카운터 보장 로직
        self._ensure_shared_counters(outer_grid, room_placements, shared_counter_prop)
        
        return room_placements

    def _ensure_shared_counters(self, outer_grid, room_placements, shared_counter_prop):
        """
        방들 사이에 충분한 공유 카운터가 있는지 확인하고, 부족하면 추가로 생성합니다.
        """
        if len(room_placements) < 2:
            return  # 방이 2개 미만이면 공유 카운터가 필요 없음
        
        # 각 방 쌍에 대해 공유 카운터 확인 및 생성
        for i in range(len(room_placements) - 1):
            room1_info = room_placements[i]
            room2_info = room_placements[i + 1]
            
            # 두 방 사이의 경계 영역 찾기
            boundary_area = self._find_boundary_area(room1_info, room2_info)
            
            # 필요한 최소 공유 카운터 수 계산
            room1_area = room1_info['shape'][0] * room1_info['shape'][1]
            room2_area = room2_info['shape'][0] * room2_info['shape'][1]
            min_shared_counters = int(np.sqrt(room1_area + room2_area)) - 1
            
            # 현재 공유 카운터 수 확인
            current_shared = self._count_shared_counters(outer_grid, room1_info, room2_info)
            
            # 부족하면 추가로 생성
            if current_shared < min_shared_counters:
                self._create_additional_shared_counters(outer_grid, room1_info, room2_info, 
                                                      min_shared_counters - current_shared)

    def _find_boundary_area(self, room1_info, room2_info):
        """두 방 사이의 경계 영역을 찾습니다."""
        room1_end_x = room1_info['end_pos'][0]
        room2_start_x = room2_info['start_pos'][0]
        
        # 두 방 사이의 1칸 벽 영역
        boundary_x = room1_end_x + 1
        
        # 두 방의 높이 범위 계산
        min_y = min(room1_info['start_pos'][1], room2_info['start_pos'][1])
        max_y = max(room1_info['end_pos'][1], room2_info['end_pos'][1])
        
        return {
            'x': boundary_x,
            'y_range': (min_y, max_y)
        }

    def _get_fringe_counters_for_room(self, outer_grid, room_info):
        """
        특정 방의 fringe에 해당하는 카운터들을 찾습니다.
        방 영역 내의 모든 빈공간에 인접한 카운터들의 좌표 집합을 반환합니다.
        """
        fringe_counters = set()
        start_pos = room_info['start_pos']
        end_pos = room_info['end_pos']
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        # 방 영역 내의 모든 빈공간을 찾음
        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                location = (x, y)
                if (outer_grid.is_in_bounds(location) and 
                    outer_grid.location_is_empty(location)):
                    
                    # 이 빈공간에 인접한 모든 카운터들을 fringe에 추가
                    for neighbor in outer_grid.get_near_locations(location):
                        if (outer_grid.is_in_bounds(neighbor) and 
                            outer_grid.terrain_at_loc(neighbor) == TYPE_TO_CODE[COUNTER]):
                            fringe_counters.add(neighbor)
        
        return fringe_counters

    def _count_shared_counters(self, outer_grid, room1_info, room2_info):
        """두 방 사이의 실제 공유 카운터 수를 계산합니다."""
        # 각 방의 fringe 카운터 집합을 구함
        fringe1 = self._get_fringe_counters_for_room(outer_grid, room1_info)
        fringe2 = self._get_fringe_counters_for_room(outer_grid, room2_info)

        # 두 집합의 교집합(intersection)이 바로 공유 카운터임
        shared_counters = fringe1.intersection(fringe2)

        return len(shared_counters)

    def _create_additional_shared_counters(self, outer_grid, room1_info, room2_info, needed_count):
        """
        추가 공유 카운터를 생성합니다.
        두 방의 fringe 영역 중 근접한 fringe들을 찾아서 빈 공간으로 만들고,
        공유 카운터의 개수가 최소 기준을 충족할 때까지 반복합니다.
        """
        created = 0
        max_attempts = 50  # 무한 루프 방지
        
        for attempt in range(max_attempts):
            if created >= needed_count:
                break
                
            # 현재 공유 카운터 수 확인
            current_shared = self._count_shared_counters(outer_grid, room1_info, room2_info)
            if current_shared >= needed_count:
                break
            
            # 두 방의 fringe 카운터 집합을 구함
            fringe1 = self._get_fringe_counters_for_room(outer_grid, room1_info)
            fringe2 = self._get_fringe_counters_for_room(outer_grid, room2_info)
            
            # 공유 가능한 카운터 후보들을 찾음 (한쪽 fringe에만 속한 카운터들)
            potential_shared = fringe1.symmetric_difference(fringe2)
            
            # 공유 카운터로 만들 수 있는 위치를 찾음
            for counter_pos in potential_shared:
                if created >= needed_count:
                    break
                    
                # 이 카운터가 두 방 모두에서 접근 가능한지 확인
                if self._can_become_shared_counter(outer_grid, counter_pos, room1_info, room2_info):
                    # 카운터를 빈 공간으로 만들어서 공유 공간으로 전환
                    outer_grid.change_location(counter_pos, EMPTY)
                    created += 1
                    
                    # 공유 카운터 수가 증가했는지 확인
                    new_shared = self._count_shared_counters(outer_grid, room1_info, room2_info)
                    if new_shared > current_shared:
                        break  # 성공적으로 공유 카운터가 생성됨

    def _can_become_shared_counter(self, outer_grid, counter_pos, room1_info, room2_info):
        """
        특정 카운터 위치가 두 방 모두에서 접근 가능한 공유 카운터가 될 수 있는지 확인합니다.
        """
        # 카운터 주변에 두 방 모두의 빈 공간이 인접해 있는지 확인
        neighbors = outer_grid.get_near_locations(counter_pos)
        
        room1_accessible = False
        room2_accessible = False
        
        for neighbor in neighbors:
            if not outer_grid.is_in_bounds(neighbor):
                continue
                
            if outer_grid.location_is_empty(neighbor):
                # 이 빈 공간이 어느 방에 속하는지 확인
                if self._is_position_in_room(neighbor, room1_info):
                    room1_accessible = True
                if self._is_position_in_room(neighbor, room2_info):
                    room2_accessible = True
        
        return room1_accessible and room2_accessible

    def _is_position_in_room(self, position, room_info):
        """특정 위치가 방 영역 내에 있는지 확인합니다."""
        x, y = position
        start_x, start_y = room_info['start_pos']
        end_x, end_y = room_info['end_pos']
        
        return start_x <= x <= end_x and start_y <= y <= end_y

    def _place_features_strategically(self, outer_grid, room_placements, feature_distribution):
        """
        feature_distribution 딕셔너리에 따라 각 방의 정해진 위치에 시설을 배치합니다.
        
        feature_distribution 예시:
        {
            0: [POT, ONION_DISPENSER],  # 방 0에 냄비와 양파 디스펜서
            1: [DISH_DISPENSER, SERVING_LOC],  # 방 1에 접시 디스펜서와 서빙 위치
        }
        """
        for room_id, features in feature_distribution.items():
            if room_id not in room_placements:
                continue
                
            room_info = room_placements[room_id]
            start_pos = room_info['start_pos']
            end_pos = room_info['end_pos']
            
            # 해당 방 영역 내에서 유효한 시설 위치 찾기
            valid_locations = self._get_valid_feature_locations_in_room(
                outer_grid, start_pos, end_pos
            )
            
            # 지정된 시설들을 배치
            for i, feature in enumerate(features):
                if i < len(valid_locations):
                    location = valid_locations[i]
                    outer_grid.add_feature(location, feature)

    def _get_valid_feature_locations_in_room(self, outer_grid, start_pos, end_pos):
        """특정 방 영역 내에서 시설을 배치할 수 있는 유효한 위치들을 찾습니다."""
        valid_locations = []
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                location = (x, y)
                if outer_grid.is_valid_feature_location(location):
                    valid_locations.append(location)
        
        # 무작위로 섞어서 다양한 배치 가능
        np.random.shuffle(valid_locations)
        return valid_locations

    def _place_players_strategically(self, outer_grid, room_placements, player_distribution):
        """
        player_distribution 딕셔너리에 따라 각 방에 플레이어를 배치합니다.
        
        player_distribution 예시:
        {
            0: [0],  # 방 0에 플레이어 0
            1: [1],  # 방 1에 플레이어 1
        }
        
        Returns:
            list: 플레이어들의 시작 위치 리스트
        """
        start_positions = [None] * self._get_total_players(player_distribution)
        
        for room_id, player_ids in player_distribution.items():
            if room_id not in room_placements:
                continue
                
            room_info = room_placements[room_id]
            start_pos = room_info['start_pos']
            end_pos = room_info['end_pos']
            
            # 해당 방 영역 내에서 빈 공간들 찾기
            empty_locations = self._get_empty_locations_in_room(
                outer_grid, start_pos, end_pos
            )
            
            # 각 플레이어를 해당 방의 빈 공간에 배치
            for i, player_id in enumerate(player_ids):
                if i < len(empty_locations) and player_id < len(start_positions):
                    location = empty_locations[i]
                    start_positions[player_id] = location
        
        # None이 있는 경우 기본 위치로 채우기
        for i, pos in enumerate(start_positions):
            if pos is None:
                start_positions[i] = outer_grid.get_random_empty_location()
        
        return start_positions

    def _get_total_players(self, player_distribution):
        """총 플레이어 수를 계산합니다."""
        max_player_id = -1
        for player_ids in player_distribution.values():
            if player_ids:
                max_player_id = max(max_player_id, max(player_ids))
        return max_player_id + 1 if max_player_id >= 0 else 0

    def _get_empty_locations_in_room(self, outer_grid, start_pos, end_pos):
        """특정 방 영역 내에서 빈 공간들을 찾습니다."""
        empty_locations = []
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                location = (x, y)
                if (outer_grid.is_in_bounds(location) and 
                    outer_grid.location_is_empty(location)):
                    empty_locations.append(location)
        
        # 무작위로 섞어서 다양한 배치 가능
        np.random.shuffle(empty_locations)
        return empty_locations

    def make_disjoint_sets_layout(
        self,
        inner_shape,
        prop_empty,
        prop_features,
        base_param,
        feature_types=DEFAULT_FEATURE_TYPES,
        display=True,
    ):
        grid = Grid(inner_shape)
        self.dig_space_with_disjoint_sets(grid, prop_empty)
        self.add_features(grid, prop_features, feature_types)

        padded_grid = self.embed_grid(grid)
        start_positions = self.get_random_starting_positions(padded_grid)
        mdp_grid = self.padded_grid_to_layout_grid(
            padded_grid, start_positions, display=display
        )
        return OvercookedGridworld.from_grid(mdp_grid, base_param)

    @staticmethod
    def padded_grid_to_layout_grid(
            padded_grid, start_positions, display=False
    ):
        if display:
            print("Generated layout")
            print(padded_grid)

        # Start formatting to the actual OvercookedGridworld input type
        mdp_grid = padded_grid.convert_to_string()

        for i, pos in enumerate(start_positions):
            x, y = pos
            mdp_grid[y][x] = str(i + 1)

        return mdp_grid

    def embed_grid(self, grid):
        """Randomly embeds a smaller grid in a grid of size self.outer_shape"""
        # Check that smaller grid fits
        assert all(grid.shape <= self.outer_shape)

        padded_grid = Grid(self.outer_shape)
        x_leeway, y_leeway = self.outer_shape - grid.shape
        starting_x = np.random.randint(0, x_leeway) if x_leeway else 0
        starting_y = np.random.randint(0, y_leeway) if y_leeway else 0

        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                item = grid.terrain_at_loc((x, y))
                # Abstraction violation
                padded_grid.mtx[x + starting_x][y + starting_y] = item

        return padded_grid

    @staticmethod
    def dig_space_with_disjoint_sets(grid, prop_empty):
        dsets = DisjointSets([])
        while not (
            grid.proportion_empty() > prop_empty and dsets.num_sets == 1
        ):
            valid_dig_location = False
            while not valid_dig_location:
                loc = grid.get_random_interior_location()
                valid_dig_location = grid.is_valid_dig_location(loc)

            grid.dig(loc)
            dsets.add_singleton(loc)

            for neighbour in grid.get_near_locations(loc):
                if dsets.contains(neighbour):
                    dsets.union(neighbour, loc)

    def make_fringe_expansion_layout(self, shape, prop_empty=0.1):
        grid = Grid(shape)
        self.dig_space_with_fringe_expansion(grid, prop_empty)
        self.add_features(grid)

    @staticmethod
    def dig_space_with_fringe_expansion(grid, prop_empty=0.1):
        starting_location = grid.get_random_interior_location()
        fringe = Fringe(grid)
        fringe.add(starting_location)

        while grid.proportion_empty() < prop_empty:
            curr_location = fringe.pop()
            grid.dig(curr_location)

            for location in grid.get_near_locations(curr_location):
                if grid.is_valid_dig_location(location):
                    fringe.add(location)

    @staticmethod
    def add_features(
            grid, prop_features=0, feature_types=DEFAULT_FEATURE_TYPES
    ):
        """
        Places one round of basic features and then adds random features
        until prop_features of valid locations are filled"""

        valid_locations = grid.valid_feature_locations()
        np.random.shuffle(valid_locations)
        assert len(valid_locations) > len(feature_types)

        num_features_placed = 0
        for location in valid_locations:
            current_prop = num_features_placed / len(valid_locations)
            if num_features_placed < len(feature_types):
                grid.add_feature(location, feature_types[num_features_placed])
            elif current_prop >= prop_features:
                break
            else:
                random_feature = np.random.choice(feature_types)
                grid.add_feature(location, random_feature)
            num_features_placed += 1

    @staticmethod
    def get_random_starting_positions(grid, divider_x=None):
        pos0 = grid.get_random_empty_location()
        pos1 = grid.get_random_empty_location()
        # NOTE: Assuming more than 1 empty location, hacky code
        while pos0 == pos1:
            pos0 = grid.get_random_empty_location()
        return pos0, pos1


class Grid(object):
    def __init__(self, shape):
        assert len(shape) == 2, "Grid must be 2 dimensional"
        grid = (np.ones(shape) * TYPE_TO_CODE[COUNTER]).astype(int)
        self.mtx = grid
        self.shape = np.array(shape)
        self.width = shape[0]
        self.height = shape[1]

    @staticmethod
    def from_mdp(mdp):
        terrain_matrix = np.array(mdp.terrain_mtx)
        mdp_grid = Grid((terrain_matrix.shape[1], terrain_matrix.shape[0]))
        for y in range(terrain_matrix.shape[0]):
            for x in range(terrain_matrix.shape[1]):
                feature = terrain_matrix[y][x]
                mdp_grid.mtx[x][y] = TYPE_TO_CODE[feature]
        return mdp_grid

    def terrain_at_loc(self, location):
        x, y = location
        return self.mtx[x][y]

    def dig(self, location):
        assert self.is_valid_dig_location(location)
        self.change_location(location, EMPTY)

    def add_feature(self, location, feature_string):
        assert self.is_valid_feature_location(location)
        self.change_location(location, feature_string)

    def change_location(self, location, feature_string):
        x, y = location
        self.mtx[x][y] = TYPE_TO_CODE[feature_string]

    def proportion_empty(self):
        flattened_grid = self.mtx.flatten()
        num_eligible = len(flattened_grid) - 2 * sum(self.shape) + 4
        num_empty = sum(
            [1 for x in flattened_grid if x == TYPE_TO_CODE[EMPTY]]
        )
        return float(num_empty) / num_eligible

    def get_near_locations(self, location):
        """Get neighbouring locations to the passed in location"""
        near_locations = []
        for d in Direction.ALL_DIRECTIONS:
            new_location = Action.move_in_direction(location, d)
            if self.is_in_bounds(new_location):
                near_locations.append(new_location)
        return near_locations

    def is_in_bounds(self, location):
        x, y = location
        return x >= 0 and y >= 0 and x < self.shape[0] and y < self.shape[1]

    def is_valid_dig_location(self, location):
        x, y = location

        # If already empty
        if self.location_is_empty(location):
            return False

        # If one of the edges of the map, or outside the map
        if (
            x <= 0
            or y <= 0
            or x >= self.shape[0] - 1
            or y >= self.shape[1] - 1
        ):
            return False
        return True

    def valid_feature_locations(self):
        valid_locations = []
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                location = (x, y)
                if self.is_valid_feature_location(location):
                    valid_locations.append(location)
        return np.array(valid_locations)

    def is_valid_feature_location(self, location):
        x, y = location

        # If is empty or has a feature on it
        if not self.mtx[x][y] == TYPE_TO_CODE[COUNTER]:
            return False

        # If outside the map
        if not self.is_in_bounds(location):
            return False

        # If location is next to at least one empty square
        if any(
            [
                loc
                for loc in self.get_near_locations(location)
                if CODE_TO_TYPE[self.terrain_at_loc(loc)] == EMPTY
            ]
        ):
            return True
        else:
            return False

    def location_is_empty(self, location):
        x, y = location
        return self.mtx[x][y] == TYPE_TO_CODE[EMPTY]

    def get_random_interior_location(self):
        rand_x = np.random.randint(low=1, high=self.shape[0] - 1)
        rand_y = np.random.randint(low=1, high=self.shape[1] - 1)
        return rand_x, rand_y

    def get_random_empty_location(self):
        is_empty = False
        while not is_empty:
            loc = self.get_random_interior_location()
            is_empty = self.location_is_empty(loc)
        return loc

    def convert_to_string(self):
        rows = []
        for y in range(self.shape[1]):
            column = []
            for x in range(self.shape[0]):
                column.append(CODE_TO_TYPE[self.mtx[x][y]])
            rows.append(column)
        string_grid = np.array(rows)
        assert np.array_equal(
            string_grid.T.shape, self.shape
        ), "{} vs {}".format(string_grid.shape, self.shape)
        return string_grid

    def __repr__(self):
        s = ""
        for y in range(self.shape[1]):
            for x in range(self.shape[0]):
                s += CODE_TO_TYPE[self.mtx[x][y]]
                s += " "
            s += "\n"
        return s


class Fringe(object):
    def __init__(self, grid):
        self.fringe_list = []
        self.distribution = []
        self.grid = grid

    def add(self, item):
        if item not in self.fringe_list:
            self.fringe_list.append(item)
            self.update_probs()

    def pop(self):
        assert len(self.fringe_list) > 0
        choice_idx = np.random.choice(
            len(self.fringe_list), p=self.distribution
        )
        removed_pos = self.fringe_list.pop(choice_idx)
        self.update_probs()
        return removed_pos

    def update_probs(self):
        self.distribution = np.ones(len(self.fringe_list)) / len(
            self.fringe_list
        )


class DisjointSets(object):
    """A simple implementation of the Disjoint Sets data structure.

    Implements path compression but not union-by-rank.

    Taken from https://github.com/HumanCompatibleAI/planner-inference
    """

    def __init__(self, elements):
        self.num_elements = len(elements)
        self.num_sets = len(elements)
        self.parents = {element: element for element in elements}

    def is_connected(self):
        return self.num_sets == 1

    def get_num_elements(self):
        return self.num_elements

    def contains(self, element):
        return element in self.parents

    def add_singleton(self, element):
        assert not self.contains(element)
        self.num_elements += 1
        self.num_sets += 1
        self.parents[element] = element

    def find(self, element):
        parent = self.parents[element]
        if element == parent:
            return parent

        result = self.find(parent)
        self.parents[element] = result
        return result

    def union(self, e1, e2):
        p1, p2 = map(self.find, (e1, e2))
        if p1 != p2:
            self.num_sets -= 1
            self.parents[p1] = p2
