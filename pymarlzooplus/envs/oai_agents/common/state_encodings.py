from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, Direction, PlayerState
from overcooked_ai_py.planning.planners import MediumLevelActionManager
from typing import Dict, Tuple
import numpy as np

# MLAM Action Encoder 클래스 추가
class MLAMActionEncoder:
    """MLAM을 활용한 Medium Level Actions 인코딩 시스템"""
    
    def __init__(self, mdp: OvercookedGridworld, mlam: MediumLevelActionManager):
        self.mdp = mdp
        self.mlam = mlam
        
        # Medium Level Actions 정의
        self.mla_types = {
            'pickup_onion': 0,
            'pickup_tomato': 1,
            'pickup_dish': 2,
            'pickup_soup': 3,
            'put_onion_in_pot': 4,
            'put_tomato_in_pot': 5,
            'place_on_counter': 6,
            'deliver_soup': 7,
            'start_cooking': 8,
            'wait': 9
        }
        
        self.num_mla_types = len(self.mla_types)
    
    def get_available_mla_for_agent(self, state: OvercookedState, agent_idx: int) -> Dict[str, bool]:
        """특정 에이전트가 수행 가능한 Medium Level Actions 반환"""
        agent = state.players[agent_idx]
        available_mla = {mla_name: False for mla_name in self.mla_types.keys()}
        
        # 각 액션 타입별로 가능성 확인
        if not agent.has_object():
            # 오브젝트를 가지고 있지 않은 경우
            counter_objects = self.mdp.get_counter_objects_dict(state, self.mlam.counter_pickup)
            
            # 양파 픽업 가능성
            onion_actions = self.mlam.pickup_onion_actions(counter_objects)
            available_mla['pickup_onion'] = len(onion_actions) > 0
            
            # 토마토 픽업 가능성
            tomato_actions = self.mlam.pickup_tomato_actions(counter_objects)
            available_mla['pickup_tomato'] = len(tomato_actions) > 0
            
            # 접시 픽업 가능성
            dish_actions = self.mlam.pickup_dish_actions(counter_objects)
            available_mla['pickup_dish'] = len(dish_actions) > 0
            
            # 카운터에서 수프 픽업 가능성
            soup_actions = self.mlam.pickup_counter_soup_actions(counter_objects)
            available_mla['pickup_soup'] = len(soup_actions) > 0
            
            # 조리 시작 가능성
            pot_states = self.mdp.get_pot_states(state)
            cooking_actions = self.mlam.start_cooking_actions(pot_states)
            available_mla['start_cooking'] = len(cooking_actions) > 0
            
        else:
            # 오브젝트를 가지고 있는 경우
            obj = agent.get_object()
            pot_states = self.mdp.get_pot_states(state)
            
            # 카운터에 놓기 가능성
            counter_actions = self.mlam.place_obj_on_counter_actions(state)
            available_mla['place_on_counter'] = len(counter_actions) > 0
            
            if obj.name == 'onion':
                # 양파를 팟에 넣기 가능성
                onion_pot_actions = self.mlam.put_onion_in_pot_actions(pot_states)
                available_mla['put_onion_in_pot'] = len(onion_pot_actions) > 0
                
            elif obj.name == 'tomato':
                # 토마토를 팟에 넣기 가능성
                tomato_pot_actions = self.mlam.put_tomato_in_pot_actions(pot_states)
                available_mla['put_tomato_in_pot'] = len(tomato_pot_actions) > 0
                
            elif obj.name == 'dish':
                # 접시로 수프 픽업 가능성
                soup_dish_actions = self.mlam.pickup_soup_with_dish_actions(pot_states)
                available_mla['pickup_soup'] = len(soup_dish_actions) > 0
                
            elif obj.name == 'soup':
                # 수프 배달 가능성
                deliver_actions = self.mlam.deliver_soup_actions()
                available_mla['deliver_soup'] = len(deliver_actions) > 0
        
        # 대기 액션 가능성
        if self.mlam.wait_allowed:
            available_mla['wait'] = True
        
        return available_mla
    
    def encode_mla_features(self, state: OvercookedState, p_idx=None) -> np.ndarray:
        """모든 에이전트의 Medium Level Actions 가능성을 인코딩"""
        num_agents = len(state.players)
        mla_features = np.zeros((num_agents, self.num_mla_types), dtype=np.float32)
        
        for agent_idx in range(num_agents):
            available_mla = self.get_available_mla_for_agent(state, agent_idx)
            for mla_name, is_available in available_mla.items():
                mla_idx = self.mla_types[mla_name]
                mla_features[agent_idx, mla_idx] = 1.0 if is_available else 0.0
        
        if p_idx is not None:
            return mla_features[p_idx]
        return mla_features
    
    def get_collaboration_opportunities(self, state: OvercookedState) -> Dict[str, float]:
        """협력 기회 탐지 및 점수화"""
        collaboration_scores = {
            'onion_supply_chain': 0.0,  # 양파 공급 체인
            'dish_supply_chain': 0.0,   # 접시 공급 체인
            'soup_delivery_chain': 0.0, # 수프 배달 체인
            'pot_management': 0.0,      # 팟 관리
            'counter_coordination': 0.0 # 카운터 조율
        }
        
        # 각 에이전트의 가능한 액션들
        agent_mla = []
        for agent_idx in range(len(state.players)):
            agent_mla.append(self.get_available_mla_for_agent(state, agent_idx))
        
        # 양파 공급 체인 점수
        onion_suppliers = sum(1 for mla in agent_mla if mla['pickup_onion'])
        onion_consumers = sum(1 for mla in agent_mla if mla['put_onion_in_pot'])
        collaboration_scores['onion_supply_chain'] = min(onion_suppliers, onion_consumers) * 0.5
        
        # 접시 공급 체인 점수
        dish_suppliers = sum(1 for mla in agent_mla if mla['pickup_dish'])
        dish_consumers = sum(1 for mla in agent_mla if mla['pickup_soup'])
        collaboration_scores['dish_supply_chain'] = min(dish_suppliers, dish_consumers) * 0.5
        
        # 수프 배달 체인 점수
        soup_pickers = sum(1 for mla in agent_mla if mla['pickup_soup'])
        soup_deliverers = sum(1 for mla in agent_mla if mla['deliver_soup'])
        collaboration_scores['soup_delivery_chain'] = min(soup_pickers, soup_deliverers) * 0.5
        
        # 팟 관리 점수
        pot_managers = sum(1 for mla in agent_mla if mla['start_cooking'])
        collaboration_scores['pot_management'] = pot_managers * 0.3
        
        # 카운터 조율 점수
        counter_users = sum(1 for mla in agent_mla if mla['place_on_counter'])
        collaboration_scores['counter_coordination'] = counter_users * 0.2
        
        return collaboration_scores

def OAI_feats_closure():
    mlams = {}
    def OAI_get_feats(mdp: OvercookedGridworld, state: OvercookedState, grid_shape: tuple, horizon: int,
                      num_pots: int = 2, p_idx=None, goal_objects=None):
        """
        Uses Overcooked-ai's BC 96 dim BC featurization. Only returns agent_obs
        """
        nonlocal mlams
        if mdp.layout_name not in mlams:
            all_counters = mdp.get_counter_locations()
            COUNTERS_PARAMS = {
                'start_orientations': False,
                'wait_allowed': True,  # ZSC-Eval 호환성: 대기 액션 허용으로 플래닝 실패 방지
                'counter_goals': all_counters,
                'counter_drop': all_counters,
                'counter_pickup': all_counters,
                'same_motion_goals': True
            }
            mlams[mdp.layout_name] = MediumLevelActionManager.from_pickle_or_compute(mdp, COUNTERS_PARAMS, force_compute=False)
        mlam = mlams[mdp.layout_name]
        agent_obs = mdp.featurize_state(state, mlam, num_pots=num_pots, p_idx=p_idx)
        if p_idx is None:
        #     agent_obs = agent_obs[p_idx]
        # else:
            agent_obs = np.stack(agent_obs, axis=0)
        return {'agent_obs': agent_obs}
    return OAI_get_feats

OAI_feats = OAI_feats_closure()

# MLAM이 강화된 OAI_feats 함수
def OAI_feats_enhanced(mdp: OvercookedGridworld, state: OvercookedState, 
                      grid_shape: tuple, horizon: int, mlam: MediumLevelActionManager,
                      num_pots: int = 2, p_idx=None, goal_objects=None) -> Dict:
    """OAI_feats 인코딩에 MLAM 정보 추가"""
    # 기존 OAI_feats 인코딩
    original_obs = OAI_feats(mdp, state, grid_shape, horizon, num_pots, p_idx, goal_objects)
    
    # 전달받은 MLAM을 사용하여 인코더 생성
    mlam_encoder = MLAMActionEncoder(mdp, mlam)
    
    # MLAM 특징 추가
    mla_features = mlam_encoder.encode_mla_features(state, p_idx)
    collaboration_features = mlam_encoder.get_collaboration_opportunities(state)
    
    # 협력 특징을 벡터로 변환
    collaboration_vector = np.array([
        collaboration_features['onion_supply_chain'],
        collaboration_features['dish_supply_chain'],
        collaboration_features['soup_delivery_chain'],
        collaboration_features['pot_management'],
        collaboration_features['counter_coordination']
    ], dtype=np.float32)
    
    # 기존 특징에 MLAM 특징 추가
    if p_idx is not None:
        # 단일 에이전트
        enhanced_agent_obs = np.concatenate([
            original_obs['agent_obs'],
            mla_features,
            collaboration_vector
        ])
        original_obs['agent_obs'] = enhanced_agent_obs
    else:
        # 모든 에이전트
        num_agents = len(state.players)
        enhanced_agent_obs = []
        for agent_idx in range(num_agents):
            agent_mla = mlam_encoder.encode_mla_features(state, agent_idx)
            enhanced_obs = np.concatenate([
                original_obs['agent_obs'][agent_idx],
                agent_mla,
                collaboration_vector
            ])
            enhanced_agent_obs.append(enhanced_obs)
        original_obs['agent_obs'] = np.stack(enhanced_agent_obs, axis=0)
    
    return original_obs

def OAI_encode_state(mdp: OvercookedGridworld, state: OvercookedState, grid_shape: tuple, horizon: int, p_idx=None,
                     goal_objects=None):
    """
    Uses Overcooked-ai's RL lossless encoding by stacking env.num_enc_channels binary masks (env.num_enc_channelsxNxM). Only returns visual_obs.
    """
    visual_obs = mdp.lossless_state_encoding(state, horizon=horizon, goal_objects=goal_objects, p_idx=p_idx)
    
    # visual_obs의 형태에 따라 다르게 처리
    if isinstance(visual_obs, list):
        visual_obs = np.stack(visual_obs, axis=0)
    
    # 차원 수에 따라 다르게 처리
    if len(visual_obs.shape) == 4:  # (agents, height, width, features)
        # Reorder to channels first
        visual_obs = np.transpose(visual_obs, (0, 3, 1, 2))  # (agents, features, height, width)
        grid_shape = (2, visual_obs.shape[1], *grid_shape)
        assert len(visual_obs.shape) == len(grid_shape)
        assert all([visual_obs.shape[i] <= grid_shape[i] for i in range(len(visual_obs.shape))])
        padding_amount = [(0, grid_shape[i] - visual_obs.shape[i]) for i in range(len(grid_shape))]
        visual_obs = np.pad(visual_obs, padding_amount)
    elif len(visual_obs.shape) == 3:  # (height, width, features)
        # Reorder to channels first
        visual_obs = np.transpose(visual_obs, (2, 0, 1))  # (features, height, width)
        grid_shape = (visual_obs.shape[0], *grid_shape)
        assert len(visual_obs.shape) == len(grid_shape)
        assert all([visual_obs.shape[i] <= grid_shape[i] for i in range(len(visual_obs.shape))])
        padding_amount = [(0, grid_shape[i] - visual_obs.shape[i]) for i in range(len(grid_shape))]
        visual_obs = np.pad(visual_obs, padding_amount)
    else:
        raise ValueError(f"Unexpected visual_obs shape: {visual_obs.shape}")
    
    # if p_idx is not None:
    #     visual_obs = visual_obs[p_idx]
    return {'visual_obs': visual_obs}

# MLAM이 강화된 OAI_lossless 함수
def OAI_lossless_enhanced(mdp: OvercookedGridworld, state: OvercookedState, 
                         grid_shape: tuple, horizon: int, mlam: MediumLevelActionManager,
                         p_idx=None, goal_objects=None) -> Dict:
    """
    OAI_lossless 인코딩에 MLAM 정보를 별도로 추가합니다.
    시각 정보와 비시각 정보를 분리하여 반환하여 MultiInputPolicy와 함께 사용할 수 있습니다.
    """
    # 기존 OAI_lossless 인코딩
    original_obs = OAI_encode_state(mdp, state, grid_shape, horizon, p_idx, goal_objects)
    
    # 전달받은 MLAM을 사용하여 인코더 생성
    mlam_encoder = MLAMActionEncoder(mdp, mlam)
    
    # MLAM 특징 추가
    mla_features = mlam_encoder.encode_mla_features(state, p_idx)
    collaboration_features = mlam_encoder.get_collaboration_opportunities(state)
    
    # 협력 특징을 벡터로 변환
    collaboration_vector = np.array([
        collaboration_features['onion_supply_chain'],
        collaboration_features['dish_supply_chain'],
        collaboration_features['soup_delivery_chain'],
        collaboration_features['pot_management'],
        collaboration_features['counter_coordination']
    ], dtype=np.float32)
    
    # MLAM 특징을 별도 키로 추가 (시각 정보와 분리)
    if p_idx is not None:
        # 단일 에이전트
        mla_collab_features = np.concatenate([mla_features, collaboration_vector])
        original_obs['mlam_features'] = mla_collab_features
    else:
        # 모든 에이전트
        all_mlam_features = []
        for agent_idx in range(len(state.players)):
            agent_mla = mlam_encoder.encode_mla_features(state, agent_idx)
            mla_collab_features = np.concatenate([agent_mla, collaboration_vector])
            all_mlam_features.append(mla_collab_features)
        original_obs['mlam_features'] = np.stack(all_mlam_features, axis=0)
    
    return original_obs


def OAI_egocentric_encode_state(mdp: OvercookedGridworld, state: OvercookedState,
                                grid_shape: tuple, horizon: int, p_idx=None, goal_objects=None) -> Dict[str, np.array]:
    """
    Returns the egocentric encode state. Player will always be facing down (aka. SOUTH).
    grid_shape: The desired padded output shape from the egocentric view
    Now supports any grid shape (both odd and even dimensions)
    """
    if len(grid_shape) > 2:
        raise ValueError(f'Ego grid shape must be 2D! {grid_shape} is invalid.')
    
    # 모든 grid_shape를 그대로 사용 (홀수/짝수 구분 없음)
    # get_egocentric_grid 함수에서 자동으로 처리

    # Get np.array representing current state
    # This returns 2xNxMxF (F is # features) if p_idx is None else NxMxF
    visual_obs = mdp.lossless_state_encoding(state, horizon=horizon, goal_objects=goal_objects, p_idx=p_idx)

    if p_idx is None:
        visual_obs = np.stack(visual_obs, axis=0)
        visual_obs = np.transpose(visual_obs, (0, 3, 1, 2))  # Reorder to features first --> 2xFxNxM
        num_players, num_features = visual_obs.shape[0], visual_obs.shape[1]
    else:
        visual_obs = np.transpose(visual_obs, (2, 0, 1))
        num_features = visual_obs.shape[0]
    # Remove orientation features since they are now irrelevant.
    # There are num_players * num_directions features.
    #num_layers_to_skip = num_players*len(Direction.ALL_DIRECTIONS)
    #idx_slice = list(range(num_players)) + list(range(num_players+num_layers_to_skip, num_features))
    #visual_obs = visual_obs[:, idx_slice, :, :]
    #assert visual_obs.shape[1] == num_features - num_layers_to_skip
    #num_features = num_features - num_layers_to_skip

    # Now we mask out the egocentric view
    if p_idx is not None:
        ego_visual_obs = get_egocentric_grid(visual_obs, grid_shape, state.players[p_idx])
        assert ego_visual_obs.shape == (num_features, *grid_shape)
    else:
        assert len(state.players) == num_players
        ego_visual_obs = np.stack([get_egocentric_grid(visual_obs[idx], grid_shape, player)
                                   for idx, player in enumerate(state.players)])
        assert ego_visual_obs.shape == (num_players, num_features, *grid_shape)
    return {'visual_obs': ego_visual_obs}

# MLAM이 강화된 OAI_egocentric 함수
def OAI_egocentric_enhanced(mdp: OvercookedGridworld, state: OvercookedState, 
                           grid_shape: tuple, horizon: int, mlam: MediumLevelActionManager,
                           p_idx=None, goal_objects=None) -> Dict:
    """
    OAI_egocentric 인코딩에 MLAM 정보를 별도로 추가합니다.
    시각 정보와 비시각 정보를 분리하여 반환하여 MultiInputPolicy와 함께 사용할 수 있습니다.
    """
    # 기존 OAI_egocentric 인코딩
    original_obs = OAI_egocentric_encode_state(mdp, state, grid_shape, horizon, p_idx, goal_objects)
    
    # 전달받은 MLAM을 사용하여 인코더 생성
    mlam_encoder = MLAMActionEncoder(mdp, mlam)
    
    # MLAM 특징 추가
    mla_features = mlam_encoder.encode_mla_features(state, p_idx)
    collaboration_features = mlam_encoder.get_collaboration_opportunities(state)
    
    # 협력 특징을 벡터로 변환
    collaboration_vector = np.array([
        collaboration_features['onion_supply_chain'],
        collaboration_features['dish_supply_chain'],
        collaboration_features['soup_delivery_chain'],
        collaboration_features['pot_management'],
        collaboration_features['counter_coordination']
    ], dtype=np.float32)
    
    # MLAM 특징을 별도 키로 추가 (시각 정보와 분리)
    if p_idx is not None:
        # 단일 에이전트
        mla_collab_features = np.concatenate([mla_features, collaboration_vector])
        original_obs['mlam_features'] = mla_collab_features
    else:
        # 모든 에이전트
        all_mlam_features = []
        for agent_idx in range(len(state.players)):
            agent_mla = mlam_encoder.encode_mla_features(state, agent_idx)
            mla_collab_features = np.concatenate([agent_mla, collaboration_vector])
            all_mlam_features.append(mla_collab_features)
        original_obs['mlam_features'] = np.stack(all_mlam_features, axis=0)
    
    return original_obs


def get_egocentric_grid(grid: np.array, ego_grid_shape: Tuple[int, int], player: PlayerState) -> np.array:
    """
    플레이어를 중심으로 한 egocentric 그리드를 추출합니다.
    회전 시에도 항상 ego_grid_shape를 유지하도록 수정되었습니다.
    """
    assert len(grid.shape) == 3, f"Expected grid shape of (Features, X, Y), but got {grid.shape}"

    # 1. 플레이어 방향에 따라 회전 전 잘라낼 모양(pre-rotation shape) 결정
    pre_rotation_shape = ego_grid_shape
    if player.orientation in [Direction.EAST, Direction.WEST]:
        # 90도 회전하면 높이와 너비가 바뀌므로, 미리 뒤집어서 모양을 설정
        pre_rotation_shape = (ego_grid_shape[1], ego_grid_shape[0])

    # 2. pre_rotation_shape에 맞춰 필요한 패딩 크기 계산
    pad_x_before = pre_rotation_shape[0] // 2
    pad_x_after = pre_rotation_shape[0] - 1 - pad_x_before
    pad_y_before = pre_rotation_shape[1] // 2
    pad_y_after = pre_rotation_shape[1] - 1 - pad_y_before
    
    # 3. 계산된 크기만큼만 정확히 패딩
    padding_amount = ((0, 0), (pad_x_before, pad_x_after), (pad_y_before, pad_y_after))
    padded_grid = np.pad(grid, padding_amount, mode='constant')
    
    # 4. 패딩된 그리드에서 플레이어의 새 좌표 계산
    player_pos_in_padded = (player.position[0] + pad_x_before, player.position[1] + pad_y_before)
    x, y = player_pos_in_padded
    
    # 5. 플레이어의 새 좌표를 중심으로 그리드 잘라내기
    start_x = x - pad_x_before
    end_x = x + pad_x_after + 1
    start_y = y - pad_y_before
    end_y = y + pad_y_after + 1
    
    player_obs = padded_grid[:, start_x:end_x, start_y:end_y]
    
    assert player_obs.shape == (grid.shape[0], *pre_rotation_shape)

    # 6. 플레이어 방향에 맞춰 그리드 회전
    if player.orientation == Direction.SOUTH:
        final_obs = player_obs
    elif player.orientation == Direction.NORTH:
        final_obs = np.rot90(player_obs, k=2, axes=(1, 2))
    elif player.orientation == Direction.EAST:
        final_obs = np.rot90(player_obs, k=-1, axes=(1, 2))
    elif player.orientation == Direction.WEST:
        final_obs = np.rot90(player_obs, k=1, axes=(1, 2))
    else:
        raise ValueError('Invalid player direction found!')
    
    # 최종 모양이 우리가 원하는 ego_grid_shape와 일치하는지 마지막으로 확인
    assert final_obs.shape == (grid.shape[0], *ego_grid_shape)
    
    return final_obs


ENCODING_SCHEMES = {
    'OAI_feats': OAI_feats,
    'OAI_lossless': OAI_encode_state,
    'OAI_egocentric': OAI_egocentric_encode_state,
    # MLAM이 강화된 인코딩 스키마들
    'OAI_feats_enhanced': OAI_feats_enhanced,
    'OAI_lossless_enhanced': OAI_lossless_enhanced,
    'OAI_egocentric_enhanced': OAI_egocentric_enhanced,
}

if __name__ == '__main__':
    # 테스트 코드는 제거하여 코드를 깔끔하게 유지
    pass

