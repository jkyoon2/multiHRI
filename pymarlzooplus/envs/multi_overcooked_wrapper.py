"""
Multi-Agent Overcooked Environment Wrapper
3명 이상의 에이전트를 지원하는 멀티 에이전트 오버쿡드 환경을 Pymarlzooplus와 호환되도록 래핑
"""

import random
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from pymarlzooplus.envs.multiagentenv import MultiAgentEnv
from .oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv
# Use local oai_agents module bundled in this repo to avoid conflicts with any
# globally installed packages of the same name.
from .oai_agents.common.arguments import get_arguments


class _MultiOvercookedWrapper(MultiAgentEnv):
    """
    멀티 에이전트 오버쿡드 환경을 Pymarlzooplus MultiAgentEnv 인터페이스로 래핑
    
    Features:
    - 3명 이상의 에이전트 지원
    - 다양한 관찰 인코딩 방식 지원
    - 커스텀 보상 계산
    - 역할 기반 에이전트 시스템
    """
    
    def __init__(self, 
                 layout_name: str = "3_chefs_smartfactory",
                 num_agents: int = 3, 
                 encoding_scheme: str = "OAI_lossless",
                 reward_type: str = "sparse",
                 horizon: int = 400,
                 render: bool = False,
                 **kwargs):
        """
        Args:
            layout_name: 오버쿡드 레이아웃 이름
            num_agents: 에이전트 수 (3명 이상)
            encoding_scheme: 관찰 인코딩 방식 ('OAI_lossless', 'OAI_feats' 등)
            reward_type: 보상 계산 방식 ('sparse', 'shaped', 'learner')
            horizon: 에피소드 최대 길이
            render: 렌더링 여부
        """
        super().__init__()
        
        # 환경 설정
        self.layout_name = layout_name
        self.num_agents = max(num_agents, 2)  # 최소 2명
        self.encoding_scheme = encoding_scheme
        self.reward_type = reward_type
        self.horizon = horizon
        self.render_enabled = render
        
        # Pymarlzooplus 인터페이스 요구사항
        self.n_agents = self.num_agents
        self.episode_limit = horizon
        
        # 간단한 코어 환경 초기화 (base_overcooked_env만 사용)
        self._init_core_env(**kwargs)
        
        # 상태 관리
        self._obs = None
        self._state = None
        self._info = {}
        self._done = False
        
    def _init_core_env(self, **kwargs):
        """base_overcooked_env.OvercookedGymEnv만 사용하여 초기화"""
        args = get_arguments()
        args.layout_names = [self.layout_name]
        args.horizon = self.horizon
        args.num_players = self.num_agents
        args.encoding_fn = self.encoding_scheme
        # 기본값 강제 설정 (파서 기본값이 있어도 덮어씀)
        args.device = kwargs.get('device', 'cpu')
        args.num_stack = kwargs.get('num_stack', 1)
        # 보상 관련 플래그를 env_args로부터 받아 일관되게 적용
        args.reward_magnifier = kwargs.get('reward_magnifier', 1.0)
        args.dynamic_reward = kwargs.get('dynamic_reward', getattr(args, 'dynamic_reward', False))
        args.final_sparse_r_ratio = kwargs.get('final_sparse_r_ratio', getattr(args, 'final_sparse_r_ratio', 0.1))
        args.overcooked_verbose = kwargs.get('overcooked_verbose', False)
        args.n_envs = kwargs.get('n_envs', getattr(args, 'n_envs', 1))

        self.env = OvercookedGymEnv(
            learner_type='originaler',
            args=args,
            full_init=True,
            shape_rewards=self.reward_type == 'shaped',
            layout_name=self.layout_name,
            horizon=self.horizon
        )

        # 관찰/행동 정보 설정
        self.n_actions = self.env.action_space.n if hasattr(self.env.action_space, 'n') else len(getattr(self.env.action_space, 'items', []))
        # 관찰 공간 추론 (visual_obs 우선)
        if 'visual_obs' in self.env.observation_space.keys():
            self.obs_shape = self.env.observation_space['visual_obs'].shape
        elif 'agent_obs' in self.env.observation_space.keys():
            self.obs_shape = self.env.observation_space['agent_obs'].shape
        else:
            self.obs_shape = (27, 7, 7)

        print(f"✅ Multi Overcooked (core) initialized: {self.layout_name}, {self.num_agents} agents")
    
    def _init_fallback_environment(self):
        """실패 시 기존 환경으로 fallback"""
        from .oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv
        from .oai_agents.common.arguments import get_arguments
        
        # 기본 arguments 생성
        args = get_arguments()
        args.layout_names = [self.layout_name]
        args.horizon = self.horizon
        args.num_players = self.num_agents
        args.encoding_fn = self.encoding_scheme
        
        # 기존 복잡한 환경 사용
        self.base_env = OvercookedGymEnv(
            learner_type='originaler',
            args=args,
            full_init=True
        )
        
        print(f"⚠️  Using fallback OvercookedGymEnv")
    
    def _setup_spaces(self):
        """관찰 및 행동 공간 설정"""
        # 관찰 공간 정보
        obs_space = self.coordinator.get_observation_space()
        if hasattr(obs_space, 'spaces') and 'visual_obs' in obs_space.spaces:
            self.obs_shape = obs_space.spaces['visual_obs'].shape
        else:
            self.obs_shape = (27, 7, 7)  # 기본값
        
        # 행동 공간 정보
        action_space = self.coordinator.get_action_space()
        self.n_actions = action_space.n if hasattr(action_space, 'n') else 6
    
    def step(self, actions: List[int]) -> Tuple[float, bool, Dict]:
        """
        환경 스텝 실행
        
        Args:
            actions: 모든 에이전트의 행동 리스트
            
        Returns:
            (reward, terminated, info)
        """
        if self._done:
            return 0.0, True, {}
        
        try:
            # base env는 joint action을 받도록 수정됨
            obs_dict, reward, done, info = self.env.step(actions)
            self._obs = self._extract_observations(obs_dict)
            self._state = self._extract_state(self.env.state)
            self._done = done
            self._info = info
            
            total_reward = float(reward)
            
            if self.render_enabled:
                self.render()
                
            return total_reward, self._done, self._info
            
        except Exception as e:
            print(f"❌ Step failed: {e}")
            return 0.0, True, {"error": str(e)}
    
    def _extract_observations(self, obs_dict: Dict[int, Dict]) -> List[np.ndarray]:
        """관찰 딕셔너리를 리스트로 변환"""
        observations = []
        
        for agent_id in range(self.num_agents):
            if agent_id in obs_dict:
                obs = obs_dict[agent_id]
                # visual_obs를 우선적으로 사용
                if 'visual_obs' in obs:
                    observations.append(obs['visual_obs'])
                elif 'agent_obs' in obs:
                    observations.append(obs['agent_obs'])
                else:
                    # 빈 관찰 생성
                    observations.append(np.zeros(self.obs_shape, dtype=np.float32))
            else:
                observations.append(np.zeros(self.obs_shape, dtype=np.float32))
        
        return observations
    
    def _extract_state(self, raw_state) -> np.ndarray:
        """글로벌 상태 추출"""
        try:
            if hasattr(raw_state, 'to_dict'):
                state_dict = raw_state.to_dict()
                # 상태를 flatten하여 벡터로 변환
                state_vector = []
                for key, value in state_dict.items():
                    if isinstance(value, (list, tuple, np.ndarray)):
                        state_vector.extend(np.array(value).flatten())
                    else:
                        state_vector.append(float(value))
                return np.array(state_vector, dtype=np.float32)
            else:
                # 관찰들을 concatenate하여 상태로 사용
                return np.concatenate([obs.flatten() for obs in self._obs])
        except:
            # Fallback: 빈 상태
            return np.zeros(100, dtype=np.float32)
    
    def get_obs(self) -> List[np.ndarray]:
        """모든 에이전트의 관찰 반환"""
        return self._obs if self._obs is not None else [np.zeros(self.obs_shape) for _ in range(self.num_agents)]
    
    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        """특정 에이전트의 관찰 반환"""
        if self._obs and 0 <= agent_id < len(self._obs):
            return self._obs[agent_id]
        return np.zeros(self.obs_shape, dtype=np.float32)
    
    def get_obs_size(self) -> tuple:
        """관찰 크기 반환 (이미지 경로용으로 (C,H,W) 그대로 반환)"""
        return tuple(int(d) for d in self.obs_shape)
    
    def get_state(self) -> np.ndarray:
        """글로벌 상태 반환"""
        return self._state if self._state is not None else np.zeros(100, dtype=np.float32)
    
    def get_state_size(self) -> int:
        """글로벌 상태 크기 반환"""
        return len(self.get_state())
    
    def get_avail_actions(self) -> List[List[int]]:
        """모든 에이전트의 가능한 행동 반환"""
        return [[1] * self.n_actions for _ in range(self.num_agents)]
    
    def get_avail_agent_actions(self, agent_id: int) -> List[int]:
        """특정 에이전트의 가능한 행동 반환"""
        return [1] * self.n_actions
    
    def get_total_actions(self) -> int:
        """총 행동 수 반환"""
        return self.n_actions
    
    def reset(self, seed: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """환경 리셋"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        try:
            # base env 리셋은 모든 플레이어 관찰 dict 반환
            obs_dict = self.env.reset()
            self._obs = self._extract_observations(obs_dict)
            self._state = self._extract_state(self.env.state)
            self._done = False
            self._info = {}
            
            return self.get_obs(), self.get_state()
            
        except Exception as e:
            print(f"❌ Reset failed: {e}")
            # 기본 관찰 반환
            self._obs = [np.zeros(self.obs_shape) for _ in range(self.num_agents)]
            self._state = np.zeros(100, dtype=np.float32)
            return self.get_obs(), self.get_state()
    
    def render(self):
        """환경 렌더링"""
        try:
            if hasattr(self, 'env') and hasattr(self.env, 'render'):
                # base overcooked env renders via pygame
                # ensure visualization initialized
                if not getattr(self.env, 'visualization_enabled', False) and hasattr(self.env, 'setup_visualization'):
                    self.env.setup_visualization()
                self.env.render()
        except Exception as e:
            print(f"⚠️  Render failed: {e}")

    def get_print_info(self):
        """runner가 초기화 시 표준 출력용으로 호출. 메시지 없으면 None 반환"""
        # 이 래퍼는 별도 초기 경고 메시지를 유지하지 않으므로 항상 None
        return None
    
    def close(self):
        """리소스 정리"""
        if hasattr(self, 'coordinator'):
            self.coordinator.close()
    
    def seed(self, seed: int = None):
        """시드 설정"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return seed
    
    def save_replay(self):
        """리플레이 저장 (구현 필요시)"""
        pass
    
    def get_stats(self) -> Dict:
        """환경 통계 반환"""
        return {
            "layout_name": self.layout_name,
            "num_agents": self.num_agents,
            "encoding_scheme": self.encoding_scheme,
            "reward_type": self.reward_type
        }


# 편의 함수
def create_multi_overcooked_env(layout_name: str = "cramped_room",
                               num_agents: int = 3,
                               **kwargs) -> _MultiOvercookedWrapper:
    """멀티 에이전트 오버쿡드 환경 생성 편의 함수"""
    return _MultiOvercookedWrapper(
        layout_name=layout_name,
        num_agents=num_agents,
        **kwargs
    )
