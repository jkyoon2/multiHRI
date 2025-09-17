"""
GameCoordinator - 전체 게임 시스템 조율

모든 컴포넌트(OvercookedCoreEnv, ObservationEncoder, RewardCalculator)를 
조율하여 전체 게임 흐름을 관리합니다.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from .observation_encoder import ObservationEncoder, ObservationEncoderFactory
from .reward_calculator import RewardCalculator, RewardCalculatorFactory


class GameCoordinator:
    """
    게임 시스템 전체 조율자
    
    책임:
    - OvercookedCoreEnv에서 원시 상태/보상/정보 받아오기
    - ObservationEncoder로 관찰 생성
    - RewardCalculator로 보상 계산
    - 에이전트 행동 수집 및 조율
    - 전체 게임 흐름 관리
    """
    
    def __init__(self, 
                 core_env,  # OvercookedCoreEnv 인스턴스
                 observation_encoder: ObservationEncoder = None,
                 reward_calculator: RewardCalculator = None,
                 agent_manager = None):  # 선택적 에이전트 관리자
        """
        Args:
            core_env: OvercookedCoreEnv 인스턴스
            observation_encoder: 관찰 인코더 (기본값: lossless encoder)
            reward_calculator: 보상 계산기 (기본값: sparse calculator)
            agent_manager: 에이전트 관리자 (선택사항)
        """
        self.core_env = core_env
        
        # 기본 인코더 설정
        if observation_encoder is None:
            observation_encoder = ObservationEncoderFactory.create_encoder(
                'overcooked', encoding_scheme='OAI_lossless'
            )
        self.observation_encoder = observation_encoder
        
        # 기본 보상 계산기 설정
        if reward_calculator is None:
            reward_calculator = RewardCalculatorFactory.create_calculator('sparse')
        self.reward_calculator = reward_calculator
        
        self.agent_manager = agent_manager
        
        # 상태 관리
        self.current_raw_state = None
        self.episode_done = False
        self.step_count = 0
        
        # 캐시된 관찰 (성능 최적화)
        self._cached_observations = {}
        self._cache_valid = False
    
    def reset(self, start_player_positions: Optional[Dict[int, Tuple[int, int]]] = None) -> Dict[str, Any]:
        """
        게임 리셋
        
        Args:
            start_player_positions: 플레이어별 시작 위치 {player_idx: (x, y)}
            
        Returns:
            리셋 결과 정보
        """
        # 코어 환경 리셋
        self.current_raw_state = self.core_env.reset(start_player_positions)
        self.episode_done = False
        self.step_count = 0
        self._cache_valid = False
        
        # 에이전트 관리자 리셋 (있는 경우)
        if self.agent_manager:
            self.agent_manager.reset()
        
        # 초기 관찰 생성
        observations = self._generate_observations()
        
        return {
            'observations': observations,
            'raw_state': self.current_raw_state,
            'done': self.episode_done,
            'step_count': self.step_count
        }
    
    def step(self, actions: Union[Dict[int, int], List[int], int], 
             ego_player_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        게임 스텝 실행
        
        Args:
            actions: 행동 정보
                - Dict[int, int]: {player_idx: action} (다중 에이전트)
                - List[int]: [action0, action1, ...] (순서대로)
                - int: ego agent 행동 (단일 에이전트, ego_player_idx 필요)
            ego_player_idx: 단일 에이전트 모드에서 ego agent 인덱스
            
        Returns:
            스텝 결과 정보
        """
        if self.episode_done:
            raise ValueError("Episode is done. Call reset() first.")
        
        # 행동을 joint_action 리스트로 변환
        joint_action = self._process_actions(actions, ego_player_idx)
        
        # 코어 환경에서 스텝 실행
        self.current_raw_state, raw_reward, self.episode_done, raw_info = self.core_env.step(joint_action)
        self.step_count += 1
        self._cache_valid = False
        
        # 관찰 생성
        observations = self._generate_observations()
        
        # 보상 계산 (플레이어별)
        rewards = self._calculate_rewards(raw_reward, raw_info)
        
        # 결과 구성
        result = {
            'observations': observations,
            'rewards': rewards,
            'raw_reward': raw_reward,
            'done': self.episode_done,
            'info': raw_info,
            'step_count': self.step_count,
            'raw_state': self.current_raw_state
        }
        
        # 단일 에이전트 모드인 경우 ego agent 중심으로 결과 재구성
        if isinstance(actions, int) and ego_player_idx is not None:
            result.update({
                'observation': observations.get(ego_player_idx, {}),
                'reward': rewards.get(ego_player_idx, 0.0)
            })
        
        return result
    
    def _process_actions(self, actions: Union[Dict[int, int], List[int], int], 
                        ego_player_idx: Optional[int]) -> List[int]:
        """행동을 joint_action 리스트로 변환"""
        num_players = self.core_env.get_num_players()
        
        if isinstance(actions, dict):
            # 딕셔너리 형태: {player_idx: action}
            joint_action = []
            for i in range(num_players):
                if i in actions:
                    joint_action.append(actions[i])
                else:
                    # 누락된 플레이어는 기본 행동 (interact)
                    joint_action.append(5)
        
        elif isinstance(actions, list):
            # 리스트 형태: [action0, action1, ...]
            joint_action = actions[:num_players]
            # 부족한 경우 기본 행동으로 채움
            while len(joint_action) < num_players:
                joint_action.append(5)
        
        elif isinstance(actions, int):
            # 단일 행동: ego agent만 지정
            if ego_player_idx is None:
                raise ValueError("ego_player_idx must be provided when actions is int")
            
            joint_action = [5] * num_players  # 기본 행동으로 초기화
            joint_action[ego_player_idx] = actions
            
            # 에이전트 관리자가 있으면 다른 플레이어들의 행동 수집
            if self.agent_manager:
                observations = self._generate_observations()
                for player_idx in range(num_players):
                    if player_idx != ego_player_idx:
                        agent_action = self.agent_manager.get_action(
                            player_idx, observations[player_idx]
                        )
                        if agent_action is not None:
                            joint_action[player_idx] = agent_action
        
        else:
            raise ValueError(f"Invalid actions type: {type(actions)}")
        
        return joint_action
    
    def _generate_observations(self) -> Dict[int, Dict[str, Any]]:
        """모든 플레이어의 관찰 생성"""
        if self._cache_valid and self._cached_observations:
            return self._cached_observations
        
        observations = {}
        num_players = self.core_env.get_num_players()
        
        for player_idx in range(num_players):
            obs = self.observation_encoder.encode(
                raw_state=self.current_raw_state,
                mdp=self.core_env.mdp,
                player_idx=player_idx
            )
            observations[player_idx] = obs
        
        # 캐시 업데이트
        self._cached_observations = observations
        self._cache_valid = True
        
        return observations
    
    def _calculate_rewards(self, raw_reward: float, raw_info: Dict[str, Any]) -> Dict[int, float]:
        """모든 플레이어의 보상 계산"""
        rewards = {}
        num_players = self.core_env.get_num_players()
        
        for player_idx in range(num_players):
            reward = self.reward_calculator.calculate_reward(
                raw_reward=raw_reward,
                raw_info=raw_info,
                player_idx=player_idx,
                step_count=self.step_count
            )
            rewards[player_idx] = reward
        
        return rewards
    
    def get_observation_for_player(self, player_idx: int) -> Dict[str, Any]:
        """특정 플레이어의 관찰 반환"""
        if not self._cache_valid:
            self._generate_observations()
        return self._cached_observations.get(player_idx, {})
    
    def get_observation_space(self) -> 'spaces.Space':
        """관찰 공간 반환"""
        return self.observation_encoder.get_observation_space()
    
    def get_action_space(self) -> 'spaces.Space':
        """행동 공간 반환"""
        from gymnasium import spaces
        from overcooked_ai_py.mdp.actions import Action
        return spaces.Discrete(len(Action.ALL_ACTIONS))
    
    def render(self, mode: str = 'human'):
        """렌더링"""
        return self.core_env.render(mode)
    
    def close(self):
        """리소스 정리"""
        self.core_env.close()
    
    # 유틸리티 메서드들
    def get_layout_name(self) -> str:
        """레이아웃 이름 반환"""
        return self.core_env.get_layout_name()
    
    def get_num_players(self) -> int:
        """플레이어 수 반환"""
        return self.core_env.get_num_players()
    
    def is_done(self) -> bool:
        """에피소드 종료 여부"""
        return self.episode_done
    
    def get_step_count(self) -> int:
        """현재 스텝 수"""
        return self.step_count
    
    def get_current_state(self):
        """현재 원시 상태 반환"""
        return self.current_raw_state


class SimpleAgentManager:
    """
    간단한 에이전트 관리자
    
    GameCoordinator에서 사용할 수 있는 기본 에이전트 관리 기능을 제공합니다.
    """
    
    def __init__(self, agents: Dict[int, Any] = None):
        """
        Args:
            agents: {player_idx: agent} 에이전트 딕셔너리
        """
        self.agents = agents or {}
    
    def add_agent(self, player_idx: int, agent):
        """에이전트 추가"""
        self.agents[player_idx] = agent
    
    def get_action(self, player_idx: int, observation: Dict[str, Any]) -> Optional[int]:
        """특정 플레이어의 행동 가져오기"""
        if player_idx not in self.agents:
            return None
        
        agent = self.agents[player_idx]
        
        # 에이전트 타입에 따라 다른 방식으로 행동 예측
        try:
            if hasattr(agent, 'predict'):
                # Stable-Baselines3 스타일
                action, _ = agent.predict(observation, deterministic=False)
                return int(action)
            elif hasattr(agent, 'action'):
                # 커스텀 에이전트 스타일
                return agent.action(observation)
            elif callable(agent):
                # 함수형 에이전트
                return agent(observation)
            else:
                # 알 수 없는 타입
                return None
        except Exception as e:
            print(f"Warning: Failed to get action from agent {player_idx}: {e}")
            return None
    
    def reset(self):
        """모든 에이전트 리셋"""
        for agent in self.agents.values():
            if hasattr(agent, 'reset'):
                agent.reset()


# 편의 함수들
def create_game_coordinator(layout_name: str, 
                          encoding_scheme: str = 'OAI_lossless',
                          reward_type: str = 'sparse',
                          horizon: int = 400,
                          **kwargs):
    """
    GameCoordinator 생성 편의 함수
    
    Args:
        layout_name: Overcooked 레이아웃 이름
        encoding_scheme: 관찰 인코딩 방식
        reward_type: 보상 계산 방식
        horizon: 에피소드 길이
        **kwargs: 추가 파라미터
        
    Returns:
        GameCoordinator 인스턴스
    """
    from ..gym_environments.overcooked_core_env import OvercookedCoreEnv
    
    # 코어 환경 생성
    core_env = OvercookedCoreEnv(layout_name, horizon)
    
    # 관찰 인코더 생성
    obs_encoder = ObservationEncoderFactory.create_encoder(
        'overcooked', 
        encoding_scheme=encoding_scheme,
        **kwargs.get('encoder_kwargs', {})
    )
    
    # 보상 계산기 생성
    reward_calc = RewardCalculatorFactory.create_calculator(
        reward_type,
        **kwargs.get('reward_kwargs', {})
    )
    
    # 에이전트 관리자 생성 (선택적)
    agent_manager = None
    if 'agents' in kwargs:
        agent_manager = SimpleAgentManager(kwargs['agents'])
    
    return GameCoordinator(
        core_env=core_env,
        observation_encoder=obs_encoder,
        reward_calculator=reward_calc,
        agent_manager=agent_manager
    )


def create_single_agent_coordinator(layout_name: str,
                                  ego_player_idx: int = 0,
                                  teammates: Dict[int, Any] = None,
                                  **kwargs):
    """
    단일 에이전트 학습용 GameCoordinator 생성
    
    Args:
        layout_name: 레이아웃 이름
        ego_player_idx: 학습할 에이전트의 플레이어 인덱스
        teammates: {player_idx: agent} 팀메이트 에이전트들
        **kwargs: 추가 파라미터
        
    Returns:
        GameCoordinator 인스턴스
    """
    # 에이전트 관리자 설정
    agents = teammates or {}
    kwargs['agents'] = agents
    
    coordinator = create_game_coordinator(layout_name, **kwargs)
    coordinator.ego_player_idx = ego_player_idx
    
    return coordinator
