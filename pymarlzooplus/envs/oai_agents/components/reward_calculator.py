"""
RewardCalculator - 원시 보상을 학습용 보상으로 변환

OvercookedCoreEnv가 반환한 raw_reward와 raw_info를 바탕으로
학습에 도움이 되는 Shaped Reward를 계산합니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np


class RewardCalculator(ABC):
    """
    보상 계산기 추상 클래스
    
    원시 보상과 환경 정보를 바탕으로 학습용 보상을 계산합니다.
    """
    
    @abstractmethod
    def calculate_reward(self, raw_reward: float, raw_info: Dict[str, Any], 
                        player_idx: int, **kwargs) -> float:
        """
        보상 계산
        
        Args:
            raw_reward: OvercookedCoreEnv에서 반환한 원시 보상 (sparse reward)
            raw_info: OvercookedCoreEnv에서 반환한 정보 딕셔너리
            player_idx: 보상을 계산할 플레이어 인덱스
            **kwargs: 추가 파라미터
            
        Returns:
            계산된 보상값
        """
        pass


class SparseRewardCalculator(RewardCalculator):
    """
    Sparse 보상만 사용하는 계산기
    
    환경에서 제공하는 원시 보상을 그대로 사용합니다.
    """
    
    def __init__(self, magnifier: float = 1.0):
        """
        Args:
            magnifier: 보상 배율
        """
        self.magnifier = magnifier
    
    def calculate_reward(self, raw_reward: float, raw_info: Dict[str, Any], 
                        player_idx: int, **kwargs) -> float:
        """Sparse 보상만 반환"""
        return self.magnifier * raw_reward


class ShapedRewardCalculator(RewardCalculator):
    """
    Shaped 보상을 사용하는 계산기
    
    환경 정보에서 shaped_reward_by_agent를 활용합니다.
    """
    
    def __init__(self, magnifier: float = 1.0, sparse_ratio: float = 0.1):
        """
        Args:
            magnifier: 보상 배율
            sparse_ratio: sparse reward의 비율 (0~1)
        """
        self.magnifier = magnifier
        self.sparse_ratio = sparse_ratio
    
    def calculate_reward(self, raw_reward: float, raw_info: Dict[str, Any], 
                        player_idx: int, **kwargs) -> float:
        """Sparse + Shaped 보상 조합"""
        # Sparse reward
        sparse_reward = raw_reward
        
        # Shaped reward (환경에서 제공)
        shaped_rewards = raw_info.get('shaped_r_by_agent', [0.0] * 2)
        if player_idx < len(shaped_rewards):
            shaped_reward = shaped_rewards[player_idx]
        else:
            shaped_reward = 0.0
        
        # 조합
        total_reward = (self.sparse_ratio * sparse_reward + 
                       (1 - self.sparse_ratio) * shaped_reward)
        
        return self.magnifier * total_reward


class LearnerBasedRewardCalculator(RewardCalculator):
    """
    기존 Learner 클래스를 활용한 보상 계산기
    
    다양한 협력 성향을 가진 learner 타입들을 지원합니다.
    """
    
    def __init__(self, learner_type: str = 'originaler', magnifier: float = 1.0):
        """
        Args:
            learner_type: learner 타입
                - 'originaler': 기본 개인 보상
                - 'collaborator': 개인 + 그룹 보상 균형
                - 'supporter': 그룹 보상 중심
                - 'soloworker': 개인 보상만
                - 'selfisher': 개인 보상 - 그룹 보상
                - 'saboteur': 그룹 방해
            magnifier: 보상 배율
        """
        self.learner_type = learner_type
        self.magnifier = magnifier
        
        # Learner 가중치 설정
        self.weights = self._get_learner_weights(learner_type)
    
    def _get_learner_weights(self, learner_type: str) -> Dict[str, float]:
        """Learner 타입별 가중치 반환"""
        weights_map = {
            'originaler': {'personal': 1.0, 'group': 0.0},
            'collaborator': {'personal': 0.5, 'group': 0.5},
            'supporter': {'personal': 1/3, 'group': 2/3},
            'soloworker': {'personal': 1.0, 'group': 0.0},
            'selfisher': {'personal': 1.0, 'group': -1.0},
            'saboteur': {'personal': 2/3, 'group': -1/3}
        }
        
        if learner_type not in weights_map:
            raise ValueError(f"Unknown learner type: {learner_type}")
        
        return weights_map[learner_type]
    
    def calculate_reward(self, raw_reward: float, raw_info: Dict[str, Any], 
                        player_idx: int, ratio: float = 0.1, 
                        num_players: int = 2, **kwargs) -> float:
        """Learner 기반 보상 계산"""
        # 환경 정보에서 보상 추출
        sparse_rewards = raw_info.get('sparse_r_by_agent', [0.0] * num_players)
        shaped_rewards = raw_info.get('shaped_r_by_agent', [0.0] * num_players)
        
        # 그룹 보상 계산
        group_sparse_r = sum(sparse_rewards)
        group_shaped_r = sum(shaped_rewards)
        
        # 개인 보상 계산
        if player_idx < len(sparse_rewards):
            personal_sparse_r = sparse_rewards[player_idx]
            personal_shaped_r = shaped_rewards[player_idx]
        else:
            personal_sparse_r = raw_reward  # fallback
            personal_shaped_r = 0.0
        
        # 비율에 따른 보상 조합
        personal_reward = group_sparse_r * ratio + personal_shaped_r * (1 - ratio)
        group_reward = (1/num_players) * (num_players * group_sparse_r * ratio + 
                                        group_shaped_r * (1 - ratio))
        
        # Learner 가중치 적용
        final_reward = (self.weights['personal'] * personal_reward + 
                       self.weights['group'] * group_reward)
        
        return self.magnifier * final_reward


class CustomRewardCalculator(RewardCalculator):
    """
    커스텀 보상 함수를 사용하는 계산기
    
    사용자 정의 보상 함수를 외부에서 주입받아 사용합니다.
    """
    
    def __init__(self, reward_fn, magnifier: float = 1.0):
        """
        Args:
            reward_fn: 보상 계산 함수 (raw_reward, raw_info, player_idx) -> float
            magnifier: 보상 배율
        """
        self.reward_fn = reward_fn
        self.magnifier = magnifier
    
    def calculate_reward(self, raw_reward: float, raw_info: Dict[str, Any], 
                        player_idx: int, **kwargs) -> float:
        """커스텀 함수로 보상 계산"""
        reward = self.reward_fn(raw_reward, raw_info, player_idx, **kwargs)
        return self.magnifier * reward


class MultiAgentRewardCalculator:
    """
    다중 에이전트용 보상 계산기
    
    여러 플레이어의 보상을 동시에 계산합니다.
    """
    
    def __init__(self, calculators: Dict[int, RewardCalculator]):
        """
        Args:
            calculators: {player_idx: RewardCalculator} 플레이어별 보상 계산기
        """
        self.calculators = calculators
    
    def calculate_rewards(self, raw_reward: float, raw_info: Dict[str, Any], 
                         **kwargs) -> Dict[int, float]:
        """모든 플레이어의 보상 계산"""
        rewards = {}
        
        for player_idx, calculator in self.calculators.items():
            reward = calculator.calculate_reward(
                raw_reward, raw_info, player_idx, **kwargs
            )
            rewards[player_idx] = reward
        
        return rewards


class RewardCalculatorFactory:
    """보상 계산기 생성 팩토리"""
    
    @staticmethod
    def create_calculator(calculator_type: str, **kwargs) -> RewardCalculator:
        """
        보상 계산기 생성
        
        Args:
            calculator_type: 계산기 타입
                - 'sparse': sparse 보상만
                - 'shaped': sparse + shaped 보상
                - 'learner': learner 기반 보상
                - 'custom': 커스텀 함수 사용
            **kwargs: 계산기별 추가 파라미터
            
        Returns:
            보상 계산기 인스턴스
        """
        if calculator_type == 'sparse':
            return SparseRewardCalculator(**kwargs)
        elif calculator_type == 'shaped':
            return ShapedRewardCalculator(**kwargs)
        elif calculator_type == 'learner':
            return LearnerBasedRewardCalculator(**kwargs)
        elif calculator_type == 'custom':
            return CustomRewardCalculator(**kwargs)
        else:
            raise ValueError(f"Unknown calculator type: {calculator_type}")


# 편의 함수들
def create_sparse_calculator(magnifier: float = 1.0):
    """Sparse 보상 계산기 생성"""
    return RewardCalculatorFactory.create_calculator('sparse', magnifier=magnifier)

def create_shaped_calculator(magnifier: float = 1.0, sparse_ratio: float = 0.1):
    """Shaped 보상 계산기 생성"""
    return RewardCalculatorFactory.create_calculator(
        'shaped', magnifier=magnifier, sparse_ratio=sparse_ratio
    )

def create_learner_calculator(learner_type: str = 'originaler', magnifier: float = 1.0):
    """Learner 기반 보상 계산기 생성"""
    return RewardCalculatorFactory.create_calculator(
        'learner', learner_type=learner_type, magnifier=magnifier
    )

def create_multi_agent_calculator(player_configs: Dict[int, Dict[str, Any]]):
    """
    다중 에이전트 보상 계산기 생성
    
    Args:
        player_configs: {player_idx: {'type': str, **kwargs}} 플레이어별 설정
        
    Returns:
        MultiAgentRewardCalculator 인스턴스
    """
    calculators = {}
    
    for player_idx, config in player_configs.items():
        calc_type = config.pop('type')
        calculator = RewardCalculatorFactory.create_calculator(calc_type, **config)
        calculators[player_idx] = calculator
    
    return MultiAgentRewardCalculator(calculators)


# 샘플 커스텀 보상 함수들
def collaboration_bonus_reward_fn(raw_reward: float, raw_info: Dict[str, Any], 
                                 player_idx: int, **kwargs) -> float:
    """협력 보너스가 포함된 보상 함수"""
    base_reward = raw_reward
    
    # 충돌 페널티
    collision_penalty = 0.0
    if raw_info.get('agent_collision_detected', False):
        collision_penalty = -0.1
    
    # 협력 보너스 (예: 동시에 같은 오브젝트 조작)
    cooperation_bonus = 0.0
    # 실제 구현에서는 raw_info에서 협력 행동 감지
    
    return base_reward + collision_penalty + cooperation_bonus

def efficiency_reward_fn(raw_reward: float, raw_info: Dict[str, Any], 
                        player_idx: int, **kwargs) -> float:
    """효율성 기반 보상 함수"""
    base_reward = raw_reward
    
    # 시간 효율성 보너스
    step_count = raw_info.get('step_count', 0)
    if step_count > 0 and base_reward > 0:
        # 빠르게 완료할수록 보너스
        efficiency_bonus = max(0, (400 - step_count) / 400 * 0.1)
    else:
        efficiency_bonus = 0.0
    
    return base_reward + efficiency_bonus
