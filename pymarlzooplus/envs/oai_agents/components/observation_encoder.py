"""
ObservationEncoder - 원시 상태를 신경망용 관찰로 변환

OvercookedCoreEnv가 반환한 raw_state를 받아 에이전트가 사용할 수 있는
숫자 배열 형태의 관찰로 변환합니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from gymnasium import spaces


class ObservationEncoder(ABC):
    """
    관찰 인코더 추상 클래스
    
    원시 상태 정보를 에이전트가 사용할 수 있는 관찰 형식으로 변환합니다.
    """
    
    @abstractmethod
    def encode(self, raw_state, mdp, player_idx: int, **kwargs) -> Dict[str, Any]:
        """
        원시 상태를 관찰로 인코딩
        
        Args:
            raw_state: OvercookedCoreEnv에서 반환한 원시 상태
            mdp: OvercookedGridworld 인스턴스
            player_idx: 관찰을 생성할 플레이어 인덱스
            **kwargs: 추가 파라미터
            
        Returns:
            인코딩된 관찰 딕셔너리
        """
        pass
    
    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """
        Gymnasium 호환 관찰 공간 정보 반환
        
        Returns:
            gymnasium.spaces 객체
        """
        pass


class OvercookedObservationEncoder(ObservationEncoder):
    """
    Overcooked 환경용 관찰 인코더
    
    기존 oai_agents.common.state_encodings의 함수들을 활용하여
    다양한 인코딩 방식을 지원합니다.
    """
    
    def __init__(self, encoding_scheme: str = 'OAI_lossless', grid_shape: tuple = (7, 7), 
                 horizon: int = 400, **kwargs):
        """
        Args:
            encoding_scheme: 인코딩 방식 ('OAI_lossless', 'OAI_feats' 등)
            grid_shape: 그리드 크기
            horizon: 에피소드 최대 길이
            **kwargs: 인코딩 함수별 추가 파라미터
        """
        self.encoding_scheme = encoding_scheme
        self.grid_shape = grid_shape
        self.horizon = horizon
        self.kwargs = kwargs
        
        # 기존 인코딩 함수 로드
        try:
            from oai_agents.common.state_encodings import ENCODING_SCHEMES
            self.encoding_fn = ENCODING_SCHEMES.get(encoding_scheme)
            if self.encoding_fn is None:
                raise ValueError(f"Unknown encoding scheme: {encoding_scheme}")
        except ImportError:
            # 기존 인코딩 함수가 없으면 기본 구현 사용
            self.encoding_fn = self._basic_encoding
            print(f"Warning: Could not import {encoding_scheme}, using basic encoding")
    
    def encode(self, raw_state, mdp, player_idx: int, **kwargs) -> Dict[str, Any]:
        """Overcooked 상태를 관찰로 인코딩"""
        try:
            # 기존 인코딩 함수 호출
            if self.encoding_scheme == 'OAI_feats':
                # Feature 기반 인코딩
                obs = self.encoding_fn(mdp, raw_state, self.grid_shape, self.horizon, 
                                     p_idx=player_idx, **self.kwargs)
            else:
                # 시각적 인코딩
                obs = self.encoding_fn(mdp, raw_state, self.grid_shape, self.horizon, 
                                     p_idx=player_idx, **self.kwargs)
            
            # 딕셔너리 형태로 반환되지 않은 경우 래핑
            if not isinstance(obs, dict):
                if self.encoding_scheme == 'OAI_feats':
                    obs = {'agent_obs': obs}
                else:
                    obs = {'visual_obs': obs}
                    
        except Exception as e:
            print(f"Warning: Encoding failed ({e}), using basic encoding")
            obs = self._basic_encoding(raw_state, mdp, player_idx)
        
        return obs
    
    def _basic_encoding(self, raw_state, mdp, player_idx: int) -> Dict[str, Any]:
        """기본 인코딩 구현 (fallback)"""
        # 기본적인 lossless encoding
        try:
            lossless_encoding = mdp.lossless_state_encoding(raw_state)
            return {'visual_obs': lossless_encoding}
        except:
            # 최후의 수단: 빈 관찰
            return {'visual_obs': np.zeros((*self.grid_shape, 27), dtype=np.float32)}
    
    def get_observation_space(self) -> spaces.Space:
        """Gymnasium 호환 관찰 공간 정보 반환"""
        if self.encoding_scheme == 'OAI_feats':
            # Feature 기반 인코딩: 1차원 벡터
            return spaces.Dict({
                'agent_obs': spaces.Box(0, 400, (96,), dtype=np.int32)
            })
        else:
            # 시각적 인코딩: 3차원 텐서
            if 'enhanced' in self.encoding_scheme:
                # Enhanced 인코딩: 기본 채널 + MLAM 특징
                base_channels = 27  # 기본 시각적 특징
                mlam_features_size = 15  # MLAM 특징
                
                return spaces.Dict({
                    'visual_obs': spaces.Box(0, 20, (base_channels, *self.grid_shape), dtype=np.int32),
                    'mlam_features': spaces.Box(-1, 20, (mlam_features_size,), dtype=np.float32)
                })
            else:
                # 기본 시각적 인코딩
                return spaces.Dict({
                    'visual_obs': spaces.Box(0, 20, (27, *self.grid_shape), dtype=np.int32)
                })


class MinimalObservationEncoder(ObservationEncoder):
    """
    최소한의 정보만 포함하는 간단한 관찰 인코더
    
    디버깅이나 빠른 실험을 위해 사용합니다.
    """
    
    def __init__(self, grid_shape: tuple = (7, 7)):
        self.grid_shape = grid_shape
    
    def encode(self, raw_state, mdp, player_idx: int, **kwargs) -> Dict[str, Any]:
        """최소 정보만 인코딩"""
        # 플레이어 위치와 기본 오브젝트만 인코딩
        visual_obs = np.zeros((*self.grid_shape, 5), dtype=np.float32)
        
        try:
            # 채널 0: 현재 플레이어 위치
            if len(raw_state.players) > player_idx:
                player = raw_state.players[player_idx]
                px, py = player.position
                if 0 <= px < self.grid_shape[0] and 0 <= py < self.grid_shape[1]:
                    visual_obs[px, py, 0] = 1.0
            
            # 채널 1-4: 다른 플레이어들
            channel = 1
            for i, player in enumerate(raw_state.players):
                if i != player_idx and channel < 5:
                    px, py = player.position
                    if 0 <= px < self.grid_shape[0] and 0 <= py < self.grid_shape[1]:
                        visual_obs[px, py, channel] = 1.0
                    channel += 1
                    
        except Exception:
            # 인코딩 실패 시 빈 관찰 반환
            pass
        
        return {'visual_obs': visual_obs}
    
    def get_observation_space(self) -> spaces.Space:
        """최소 관찰 공간"""
        return spaces.Dict({
            'visual_obs': spaces.Box(0, 1, (5, *self.grid_shape), dtype=np.float32)
        })


class HeuristicObservationEncoder(ObservationEncoder):
    """
    휴리스틱 에이전트용 관찰 인코더
    
    휴리스틱 에이전트는 원시 상태 정보를 직접 사용하므로
    최소한의 변환만 수행합니다.
    """
    
    def encode(self, raw_state, mdp, player_idx: int, **kwargs) -> Dict[str, Any]:
        """휴리스틱 에이전트용 관찰 생성"""
        return {
            'state': raw_state,
            'player_idx': player_idx,
            'mdp': mdp
        }
    
    def get_observation_space(self) -> spaces.Space:
        """휴리스틱 에이전트용 관찰 공간 (사용하지 않음)"""
        # 휴리스틱 에이전트는 Gym 공간을 사용하지 않지만 인터페이스 호환성을 위해 제공
        return spaces.Dict({
            'state': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        })


class ObservationEncoderFactory:
    """관찰 인코더 생성 팩토리"""
    
    @staticmethod
    def create_encoder(encoder_type: str, **kwargs) -> ObservationEncoder:
        """
        관찰 인코더 생성
        
        Args:
            encoder_type: 인코더 타입
                - 'overcooked': 기존 인코딩 함수 사용
                - 'minimal': 최소 정보만 인코딩
                - 'heuristic': 휴리스틱 에이전트용
            **kwargs: 인코더별 추가 파라미터
            
        Returns:
            관찰 인코더 인스턴스
        """
        if encoder_type == 'overcooked':
            return OvercookedObservationEncoder(**kwargs)
        elif encoder_type == 'minimal':
            return MinimalObservationEncoder(**kwargs)
        elif encoder_type == 'heuristic':
            return HeuristicObservationEncoder()
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")


# 편의 함수들
def create_lossless_encoder(grid_shape=(7, 7), horizon=400):
    """무손실 인코딩 사용하는 인코더 생성"""
    return ObservationEncoderFactory.create_encoder(
        'overcooked', 
        encoding_scheme='OAI_lossless',
        grid_shape=grid_shape,
        horizon=horizon
    )

def create_feature_encoder(grid_shape=(7, 7), horizon=400):
    """특징 기반 인코딩 사용하는 인코더 생성"""
    return ObservationEncoderFactory.create_encoder(
        'overcooked',
        encoding_scheme='OAI_feats', 
        grid_shape=grid_shape,
        horizon=horizon
    )

def create_enhanced_encoder(grid_shape=(7, 7), horizon=400, mlam=None):
    """Enhanced 인코딩 사용하는 인코더 생성"""
    return ObservationEncoderFactory.create_encoder(
        'overcooked',
        encoding_scheme='OAI_lossless_enhanced',
        grid_shape=grid_shape,
        horizon=horizon,
        mlam=mlam
    )
