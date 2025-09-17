"""
Overcooked 전문화된 컴포넌트 시스템

기존의 복잡한 OvercookedGymEnv를 여러 전문화된 클래스로 분리:

1. OvercookedCoreEnv - 순수 게임 엔진 (gym_environments/overcooked_core_env.py)
2. ObservationEncoder - 관찰 생성기
3. RewardCalculator - 보상 계산기  
4. GameCoordinator - 전체 조율자
5. API Wrapper - PettingZoo/Gym 호환성

각 컴포넌트는 단일 책임을 가지며, 독립적으로 테스트하고 교체할 수 있습니다.
"""

# 핵심 컴포넌트들
from .observation_encoder import (
    ObservationEncoder,
    OvercookedObservationEncoder,
    MinimalObservationEncoder,
    HeuristicObservationEncoder,
    ObservationEncoderFactory,
    create_lossless_encoder,
    create_feature_encoder,
    create_enhanced_encoder
)

from .reward_calculator import (
    RewardCalculator,
    SparseRewardCalculator,
    ShapedRewardCalculator,
    LearnerBasedRewardCalculator,
    CustomRewardCalculator,
    MultiAgentRewardCalculator,
    RewardCalculatorFactory,
    create_sparse_calculator,
    create_shaped_calculator,
    create_learner_calculator,
    create_multi_agent_calculator,
    collaboration_bonus_reward_fn,
    efficiency_reward_fn
)

from .game_coordinator import (
    GameCoordinator,
    SimpleAgentManager,
    create_game_coordinator,
    create_single_agent_coordinator
)

# API 래퍼들은 gym_environments 디렉토리로 이동됨

# 편의 함수들은 각각의 래퍼 모듈에서 제공됨

# 버전 정보
__version__ = "1.0.0"

# 주요 클래스들을 __all__에 포함
__all__ = [
    # 추상 클래스들
    'ObservationEncoder',
    'RewardCalculator',
    
    # 구체 클래스들
    'OvercookedObservationEncoder',
    'MinimalObservationEncoder', 
    'HeuristicObservationEncoder',
    'SparseRewardCalculator',
    'ShapedRewardCalculator',
    'LearnerBasedRewardCalculator',
    'CustomRewardCalculator',
    'MultiAgentRewardCalculator',
    'GameCoordinator',
    'SimpleAgentManager',
    
    # API 래퍼들은 gym_environments 디렉토리로 이동됨
    
    # 팩토리들
    'ObservationEncoderFactory',
    'RewardCalculatorFactory',
    
    # 편의 함수들
    'create_lossless_encoder',
    'create_feature_encoder',
    'create_enhanced_encoder',
    'create_sparse_calculator',
    'create_shaped_calculator',
    'create_learner_calculator',
    'create_multi_agent_calculator',
    'create_game_coordinator',
    'create_single_agent_coordinator',
    
    # 샘플 함수들
    'collaboration_bonus_reward_fn',
    'efficiency_reward_fn'
]
