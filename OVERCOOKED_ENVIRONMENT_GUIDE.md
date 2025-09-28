# 🍳 PyMARLzoo+ 오버쿡드(Overcooked) 환경 가이드

## 📝 개요

오버쿡드는 여러 에이전트(요리사)가 협력하여 주문을 처리하는 게임입니다. 
기존 pymarlzooplus 패키지와 호환을 위해 멀티에이전트 환경을 통합했습니다.



## 🏗️ 코드 아키텍처

핵심 게임 로직(MDP)부터 PyMARLzoo+ 와의 연동을 위한 래퍼(Wrapper)까지 여러 계층으로 구성됩니다.

### 핵심 디렉토리 구조

```
pymarlzooplus/
├── envs/
│   ├── overcooked_ai/            # 🔵 오버쿡드 AI 원본 소스코드
│   │   └── src/overcooked_ai_py/
│   │       ├── mdp/              #  CORE: MDP(게임 규칙, 상태 전이)
│   │       ├── agents/           #      에이전트 로직
│   │       └── planning/         #      플래닝 알고리즘
│   ├── overcooked_wrapper.py     # 🟢 PyMARLzoo+ 2-에이전트 래퍼
│   └── multi_overcooked_wrapper.py # 🟢 PyMARLzoo+ 다중(3+) 에이전트 래퍼
└── config/
    └── envs/
        ├── overcooked.yaml       # 🟡 2-에이전트 환경 설정 파일
        └── multi_overcooked.yaml # 🟡 다중 에이전트 환경 설정 파일
```

### 핵심 컴포넌트: MDP에서 래퍼까지

1.  **`OvercookedGridworld`** (`mdp/overcooked_mdp.py`):
    * 환경의 가장 핵심적인 **"게임 엔진"** 역할
    * 맵 레이아웃, 상태 전이, 보상 함수, 레시피 등 모든 게임의 기본 규칙 정의

2.  **`OvercookedEnv`** (`mdp/overcooked_env.py`):
    * `OvercookedGridworld`를 한번 감싸서 Gymnasium `Env`와 유사한 인터페이스를 제공하는 기본 래퍼

3.  **`_OvercookedWrapper` & `_MultiOvercookedWrapper`**:
    * PyMARLzoo+의 `MultiAgentEnv` 클래스와 호환되도록 환경을 최종적으로 변환하는 **"통합 어댑터"**
    * `get_obs()`, `get_state()` 등 PyMARL 알고리즘에 필요한 모든 인터페이스를 구현

---

## 🚀 빠른 시작 (Quick Start)

### 1. 기본 2-에이전트 환경 실행

가장 일반적인 2인용 `cramped_room` 맵에서 QMIX 알고리즘을 실행하는 예제

```python
from pymarlzooplus import pymarlzooplus

params = {
    "config": "qmix",
    "env-config": "overcooked",
    "env_args": {
        "key": "cramped_room", # 실행할 맵 레이아웃
        "time_limit": 400,
        "reward_type": "sparse" # "sparse" 또는 "shaped"
    }
}

pymarlzooplus(params)
```

### 2. 다중 에이전트(3명 이상) 환경 실행

3명의 요리사가 협력하는 커스텀 맵 `3_chefs_smartfactory`를 실행하는 예제

```python
from pymarlzooplus import pymarlzooplus

params = {
    "config": "qmix",
    "env-config": "multi_overcooked", # 다중 에이전트 설정 파일 사용
    "env_args": {
        "layout_name": "3_chefs_smartfactory",
        "num_agents": 3,
        "horizon": 400
    }
}

pymarlzooplus(params)
```

### 3. 환경 API 직접 사용 (디버깅용)

환경과 직접 상호작용하여 테스트

```python
from pymarlzooplus.envs import REGISTRY as env_REGISTRY

# 환경 레지스트리에서 직접 생성
env_args = {"key": "asymmetric_advantages", "time_limit": 400}
env = env_REGISTRY["overcooked"](**env_args)

# 에피소드 실행
obs, state = env.reset()
done = False
episode_reward = 0

while not done:
    actions = env.sample_actions()  # 임의의 액션 샘플링
    reward, done, info = env.step(actions)
    episode_reward += reward

print(f"에피소드 종료! 총 보상: {episode_reward}")
```

---

## ⚙️ 환경 설정 (Configuration)

`pymarlzooplus/config/envs/` 경로의 `.yaml` 파일을 통해 환경의 세부 사항 설정

### `overcooked.yaml` (2-에이전트)

```yaml
env_args:
  key: "cramped_room"   # 레이아웃 이름 (layout_name과 동일)
  time_limit: 400       # 최대 에피소드 길이
  reward_type: "sparse" # 보상 타입: "sparse" 또는 "shaped"
  render: False         # 렌더링 활성화 여부
```

### `multi_overcooked.yaml` (다중 에이전트)

```yaml
env_args:
  layout_name: "3_chefs_smartfactory" # 레이아웃 이름
  num_agents: 3                     # 에이전트 수
  encoding_scheme: "OAI_lossless"   # 관찰 인코딩 방식
  horizon: 400                      # 최대 에피소드 길이
```

---

## 🎯 게임 핵심 기능

### 유명 레이아웃

다양한 협력 시나리오를 테스트할 수 있는 레이아웃들을 미리 정의했습니다.

* **`cramped_room`**
* **`asymmetric_advantages`**
* **`coordination_ring`**
* **`forced_coordination`**
* **`3_chefs_smartfactory`**: 스마트팩토리 시나리오 맞게 만든 새로운 layout

이외에도 추가적으로 layout_generatory.py를 통해 다양한 layout을 생성할 수 있습니다. 

### 보상 시스템 (Reward System)

* **Sparse Reward (희소 보상)**: 주문을 **최종적으로 서빙했을 때만 +20점**의 보상을 받습니다. 목표 지향적이지만 학습이 어렵습니다.
* **Shaped Reward (설계 보상)**: 목표 달성을 위한 중간 과정들에 대해 세분화된 보상을 제공하여 학습을 용이하게 합니다.
    * 냄비에 재료 넣기: +3점
    * 접시 집기: +3점
    * 요리된 수프 집기: +5점

### 레시피

양파(onion)와 토마토(tomato)를 조합하여 다양한 수프를 만들 수 있습니다.

```python
# 예시 레시피: 양파 3개로 구성된 수프
{"ingredients": ["onion", "onion", "onion"]}
```

---

## 🔧 고급 기능 및 심화 탐구

* **동적 레이아웃 생성 (`layout_generator.py`)**: 기존 맵 외에, 특정 규칙에 따라 새로운 맵을 동적으로 생성할 수 있습니다.
* **시각화 도구 (`visualization/`)**: `render: True` 옵션을 활성화하면 `pygame`을 통해 게임 화면을 실시간으로 볼 수 있습니다.
* **플래닝 알고리즘 (`planning/`)**: 강화학습 에이전트 외에도, 정해진 규칙에 따라 최적의 행동을 계획하는 Rule-based 에이전트(e.g., `AgentPair`)를 활용할 수 있습니다.

## 📊 성능 평가

주요 평가 지표는 다음과 같습니다.

* **주문 성공률 (Success Rate)**: 시간 내에 성공적으로 서빙한 주문의 비율
* **시간당 점수 (Score per Hour)**: 에피소드 길이 대비 획득한 총 점수

`test_nepisode`, `test_interval` 등의 인자를 통해 주기적인 성능 평가를 자동화할 수 있습니다.

---

## 📚 자료 및 참고사항

### 의존성

* Python 3.8+
* `gymnasium`, `numpy`
* `pygame` (렌더링 시 필요)

### 관련 자료

* **원본 Overcooked-AI 레포지토리**: [HumanCompatibleAI/overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai)
* **PyMARLzoo+ 관련 논문**: [https://arxiv.org/html/2502.04773v1]

