#!/usr/bin/env python3
"""
Multi Overcooked 환경 통합 테스트 스크립트

이 스크립트는 멀티 에이전트 오버쿡드 환경이 Pymarlzooplus와 올바르게 통합되었는지 테스트합니다.
"""

import sys
import os
import numpy as np

# Pymarlzooplus 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pymarlzooplus'))

def test_basic_functionality():
    """기본 기능 테스트"""
    print("🔧 Testing basic Multi Overcooked integration...")
    
    try:
        # 환경 임포트 테스트
        from pymarlzooplus.envs import REGISTRY
        assert "multi_overcooked" in REGISTRY, "multi_overcooked not found in REGISTRY"
        print("✅ Environment registration successful")
        
        # 환경 생성 테스트
        env_fn = REGISTRY["multi_overcooked"]
        env = env_fn(
            layout_name="3_chefs_smartfactory",
            num_agents=3,
            encoding_scheme="OAI_lossless",
            reward_type="sparse",
            horizon=50  # 테스트용 짧은 에피소드
        )
        print("✅ Environment creation successful")
        
        # 인터페이스 테스트
        assert hasattr(env, 'reset'), "Missing reset method"
        assert hasattr(env, 'step'), "Missing step method"
        assert hasattr(env, 'get_obs'), "Missing get_obs method"
        assert hasattr(env, 'get_state'), "Missing get_state method"
        print("✅ Interface validation successful")
        
        # 환경 정보 테스트
        env_info = env.get_env_info()
        print(f"📊 Environment info: {env_info}")
        
        assert env_info['n_agents'] == 3, f"Expected 3 agents, got {env_info['n_agents']}"
        assert env_info['episode_limit'] == 50, f"Expected horizon 50, got {env_info['episode_limit']}"
        print("✅ Environment info validation successful")
        
        return env
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_episode_execution(env):
    """에피소드 실행 테스트"""
    print("\n🎮 Testing episode execution...")
    
    try:
        # 리셋 테스트
        obs, state = env.reset()
        print(f"✅ Reset successful - obs shape: {[o.shape for o in obs]}, state shape: {state.shape}")
        
        # 스텝 테스트
        n_agents = env.get_env_info()['n_agents']
        total_reward = 0
        steps = 0
        
        for step in range(10):  # 10스텝만 테스트
            # 랜덤 액션 생성
            actions = [np.random.randint(0, env.get_total_actions()) for _ in range(n_agents)]
            
            # 스텝 실행
            reward, done, info = env.step(actions)
            
            # 관찰 및 상태 업데이트
            obs = env.get_obs()
            state = env.get_state()
            
            total_reward += reward
            steps += 1
            
            print(f"  Step {step}: reward={reward:.3f}, done={done}")
            
            if done:
                print(f"  Episode finished at step {step}")
                break
        
        print(f"✅ Episode execution successful - {steps} steps, total reward: {total_reward:.3f}")
        
        # 정리
        env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Episode execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_configurations():
    """다양한 설정 테스트"""
    print("\n⚙️  Testing different configurations...")
    
    configurations = [
        {"layout_name": "3_chefs_smartfactory", "num_agents": 3, "encoding_scheme": "OAI_lossless"},
        {"layout_name": "3_chefs_smartfactory", "num_agents": 3, "encoding_scheme": "OAI_feats"},
        {"layout_name": "3_chefs_smartfactory", "num_agents": 3, "reward_type": "shaped"},
    ]
    
    success_count = 0
    
    for i, config in enumerate(configurations):
        try:
            print(f"  Config {i+1}: {config}")
            
            from pymarlzooplus.envs import REGISTRY
            env_fn = REGISTRY["multi_overcooked"]
            env = env_fn(horizon=20, **config)  # 짧은 에피소드
            
            # 빠른 테스트
            obs, state = env.reset()
            actions = [0] * config["num_agents"]  # 모든 에이전트가 정지
            reward, done, info = env.step(actions)
            
            env.close()
            print(f"    ✅ Success")
            success_count += 1
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
    
    print(f"✅ Configuration tests: {success_count}/{len(configurations)} passed")
    return success_count == len(configurations)

def test_algorithm_compatibility():
    """알고리즘 호환성 테스트"""
    print("\n🤖 Testing algorithm compatibility...")
    
    try:
        from pymarlzooplus.envs import REGISTRY
        env_fn = REGISTRY["multi_overcooked"]
        env = env_fn(
            layout_name="3_chefs_smartfactory",
            num_agents=3,
            horizon=20
        )
        
        # QMIX/VDN용 테스트 (중앙화된 상태)
        obs, state = env.reset()
        print(f"  QMIX/VDN - Global state shape: {state.shape}")
        
        # MAPPO용 테스트 (개별 관찰)
        individual_obs = env.get_obs()
        print(f"  MAPPO - Individual obs shapes: {[o.shape for o in individual_obs]}")
        
        # 가용 액션 테스트
        avail_actions = env.get_avail_actions()
        print(f"  Available actions per agent: {len(avail_actions[0])}")
        
        env.close()
        print("✅ Algorithm compatibility test successful")
        return True
        
    except Exception as e:
        print(f"❌ Algorithm compatibility test failed: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 Starting Multi Overcooked Integration Tests")
    print("=" * 50)
    
    # 기본 기능 테스트
    env = test_basic_functionality()
    if env is None:
        print("\n❌ Basic tests failed. Aborting.")
        return False
    
    # 에피소드 실행 테스트
    episode_success = test_episode_execution(env)
    
    # 다양한 설정 테스트
    config_success = test_different_configurations()
    
    # 알고리즘 호환성 테스트
    algo_success = test_algorithm_compatibility()
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print(f"  Basic functionality: ✅" if env else "  Basic functionality: ❌")
    print(f"  Episode execution: {'✅' if episode_success else '❌'}")
    print(f"  Configuration tests: {'✅' if config_success else '❌'}")
    print(f"  Algorithm compatibility: {'✅' if algo_success else '❌'}")
    
    all_passed = env and episode_success and config_success and algo_success
    
    if all_passed:
        print("\n🎉 All tests passed! Multi Overcooked integration is ready.")
        print("\nUsage examples:")
        print("python pymarlzooplus/main.py --config=qmix --env-config=multi_overcooked with env_args.num_agents=3")
        print("python pymarlzooplus/main.py --config=mappo --env-config=multi_overcooked with env_args.num_agents=4")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
