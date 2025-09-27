import argparse
import time
import numpy as np
from typing import Dict, List, Tuple, Any

from pymarlzooplus.envs import REGISTRY as env_REGISTRY
from pymarlzooplus.envs.oai_agents.policies.rule_based import RuleBasedPlanner


class RuleBasedTester:
    """Comprehensive tester for RuleBasedPlanner system."""
    
    def __init__(self, args):
        self.args = args
        self.planner = RuleBasedPlanner(prefer_onion=True, rng=np.random.default_rng(args.seed))
        self.test_results = {
            'episodes_completed': 0,
            'total_steps': 0,
            'total_rewards': 0,
            'subgoal_selections': {},
            'role_distributions': {},
            'stuck_incidents': 0,
            'collaboration_events': 0,
            'avg_steps_per_episode': 0,
            'success_rate': 0
        }
        
    def create_env(self, layout_name: str, num_agents: int):
        """Create environment with specified parameters."""
        return env_REGISTRY["multi_overcooked"](
            layout_name=layout_name,
            num_agents=num_agents,
            encoding_scheme="OAI_lossless",
            reward_type="sparse",
            horizon=400,
            render=self.args.render,
        )
    
    def run_comprehensive_test(self):
        """Run comprehensive test suite."""
        print("🚀 Starting Comprehensive Rule-Based Planner Test")
        print("=" * 80)
        
        # Test different scenarios
        test_scenarios = [
            {"layout": "3_chefs_smartfactory", "agents": 3, "episodes": 2, "description": "3-chef smart factory (main test)"},
            {"layout": "cramped_room", "agents": 2, "episodes": 2, "description": "Basic 2-agent cooperation"},
            {"layout": "asymmetric_advantages", "agents": 2, "episodes": 2, "description": "Asymmetric layout test"},
        ]
        
        for scenario in test_scenarios:
            print(f"\n🎯 Testing Scenario: {scenario['description']}")
            print(f"   Layout: {scenario['layout']}, Agents: {scenario['agents']}")
            print("-" * 60)
            
            try:
                self.run_scenario_test(scenario)
            except Exception as e:
                print(f"❌ Scenario failed: {e}")
                continue
        
        self.print_final_results()
    
    def run_scenario_test(self, scenario: Dict[str, Any]):
        """Run test for a specific scenario."""
        env = self.create_env(scenario["layout"], scenario["agents"])
        
        for ep in range(scenario["episodes"]):
            print(f"\n📋 Episode {ep + 1}/{scenario['episodes']}")
            
            # Reset environment
            env.reset(seed=self.args.seed + ep)
            base_env = env.env
            mdp = base_env.mdp
            state = base_env.state
            
            # Initialize episode tracking
            episode_stats = {
                'steps': 0,
                'reward': 0,
                'subgoal_changes': 0,
                'stuck_turns': 0,
                'collaboration_turns': 0
            }
            
            # Print initial setup
            self.analyze_initial_setup(state, mdp)
            
            # Run episode
            prev_subgoals = {}
            
            for t in range(self.args.steps):
                if t % 20 == 0:  # Reduce output frequency
                    print(f"\n⏰ Step {t}")
                
                # Get actions from planner
                actions = self.planner.act(env)
                
                # Track statistics
                self.track_episode_stats(episode_stats, prev_subgoals, state, mdp)
                
                # Render if requested
                if self.args.render and t % 5 == 0:
                    env.render()
                    time.sleep(0.1)
                
                # Step environment
                reward, done, info = env.step(actions)
                episode_stats['reward'] += sum(reward) if isinstance(reward, list) else reward
                episode_stats['steps'] = t + 1
                
                if done:
                    print(f"✅ Episode completed at step {t}")
                    break
            
            # Update global statistics
            self.update_global_stats(episode_stats)
            self.print_episode_summary(episode_stats)
    
    def analyze_initial_setup(self, state, mdp):
        """Analyze and print initial game setup."""
        print("\n🏗️  Initial Setup Analysis:")
        
        # Analyze agent capabilities
        ctxs = self.planner._gather_agent_contexts(state, mdp)
        
        for ctx in ctxs:
            roles = self.planner._get_agent_possible_roles(ctx)
            feasible = self.planner._feasible_subgoals(ctx, state, mdp)
            
            print(f"  Agent {ctx.idx}:")
            print(f"    Position: {ctx.position}")
            print(f"    Holding: {ctx.holding}")
            print(f"    Capabilities: {ctx.capabilities}")
            print(f"    Possible Roles: {roles}")
            print(f"    Initial Feasible Subgoals: {feasible}")
            
            # Track role distributions
            for role in roles:
                if role not in self.test_results['role_distributions']:
                    self.test_results['role_distributions'][role] = 0
                self.test_results['role_distributions'][role] += 1
        
        # Analyze shared counters
        layout_name = getattr(mdp, "layout_name", "unknown")
        if layout_name in self.planner._shared_counters_cache:
            shared_counters = self.planner._shared_counters_cache[layout_name]
            print(f"  Shared Counters: {len(shared_counters)} pairs")
            for pair, counters in shared_counters.items():
                print(f"    Agents {pair}: {len(counters)} shared counters at {counters}")
    
    def track_episode_stats(self, episode_stats: Dict, prev_subgoals: Dict, state, mdp):
        """Track various statistics during episode."""
        ctxs = self.planner._gather_agent_contexts(state, mdp)
        
        for ctx in ctxs:
            # Check for stuck agents
            if self.planner._is_agent_stuck(ctx):
                episode_stats['stuck_turns'] += 1
                self.test_results['stuck_incidents'] += 1
            
            # Track subgoal selections
            feasible = self.planner._feasible_subgoals(ctx, state, mdp)
            if feasible:
                current_subgoal = feasible[0] if feasible else None
                
                # Track subgoal changes
                if ctx.idx in prev_subgoals and prev_subgoals[ctx.idx] != current_subgoal:
                    episode_stats['subgoal_changes'] += 1
                
                prev_subgoals[ctx.idx] = current_subgoal
                
                # Track subgoal frequency
                if current_subgoal:
                    if current_subgoal not in self.test_results['subgoal_selections']:
                        self.test_results['subgoal_selections'][current_subgoal] = 0
                    self.test_results['subgoal_selections'][current_subgoal] += 1
            
            # Detect collaboration (items on shared counters)
            if self.detect_collaboration(state, mdp):
                episode_stats['collaboration_turns'] += 1
                self.test_results['collaboration_events'] += 1
    
    def detect_collaboration(self, state, mdp) -> bool:
        """Detect if collaboration is happening (items on counters)."""
        counter_objects = mdp.get_counter_objects_dict(state)
        total_items_on_counters = sum(len(positions) for positions in counter_objects.values())
        return total_items_on_counters > 0
    
    def update_global_stats(self, episode_stats: Dict):
        """Update global test statistics."""
        self.test_results['episodes_completed'] += 1
        self.test_results['total_steps'] += episode_stats['steps']
        self.test_results['total_rewards'] += episode_stats['reward']
        
        # Calculate averages
        if self.test_results['episodes_completed'] > 0:
            self.test_results['avg_steps_per_episode'] = (
                self.test_results['total_steps'] / self.test_results['episodes_completed']
            )
            self.test_results['success_rate'] = (
                self.test_results['total_rewards'] / self.test_results['episodes_completed']
            )
    
    def print_episode_summary(self, episode_stats: Dict):
        """Print summary of episode performance."""
        print(f"\n📊 Episode Summary:")
        print(f"  Steps: {episode_stats['steps']}")
        print(f"  Total Reward: {episode_stats['reward']}")
        print(f"  Subgoal Changes: {episode_stats['subgoal_changes']}")
        print(f"  Stuck Turns: {episode_stats['stuck_turns']}")
        print(f"  Collaboration Turns: {episode_stats['collaboration_turns']}")
    
    def print_final_results(self):
        """Print comprehensive test results."""
        print("\n" + "=" * 80)
        print("🏆 COMPREHENSIVE TEST RESULTS")
        print("=" * 80)
        
        print(f"\n📈 Overall Performance:")
        print(f"  Episodes Completed: {self.test_results['episodes_completed']}")
        print(f"  Total Steps: {self.test_results['total_steps']}")
        print(f"  Total Rewards: {self.test_results['total_rewards']}")
        print(f"  Average Steps/Episode: {self.test_results['avg_steps_per_episode']:.1f}")
        print(f"  Average Reward/Episode: {self.test_results['success_rate']:.2f}")
        
        print(f"\n🎭 Role Distribution:")
        for role, count in sorted(self.test_results['role_distributions'].items()):
            role_names = {
                1: "Direct Onion Supplier", 2: "Onion Sharing Supplier", 3: "Onion Receiver & Cooker",
                4: "Direct Dish Supplier", 5: "Dish Sharing Supplier", 6: "Dish Receiver & Soup Picker", 
                7: "Full Service", 8: "Soup Maker & Sharer", 9: "Soup Receiver & Deliverer"
            }
            print(f"  Role {role} ({role_names.get(role, 'Unknown')}): {count} times")
        
        print(f"\n🎯 Subgoal Selection Frequency:")
        sorted_subgoals = sorted(self.test_results['subgoal_selections'].items(), 
                               key=lambda x: x[1], reverse=True)
        for subgoal, count in sorted_subgoals[:10]:  # Top 10
            print(f"  {subgoal}: {count} times")
        
        print(f"\n🚨 Issues Detected:")
        print(f"  Stuck Incidents: {self.test_results['stuck_incidents']}")
        print(f"  Collaboration Events: {self.test_results['collaboration_events']}")
        
        print(f"\n✅ Test Quality Assessment:")
        if self.test_results['stuck_incidents'] < 10:
            print("  ✅ Low stuck incidents - Good pathfinding")
        else:
            print("  ⚠️  High stuck incidents - Check pathfinding logic")
            
        if self.test_results['collaboration_events'] > 5:
            print("  ✅ Good collaboration - Agents working together")
        else:
            print("  ⚠️  Low collaboration - Check shared counter logic")
            
        if self.test_results['success_rate'] > 0:
            print("  ✅ Positive rewards achieved")
        else:
            print("  ⚠️  No rewards achieved - Check goal completion")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Rule-Based Planner Tester")
    parser.add_argument("--layout", type=str, default="3_chefs_smartfactory", 
                       help="Layout to test (default: 3_chefs_smartfactory)")
    parser.add_argument("--episodes", type=int, default=2, 
                       help="Episodes per scenario (default: 2)")
    parser.add_argument("--steps", type=int, default=100, 
                       help="Max steps per episode (default: 100)")
    parser.add_argument("--render", action="store_true", 
                       help="Enable rendering")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed (default: 42)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode (detailed output)")
    parser.add_argument("--single", action="store_true",
                       help="Run single scenario test")
    
    args = parser.parse_args()
    
    tester = RuleBasedTester(args)
    
    if args.single:
        # Single scenario test
        print("🔍 Running Single Scenario Test")
        # Determine number of agents based on layout
        agents_map = {
            "3_chefs_smartfactory": 3,
            "cramped_room": 2,
            "asymmetric_advantages": 2,
            "coordination_ring": 2,
        }
        num_agents = agents_map.get(args.layout, 2)  # Default to 2 agents
        
        scenario = {
            "layout": args.layout, 
            "agents": num_agents, 
            "episodes": args.episodes, 
            "description": f"Single test on {args.layout} ({num_agents} agents)"
        }
        tester.run_scenario_test(scenario)
        tester.print_final_results()
    else:
        # Comprehensive test
        tester.run_comprehensive_test()


if __name__ == "__main__":
    main()


