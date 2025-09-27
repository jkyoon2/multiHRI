from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

import numpy as np
from collections import deque

try:
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, Action
    from overcooked_ai_py.mdp.actions import Direction
except Exception:  # pragma: no cover
    OvercookedGridworld = object  # type: ignore
    OvercookedState = object  # type: ignore
    class Action:  # type: ignore
        # Fallback tokens mirroring OAI indexing:
        # 0:NORTH, 1:SOUTH, 2:EAST, 3:WEST, 4:STAY, 5:INTERACT
        STAY = "stay"
        INTERACT = "interact"
        ACTION_TO_INDEX = {
            (0, -1): 0,  # NORTH
            (0, 1): 1,   # SOUTH
            (1, 0): 2,   # EAST
            (-1, 0): 3,  # WEST
            STAY: 4,
            INTERACT: 5,
        }
    class Direction:  # type: ignore
        NORTH = (0, -1)
        SOUTH = (0, 1)
        EAST = (1, 0)
        WEST = (-1, 0)


SUBGOALS = [
    "pickup_onion_from_dispenser",
    "pickup_onion_from_counter",
    "place_onion_on_counter",
    "pickup_dish_from_dispenser",
    "pickup_dish_from_counter",
    "place_dish_on_counter",
    "put_onion_in_pot",
    "pickup_onion_soup",
    "pickup_soup_from_counter",
    "place_soup_on_counter",
    "deliver_to_serve",
    "clear_counter_for_dish",
    "clear_counter_for_onion",
]


@dataclass
class AgentContext:
    idx: int
    position: Tuple[int, int]
    holding: Optional[str]  # None | 'onion' | 'tomato' | 'dish' | 'soup'
    # Static, layout-dependent fields (computed once per layout and cached)
    reachable: Optional[Set[Tuple[int, int]]] = None
    capabilities: Optional[Dict[str, bool]] = None  # keys: onion_disp, dish_disp, pot, serve


class RuleBasedPlanner:
    """
    Lightweight rule-based multi-agent planner for Overcooked (multi_overcooked_wrapper backend).

    - Computes feasible subgoals per agent based on agent holding and pot states (coarse checks)
    - Selects a target location according to the provided priority rules
    - Returns a single-step action per agent towards target (or INTERACT)

    NOTE: This is a skeleton. Many environment-specific getters are left as stubs and should be
    implemented with precise calls into OvercookedGridworld/OvercookedState/MLAM as needed.
    """

    def __init__(self, prefer_onion: bool = True, rng: Optional[np.random.Generator] = None):
        self.prefer_onion = prefer_onion
        self.rng = rng or np.random.default_rng(0)
        # layout_name -> { agent_idx: { 'reachable': set, 'capabilities': {...} } }
        self._static_cache: Dict[str, Dict[int, Dict[str, object]]] = {}
        # layout_name -> { (agent_i, agent_j): [shared_counter_positions] }
        self._shared_counters_cache: Dict[str, Dict[Tuple[int, int], List[Tuple[int, int]]]] = {}
        # Track agent positions for stuck detection
        self._position_history: Dict[int, List[Tuple[int, int]]] = {}
        self._stuck_counter: Dict[int, int] = {}
        self._max_stuck_turns = 5  # If stuck for 5 turns, try alternative path
        
        # 서브골 우선순위 딕셔너리 - 높을수록 우선순위 높음
        # 게임 플로우: Empty Pot → Onion → Cooking → Ready Pot → Dish → Soup → Serve
        self.SUBGOAL_PRIORITY = {
            # === 최고 우선순위: 점수 획득 ===
            "deliver_to_serve": 10,          # 최종 목표 - 점수 획득
            
            # === 높은 우선순위: 완성된 요리 처리 ===
            "pickup_onion_soup": 9,          # Ready pot에서 soup 픽업 (dish 필요)
            "pickup_soup_from_counter": 8,   # 공유된 soup 픽업
            
            # === 중간 우선순위: 요리 과정 ===
            "put_onion_in_pot": 7,           # 요리 시작/진행 (onion 필요)
            
            # === 카운터 정리: 상황별 차등 우선순위 ===
            "clear_counter_for_dish": 6,     # Ready pot 상황에서 dish 공간 확보
            "clear_counter_for_onion": 5,    # Empty pot 상황에서 onion 공간 확보
            
            # === 재료/도구 획득: 상황별 차등 우선순위 ===
            "pickup_dish_from_dispenser": 4, # Ready pot 상황에서 dish 직접 획득
            "pickup_dish_from_counter": 4,   # Ready pot 상황에서 dish 공유 획득
            "pickup_onion_from_dispenser": 3,# Empty pot 상황에서 onion 직접 획득
            "pickup_onion_from_counter": 3,  # Empty pot 상황에서 onion 공유 획득
            
            # === 낮은 우선순위: 협업 지원 ===
            "place_soup_on_counter": 2,      # Soup 공유 (serve 불가능한 에이전트)
            "place_dish_on_counter": 1,      # Dish 공유 (dish_disp 가능한 에이전트)
            "place_onion_on_counter": 1,     # Onion 공유 (onion_disp 가능한 에이전트)
            
            # === 특수 상황: 회피/양보 ===
            "evade_deadlock": -1,            # 데드락 회피
            "yield_to_deadlock": -1,         # 데드락 양보
            None: 0                          # 목표 없음
        }

    # ---------- Public API ----------

    def act(self, env) -> List[int]:
        """Compute one action per agent with dynamic replanning every turn.

        env: _MultiOvercookedWrapper
        returns: list of int action indices (len == n_agents)
        """
        base_env = env.env  # OvercookedGymEnv
        state = base_env.state  # OvercookedState
        mdp = base_env.mdp  # OvercookedGridworld

        print(f"\n{'='*60}")
        print(f"TURN START - Timestep: {getattr(state, 'timestep', 'N/A')}")
        print(f"{'='*60}")

        # Print environment state
        self._print_environment_state(state, mdp)

        agent_ctxs = self._gather_agent_contexts(state, mdp)
        
        # First pass: compute subgoals and targets for all agents
        agent_plans = []
        for ctx in agent_ctxs:
            print(f"\n--- AGENT {ctx.idx} INITIAL PLANNING ---")
            plan = self._compute_agent_plan(ctx, state, mdp)
            agent_plans.append(plan)
        
        # Resolve conflicts between agents with same subgoal and target
        resolved_plans = self._resolve_agent_conflicts(agent_plans, state, mdp)
        
        # Resolve movement deadlocks between agents
        resolved_plans = self._resolve_movement_deadlocks(resolved_plans, state, mdp)
        
        # Second pass: execute actions based on resolved plans
        actions: List[int] = []
        for i, ctx in enumerate(agent_ctxs):
            print(f"\n--- AGENT {ctx.idx} ACTION EXECUTION ---")
            plan = resolved_plans[i]
            action = self._execute_agent_plan(ctx, plan, state, mdp)
            actions.append(action)
        
        print(f"\n{'='*60}")
        print(f"TURN END - Actions: {actions}")
        print(f"{'='*60}")
        
        return actions

    # ---------- Core logic ----------

    def _gather_agent_contexts(self, state: OvercookedState, mdp: OvercookedGridworld) -> List[AgentContext]:
        ctxs: List[AgentContext] = []
        layout_name = getattr(mdp, "layout_name", "unknown_layout")
        # Build static cache for this layout if missing
        if layout_name not in self._static_cache:
            self._static_cache[layout_name] = {}
            valid_tiles = set(mdp.get_valid_player_positions())
            agent_reachable_areas = {}
            
            # Use current state positions as seed; layout is static so reachability region is invariant
            for i, player in enumerate(state.players):
                start = tuple(player.position)
                reachable = self._bfs_reachable(valid_tiles, start)
                caps = self._compute_capabilities(reachable, mdp)
                self._static_cache[layout_name][i] = {"reachable": reachable, "capabilities": caps}
                agent_reachable_areas[i] = reachable
            
            # Compute shared counters between all agent pairs
            shared_counters = self._compute_shared_counters(agent_reachable_areas, mdp)
            self._shared_counters_cache[layout_name] = shared_counters

        for i, player in enumerate(state.players):
            holding = None
            if player.has_object():
                holding = getattr(player.get_object(), "name", None)
            static = self._static_cache[layout_name][i]
            ctxs.append(AgentContext(idx=i,
                                     position=tuple(player.position),
                                     holding=holding,
                                     reachable=static["reachable"],
                                     capabilities=static["capabilities"]))
        return ctxs

    def _compute_agent_plan(self, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld) -> Dict:
        """Compute subgoal and target for an agent without executing the action."""
        # Print agent context
        print(f"Agent {ctx.idx} Context:")
        print(f"  Position: {ctx.position}")
        print(f"  Holding: {ctx.holding}")
        print(f"  Capabilities: {ctx.capabilities}")
        
        # Update position history for stuck detection
        self._update_position_history(ctx)
        
        # Check if agent is stuck and needs alternative planning
        is_stuck = self._is_agent_stuck(ctx)
        if is_stuck:
            print(f"  🚨 Agent {ctx.idx} is STUCK! (same position for {self._max_stuck_turns} turns)")
        
        # Get agent roles
        possible_roles = self._get_agent_possible_roles(ctx)
        print(f"  Possible Roles: {possible_roles}")
        
        # Check if previous subgoal was achieved (based on holding state change or previous action)
        previous_holding = getattr(self, f'_prev_holding_{ctx.idx}', None)
        previous_action = getattr(self, f'_prev_action_{ctx.idx}', None)
        
        # Detect subgoal achievement
        subgoal_achieved = False
        
        # Only detect achievement if previous action was INTERACT
        # Holding state changes can happen automatically from environment
        if previous_action == Action.ACTION_TO_INDEX[Action.INTERACT]:
            previous_subgoal = getattr(self, f'_prev_subgoal_{ctx.idx}', None)
            print(f"  🎯 Previous action was INTERACT with subgoal: {previous_subgoal}")
            
            # Check if INTERACT action achieved the expected subgoal based on current holding state
            if previous_subgoal == "pickup_onion_from_dispenser":
                if ctx.holding == "onion" and previous_holding != "onion":
                    print(f"  ✅ Successfully picked up onion from dispenser")
                    subgoal_achieved = True
                else:
                    print(f"  ❌ Failed to pick up onion from dispenser")
            elif previous_subgoal == "pickup_dish_from_dispenser":
                if ctx.holding == "dish" and previous_holding != "dish":
                    print(f"  ✅ Successfully picked up dish from dispenser")
                    subgoal_achieved = True
                else:
                    print(f"  ❌ Failed to pick up dish from dispenser")
            elif previous_subgoal in ["pickup_onion_from_counter", "pickup_dish_from_counter", "pickup_soup_from_counter"]:
                # Check if we picked up the expected item
                expected_item = "onion" if "onion" in previous_subgoal else ("dish" if "dish" in previous_subgoal else "soup")
                if ctx.holding == expected_item and previous_holding != expected_item:
                    print(f"  ✅ Successfully picked up {expected_item} from counter")
                    subgoal_achieved = True
                else:
                    print(f"  ❌ Failed to pick up {expected_item} from counter")
            elif previous_subgoal in ["place_onion_on_counter", "put_onion_in_pot"]:
                if ctx.holding != "onion" and previous_holding == "onion":
                    print(f"  ✅ Successfully placed/put onion")
                    subgoal_achieved = True
                else:
                    print(f"  ❌ Failed to place/put onion")
            elif previous_subgoal in ["place_dish_on_counter"]:
                if ctx.holding != "dish" and previous_holding == "dish":
                    print(f"  ✅ Successfully placed dish")
                    subgoal_achieved = True
                else:
                    print(f"  ❌ Failed to place dish")
            elif previous_subgoal == "pickup_onion_soup":
                if ctx.holding == "soup" and previous_holding == "dish":
                    print(f"  ✅ Successfully picked up soup")
                    subgoal_achieved = True
                else:
                    print(f"  ❌ Failed to pick up soup")
            elif previous_subgoal in ["deliver_to_serve", "place_soup_on_counter"]:
                if ctx.holding != "soup" and previous_holding == "soup":
                    print(f"  ✅ Successfully delivered/placed soup")
                    subgoal_achieved = True
                else:
                    print(f"  ❌ Failed to deliver/place soup")
            else:
                print(f"  ⚠️  Unknown subgoal for achievement detection: {previous_subgoal}")
                # Don't assume achievement for unknown subgoals
        
        if subgoal_achieved:
            if ctx.holding is not None:
                print(f"  ✅ Subgoal achieved! Agent {ctx.idx} now holding {ctx.holding}")
            elif previous_holding is not None:
                print(f"  ✅ Subgoal achieved! Agent {ctx.idx} released {previous_holding}")
            else:
                print(f"  ✅ Subgoal achieved! Agent {ctx.idx} completed interaction")
            
            # Reset stuck counter when subgoal is achieved
            self._reset_stuck_counter(ctx.idx)
            print(f"  🔄 Stuck counter reset due to subgoal achievement")
        
        # Store current state for next turn comparison
        setattr(self, f'_prev_holding_{ctx.idx}', ctx.holding)
        
        # Always recompute feasible subgoals (dynamic adaptation)
        feasible = self._feasible_subgoals(ctx, state, mdp)
        print(f"  Feasible Subgoals: {feasible}")
        
        # Select target with stuck-aware planning
        if is_stuck:
            print(f"  🔄 Using STUCK mode - but still trying to find valid targets")
            # STUCK 모드에서도 올바른 타겟을 찾아보자
            adj_target = self._select_target_for(ctx, feasible, state, mdp, is_stuck)
            selected_subgoal = getattr(self, f'_prev_subgoal_{ctx.idx}', None)
            
            # 타겟을 찾지 못했을 때만 None으로 설정 (랜덤 액션용)
            if adj_target is None:
                print(f"  ❌ No valid target found even in STUCK mode - will use random action")
                selected_subgoal = None
        else:
            adj_target = self._select_target_for(ctx, feasible, state, mdp, is_stuck)
            selected_subgoal = getattr(self, f'_prev_subgoal_{ctx.idx}', None)
        
        print(f"  Selected Subgoal: {selected_subgoal}")
        print(f"  Selected Target: {adj_target}")
        
        return {
            'agent_idx': ctx.idx,
            'context': ctx,
            'subgoal': selected_subgoal,
            'target': adj_target,
            'feasible_subgoals': feasible,
            'is_stuck': is_stuck
        }

    def _resolve_agent_conflicts(self, agent_plans: List[Dict], state: OvercookedState, mdp: OvercookedGridworld) -> List[Dict]:
        """Resolve conflicts when multiple agents have same subgoal and target."""
        print(f"\n🔍 CONFLICT RESOLUTION:")
        
        # Group agents by (subgoal, target) pairs
        conflict_groups = {}
        for plan in agent_plans:
            if plan['subgoal'] and plan['target']:
                key = (plan['subgoal'], plan['target'])
                if key not in conflict_groups:
                    conflict_groups[key] = []
                conflict_groups[key].append(plan)
        
        # Find conflicts (groups with more than one agent)
        conflicts = {k: v for k, v in conflict_groups.items() if len(v) > 1}
        
        if not conflicts:
            print("  ✅ No conflicts detected")
            return agent_plans
        
        print(f"  ⚠️  Found {len(conflicts)} conflicts:")
        for (subgoal, target), conflicting_plans in conflicts.items():
            agent_indices = [p['agent_idx'] for p in conflicting_plans]
            print(f"    - Subgoal '{subgoal}' at target {target}: Agents {agent_indices}")
        
        # Resolve each conflict
        resolved_plans = agent_plans.copy()
        for (subgoal, target), conflicting_plans in conflicts.items():
            print(f"\n  🔧 Resolving conflict for subgoal '{subgoal}' at {target}:")
            
            # Sort by agent index for deterministic resolution
            conflicting_plans.sort(key=lambda p: p['agent_idx'])
            
            # First agent keeps the original plan
            winner = conflicting_plans[0]
            print(f"    ✅ Agent {winner['agent_idx']} keeps original plan")
            
            # Other agents need alternative plans
            for plan in conflicting_plans[1:]:
                ctx = plan['context']
                print(f"    🔄 Finding alternative for Agent {ctx.idx}:")
                
                # Try alternative targets for same subgoal first
                alt_target = self._find_alternative_target_for_subgoal(ctx, subgoal, target, state, mdp)
                if alt_target:
                    plan['target'] = alt_target
                    print(f"      ✅ Alternative target found: {alt_target}")
                else:
                    # Try alternative subgoal
                    alt_subgoal, alt_target = self._find_alternative_subgoal_and_target(ctx, plan['feasible_subgoals'], subgoal, state, mdp)
                    if alt_subgoal and alt_target:
                        plan['subgoal'] = alt_subgoal
                        plan['target'] = alt_target
                        setattr(self, f'_prev_subgoal_{ctx.idx}', alt_subgoal)
                        print(f"      ✅ Alternative subgoal found: '{alt_subgoal}' at {alt_target}")
                    else:
                        # No alternatives found, mark for random action
                        plan['subgoal'] = None
                        plan['target'] = None
                        print(f"      ❌ No alternatives found, will use random action")
                
                # Update resolved plans
                for i, resolved_plan in enumerate(resolved_plans):
                    if resolved_plan['agent_idx'] == plan['agent_idx']:
                        resolved_plans[i] = plan
                        break
        
        return resolved_plans

    def _find_alternative_target_for_subgoal(self, ctx: AgentContext, subgoal: str, blocked_target: Tuple[int, int], state: OvercookedState, mdp: OvercookedGridworld) -> Optional[Tuple[int, int]]:
        """Find alternative target for the same subgoal."""
        # Get all possible targets for this subgoal
        if subgoal == "pickup_onion_from_dispenser":
            all_targets = [self._nearest_reachable_adjacent(ctx, [pos], mdp) for pos in self._get_onion_dispensers(mdp)]
        elif subgoal == "pickup_dish_from_dispenser":
            all_targets = [self._nearest_reachable_adjacent(ctx, [pos], mdp) for pos in self._get_dish_dispensers(mdp)]
        elif subgoal == "put_onion_in_pot":
            pot_states = mdp.get_pot_states(state)
            available_pots = list(mdp.get_empty_pots(pot_states)) + list(mdp.get_partially_full_pots(pot_states))
            all_targets = [self._nearest_reachable_adjacent(ctx, [pos], mdp) for pos in available_pots]
        elif subgoal == "pickup_onion_soup":
            # Only ready pots can have soup picked up
            pot_states = mdp.get_pot_states(state)
            ready_pots = list(mdp.get_ready_pots(pot_states))
            all_targets = [self._nearest_reachable_adjacent(ctx, [pos], mdp) for pos in ready_pots]
        elif subgoal == "deliver_to_serve":
            all_targets = [self._nearest_reachable_adjacent(ctx, [pos], mdp) for pos in self._get_serve_tiles(mdp)]
        else:
            # For counter-based subgoals, it's harder to find alternatives
            return None
        
        # Filter out None values and blocked target
        valid_targets = [t for t in all_targets if t and t != blocked_target and self._is_target_achievable(t, ctx, state, mdp)]
        
        if valid_targets:
            # Return nearest alternative
            return min(valid_targets, key=lambda t: self._manhattan(ctx.position, t))
        
        return None

    def _find_alternative_subgoal_and_target(self, ctx: AgentContext, feasible_subgoals: List[str], blocked_subgoal: str, state: OvercookedState, mdp: OvercookedGridworld) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
        """Find alternative subgoal and target by trying different subgoals."""
        # Sort feasible subgoals by priority (highest priority first), excluding blocked subgoal
        available_subgoals = [g for g in feasible_subgoals if g != blocked_subgoal]
        subgoals_with_priority = [(g, self.SUBGOAL_PRIORITY.get(g, 0)) for g in available_subgoals]
        subgoals_with_priority.sort(key=lambda x: x[1], reverse=True)
        
        for subgoal, priority_value in subgoals_with_priority:
            # For each alternative subgoal, try to find multiple possible targets
            # This is different from _target_for which returns only the "best" target
            possible_targets = self._get_all_possible_targets_for_subgoal(subgoal, ctx, state, mdp)
            
            for target in possible_targets:
                if target and self._is_target_achievable(target, ctx, state, mdp):
                    return subgoal, target
        
        return None, None

    def _get_all_possible_targets_for_subgoal(self, subgoal: str, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld) -> List[Tuple[int, int]]:
        """Get all possible targets for a given subgoal, not just the best one."""
        targets = []
        
        if subgoal == "pickup_onion_from_dispenser":
            dispensers = self._get_onion_dispensers(mdp)
            for dispenser in dispensers:
                adjacent_pos = self._nearest_reachable_adjacent(ctx, [dispenser], mdp)
                if adjacent_pos:
                    targets.append(adjacent_pos)
                    
        elif subgoal == "pickup_dish_from_dispenser":
            dispensers = self._get_dish_dispensers(mdp)
            for dispenser in dispensers:
                adjacent_pos = self._nearest_reachable_adjacent(ctx, [dispenser], mdp)
                if adjacent_pos:
                    targets.append(adjacent_pos)
                    
        elif subgoal == "put_onion_in_pot":
            pot_states = mdp.get_pot_states(state)
            available_pots = list(mdp.get_empty_pots(pot_states)) + list(mdp.get_partially_full_pots(pot_states))
            for pot in available_pots:
                adjacent_pos = self._nearest_reachable_adjacent(ctx, [pot], mdp)
                if adjacent_pos:
                    targets.append(adjacent_pos)
                    
        elif subgoal == "deliver_to_serve":
            serve_areas = self._get_serve_areas(mdp)
            for serve_area in serve_areas:
                adjacent_pos = self._nearest_reachable_adjacent(ctx, [serve_area], mdp)
                if adjacent_pos:
                    targets.append(adjacent_pos)
        
        # For other subgoals, fall back to _target_for result
        else:
            target = self._target_for(subgoal, ctx, state, mdp)
            if target:
                targets.append(target)
        
        # Sort by distance and return
        if targets:
            targets.sort(key=lambda t: self._manhattan(ctx.position, t))
        
        return targets

    def _execute_agent_plan(self, ctx: AgentContext, plan: Dict, state: OvercookedState, mdp: OvercookedGridworld) -> int:
        """Execute the agent's plan and return the action."""
        subgoal = plan['subgoal']
        target = plan['target']
        
        print(f"Agent {ctx.idx} executing plan:")
        print(f"  Subgoal: {subgoal}")
        print(f"  Target: {target}")
        
        # Don't immediately stay if target equals current position - might need to interact
        if target is None:
            if subgoal is None:
                # No feasible subgoal found or agent is stuck, use random action
                print(f"  🎲 Agent {ctx.idx} performing RANDOM ACTION (no feasible subgoals or stuck)")
                self._reset_stuck_counter(ctx.idx)
                return self._get_random_action(ctx, state, mdp)
            else:
                print(f"  ⏸️  Agent {ctx.idx} STAYING (no valid target)")
                return Action.ACTION_TO_INDEX[Action.STAY]
        
        # Check if target is achievable (not blocked by other agent or object)
        target_achievable = self._is_target_achievable(target, ctx, state, mdp)
        print(f"  Target Achievable: {target_achievable}")
        
        if not target_achievable:
            print(f"  🚧 Target {target} is blocked, trying random action")
            return self._get_random_action(ctx, state, mdp)
        
        # Compute single-step action towards target
        action = self._compute_next_action_towards_target(ctx, target, state, mdp)
        action_name = self._get_action_name(action)
        print(f"  🎯 Action: {action_name} (towards {target})")
        
        # Store action for next turn's subgoal achievement detection
        setattr(self, f'_prev_action_{ctx.idx}', action)
        
        return action

    def _get_random_action(self, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld) -> int:
        """Get a completely random action for the agent."""
        # All possible actions (0:NORTH, 1:SOUTH, 2:EAST, 3:WEST, 4:STAY, 5:INTERACT)
        all_actions = [0, 1, 2, 3, 4, 5]
        
        # Randomly select any action
        selected_action = self.rng.choice(all_actions)
        print(f"    🎲 Random action selected: {self._get_action_name(selected_action)}")
        return selected_action


    def _update_position_history(self, ctx: AgentContext) -> None:
        """Update position history for stuck detection and check for actual movement."""
        if ctx.idx not in self._position_history:
            self._position_history[ctx.idx] = []
        
        history = self._position_history[ctx.idx]
        
        # Check if agent actually moved based on previous action
        if len(history) > 0:
            prev_position = history[-1]
            prev_action = getattr(self, f'_prev_action_{ctx.idx}', None)
            
            # Check if agent made actual progress (position changed)
            if prev_action is not None:
                action_name = self._get_action_name(prev_action)
                
                if prev_position == ctx.position:
                    # Agent didn't move - keep stuck state regardless of action type
                    if action_name in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
                        print(f"  🚧 Agent {ctx.idx} was blocked from moving {action_name}")
                    elif action_name == 'STAY':
                        print(f"  ⏸️  Agent {ctx.idx} stayed in place")
                    else:  # INTERACT
                        print(f"  🔄 Agent {ctx.idx} interacted but didn't move")
                else:
                    # Agent actually moved - reset stuck counter
                    self._reset_stuck_counter(ctx.idx)
                    print(f"  ✅ Agent {ctx.idx} made progress - stuck counter reset")
        
        history.append(ctx.position)
        
        # Keep only recent history
        if len(history) > self._max_stuck_turns + 2:
            history.pop(0)

    def _is_agent_stuck(self, ctx: AgentContext) -> bool:
        """Check if agent has been stuck in same position for too long."""
        if ctx.idx not in self._position_history:
            return False
        
        history = self._position_history[ctx.idx]
        if len(history) < self._max_stuck_turns:
            return False
        
        # Check if last N positions are all the same
        recent_positions = history[-self._max_stuck_turns:]
        return all(pos == ctx.position for pos in recent_positions)

    def _reset_stuck_counter(self, agent_idx: int) -> None:
        """Reset stuck counter and position history when agent makes progress."""
        self._stuck_counter[agent_idx] = 0
        # Also clear position history to prevent false stuck detection
        if agent_idx in self._position_history:
            self._position_history[agent_idx].clear()

    def _is_target_achievable(self, target: Tuple[int, int], ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld) -> bool:
        """Check if target position is valid (don't check for agent blocking)."""
        # Check if target is valid player position
        if target not in mdp.get_valid_player_positions():
            return False
        
        return True

    def _compute_next_action_towards_target(self, ctx: AgentContext, target: Tuple[int, int], state: OvercookedState, mdp: OvercookedGridworld) -> int:
        """Compute single next action towards target."""
        if ctx.position == target:
            # At target, check if we can interact
            current_subgoal = getattr(self, f'_prev_subgoal_{ctx.idx}', None)
            obj_pos = self._infer_object_for_adjacent(target, ctx, state, mdp, current_subgoal)
            if obj_pos:
                print(f"    🎯 At target {target}, found object at {obj_pos}")
                
                # Get current player state to check orientation
                player = state.players[ctx.idx]
                current_orientation = player.orientation
                print(f"    🧭 Current orientation: {current_orientation}")
                
                # Calculate required facing direction
                dx = obj_pos[0] - ctx.position[0]
                dy = obj_pos[1] - ctx.position[1]
                
                required_direction = None
                if dx == 1 and dy == 0:
                    required_direction = Direction.EAST
                elif dx == -1 and dy == 0:
                    required_direction = Direction.WEST
                elif dx == 0 and dy == 1:
                    required_direction = Direction.SOUTH
                elif dx == 0 and dy == -1:
                    required_direction = Direction.NORTH
                
                if required_direction:
                    print(f"    🎯 Required direction: {required_direction}")
                    
                    # Check if we're already facing the right direction
                    if current_orientation == required_direction:
                        print(f"    ✅ Already facing correct direction, interacting!")
                        return Action.ACTION_TO_INDEX[Action.INTERACT]
                    else:
                        print(f"    🔄 Need to turn from {current_orientation} to {required_direction}")
                        return Action.ACTION_TO_INDEX[required_direction]
                else:
                    print(f"    ❌ Invalid object position relative to agent: dx={dx}, dy={dy}")
                    return Action.ACTION_TO_INDEX[Action.STAY]
            else:
                print(f"    🎯 At target {target}, no object to interact with")
                return Action.ACTION_TO_INDEX[Action.STAY]
        
        # Find next step towards target using simple pathfinding
        path = self._shortest_path(ctx.position, target, mdp)
        print(f"    🗺️  Path to {target}: {path[:5]}{'...' if len(path) > 5 else ''} (length: {len(path)})")
        
        if len(path) < 2:
            print(f"    ❌ No path found to {target}")
            return Action.ACTION_TO_INDEX[Action.STAY]
        
        # Get next position in path
        next_pos = path[1]
        dx = next_pos[0] - ctx.position[0]
        dy = next_pos[1] - ctx.position[1]
        
        print(f"    ➡️  Moving from {ctx.position} to {next_pos}")
        
        # Convert to action
        if dx == 1 and dy == 0:
            return Action.ACTION_TO_INDEX[Direction.EAST]
        elif dx == -1 and dy == 0:
            return Action.ACTION_TO_INDEX[Direction.WEST]
        elif dx == 0 and dy == 1:
            return Action.ACTION_TO_INDEX[Direction.SOUTH]
        elif dx == 0 and dy == -1:
            return Action.ACTION_TO_INDEX[Direction.NORTH]
        else:
            print(f"    ❌ Invalid movement direction: dx={dx}, dy={dy}")
            return Action.ACTION_TO_INDEX[Action.STAY]

    def _feasible_subgoals(self, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld) -> List[str]:
        """
        Apply role-based filtering, then holding-based filtering, then pot state filtering 
        to determine feasible subgoals for this agent.
        """
        # Step 1: Role-based filtering
        possible_roles = self._get_agent_possible_roles(ctx)
        role_based_candidates = self._get_subgoals_for_roles(possible_roles)

        print(f"    🔍 Debug - role_based_candidates: {role_based_candidates}")

        if not role_based_candidates:
            return []
        
        # Step 2: Holding-based filtering
        can_with = {
            "onion": ctx.holding == "onion",
            "dish": ctx.holding == "dish", 
            "soup": ctx.holding == "soup",
            "none": ctx.holding is None,
        }
        
        holding_filtered = []
        for g in role_based_candidates:
            # Pickup actions require empty hands (except pickup_onion_soup which needs dish)
            if g in ("pickup_onion_from_dispenser", "pickup_onion_from_counter",
                    "pickup_dish_from_dispenser", "pickup_dish_from_counter",
                    "pickup_soup_from_counter") and not can_with["none"]:
                continue

            # Place/put actions require specific items
            if g in ("place_onion_on_counter", "put_onion_in_pot") and not can_with["onion"]:
                continue
            if g in ("place_dish_on_counter",) and not can_with["dish"]:
                continue
            if g in ("place_soup_on_counter",) and not can_with["soup"]:
                continue

            # Clear counter actions require specific items and capabilities
            if g == "clear_counter_for_dish" and (not can_with["dish"]):
                continue
            if g == "clear_counter_for_onion" and (not can_with["onion"]):
                continue

            # Pickup onion soup from pot requires dish in hand
            if g in ("pickup_onion_soup",) and not can_with["dish"]:
                continue

            # Delivery requires soup in hand
            if g in ("deliver_to_serve",) and not can_with["soup"]:
                continue

            holding_filtered.append(g)
        
        # Step 3: Pot state filtering
        pots_info = self._get_pots_info(state, mdp)
        any_pot_cooking_or_done = any(p["cooking_or_done"] for p in pots_info)
        
        # Check for ready pots specifically
        pot_states = mdp.get_pot_states(state)
        ready_pots = mdp.get_ready_pots(pot_states)
        has_ready_pots = len(ready_pots) > 0
        
        pot_state_filtered = []
        print(f"    🔍 Debug - holding_filtered: {holding_filtered}")
        for g in holding_filtered:
            # Dish-related actions are high priority when ready pots exist
            if g in ("pickup_dish_from_dispenser", "pickup_dish_from_counter"):
                if has_ready_pots or any_pot_cooking_or_done:
                    pot_state_filtered.append(g)
                    print(f"    ➕ Added {g} to pot_state_filtered (ready/cooking pots exist)")
                continue
            
            # Soup-related actions only make sense if there are pots cooking/ready
            if g in ("pickup_onion_soup", "pickup_soup_from_counter") and not any_pot_cooking_or_done:
                continue
                
            pot_state_filtered.append(g)
        
        print(f"    🔍 Debug - pot_state_filtered: {pot_state_filtered}")
        
        # Step 4: Role-based situational priority adjustment
        caps = ctx.capabilities or {}
        possible_roles = self._get_agent_possible_roles(ctx)
        final_filtered = []
        
        # Get cooking pot information
        cooking_pots = mdp.get_cooking_pots(pot_states)
        has_cooking_pots = len(cooking_pots) > 0
        
        print(f"    🎭 Step 4: Role-based priority adjustment for Agent {ctx.idx}")
        print(f"    📋 Possible roles: {possible_roles}")
        print(f"    🤲 Holding: {ctx.holding}")
        print(f"    🍲 Ready pots: {has_ready_pots}, Cooking pots: {has_cooking_pots}")
        
        # === READY POT SITUATION: Dish is critical ===
        if has_ready_pots:
            print(f"    🚨 READY POT SITUATION - Dish priority mode")
            
            # Case 1: Empty hands - need to get dish ASAP
            if ctx.holding is None:
                # Role 4 (Direct dish supplier + soup picker): dish_disp + pot
                if 4 in possible_roles:
                    print(f"    👑 Role 4 detected - Direct dish supplier priority")
                    # Highest priority: get dish from dispenser
                    if "pickup_dish_from_dispenser" in pot_state_filtered and "pickup_dish_from_dispenser" not in final_filtered:
                        final_filtered.append("pickup_dish_from_dispenser")
                        print(f"    ➕ Added pickup_dish_from_dispenser (Role 4 priority)")
                    # Also allow counter pickup as backup
                    if "pickup_dish_from_counter" in pot_state_filtered and "pickup_dish_from_counter" not in final_filtered:
                        final_filtered.append("pickup_dish_from_counter")
                        print(f"    ➕ Added pickup_dish_from_counter (Role 4 backup)")
                
                # Role 5 (Dish supplier via sharing): dish_disp + no pot
                elif 5 in possible_roles:
                    print(f"    🤝 Role 5 detected - Dish sharing supplier priority")
                    # Need to clear shared counter if full of onions
                    if "clear_counter_for_dish" in pot_state_filtered and "clear_counter_for_dish" not in final_filtered:
                        final_filtered.append("clear_counter_for_dish")
                        print(f"    ➕ Added clear_counter_for_dish (Role 5 counter clearing)")
                    # Get dish from dispenser
                    if "pickup_dish_from_dispenser" in pot_state_filtered and "pickup_dish_from_dispenser" not in final_filtered:
                        final_filtered.append("pickup_dish_from_dispenser")
                        print(f"    ➕ Added pickup_dish_from_dispenser (Role 5 priority)")
                
                # Role 6 (Dish receiver + soup picker): no dish_disp + pot
                elif 6 in possible_roles:
                    print(f"    📥 Role 6 detected - Dish receiver priority")
                    # Must get dish from shared counter
                    if "pickup_dish_from_counter" in pot_state_filtered and "pickup_dish_from_counter" not in final_filtered:
                        final_filtered.append("pickup_dish_from_counter")
                        print(f"    ➕ Added pickup_dish_from_counter (Role 6 priority)")
            
            # Case 2: Holding onion - need to clear hands for dish
            elif ctx.holding == "onion":
                print(f"    🧅 Holding onion in ready pot situation - need to clear hands")
                # Place onion on any available counter (not necessarily shared)
                if "place_onion_on_counter" in pot_state_filtered and "place_onion_on_counter" not in final_filtered:
                    final_filtered.append("place_onion_on_counter")
                    print(f"    ➕ Added place_onion_on_counter (clear hands for dish)")
                # Also add counter clearing if needed
                if "clear_counter_for_dish" in pot_state_filtered and "clear_counter_for_dish" not in final_filtered:
                    final_filtered.append("clear_counter_for_dish")
                    print(f"    ➕ Added clear_counter_for_dish (prepare for dish)")
        
        # === EMPTY POT SITUATION: Onion is critical ===
        empty_pots = mdp.get_empty_pots(pot_states)
        all_pots_empty = len(empty_pots) == len(mdp.get_pot_locations())
        
        if all_pots_empty:
            print(f"    🍲 ALL POTS EMPTY SITUATION - Onion priority mode")
            
            # Case 1: Empty hands - need to get onion ASAP
            if ctx.holding is None:
                # Role 1 (Direct onion supplier): onion_disp + pot
                if 1 in possible_roles:
                    print(f"    👑 Role 1 detected - Direct onion supplier priority")
                    # Highest priority: get onion and put in pot
                    if "pickup_onion_from_dispenser" in pot_state_filtered and "pickup_onion_from_dispenser" not in final_filtered:
                        final_filtered.append("pickup_onion_from_dispenser")
                        print(f"    ➕ Added pickup_onion_from_dispenser (Role 1 priority)")
                    if "pickup_onion_from_counter" in pot_state_filtered and "pickup_onion_from_counter" not in final_filtered:
                        final_filtered.append("pickup_onion_from_counter")
                        print(f"    ➕ Added pickup_onion_from_counter (Role 1 backup)")
                
                # Role 2 (Onion supplier via sharing): onion_disp + no pot
                elif 2 in possible_roles:
                    print(f"    🤝 Role 2 detected - Onion sharing supplier priority")
                    # Need to clear shared counter if full of dishes
                    if "clear_counter_for_onion" in pot_state_filtered and "clear_counter_for_onion" not in final_filtered:
                        final_filtered.append("clear_counter_for_onion")
                        print(f"    ➕ Added clear_counter_for_onion (Role 2 counter clearing)")
                    # Get onion from dispenser
                    if "pickup_onion_from_dispenser" in pot_state_filtered and "pickup_onion_from_dispenser" not in final_filtered:
                        final_filtered.append("pickup_onion_from_dispenser")
                        print(f"    ➕ Added pickup_onion_from_dispenser (Role 2 priority)")
                
                # Role 3 (Onion receiver + cooker): no onion_disp + pot
                elif 3 in possible_roles:
                    print(f"    📥 Role 3 detected - Onion receiver priority")
                    # Must get onion from shared counter
                    if "pickup_onion_from_counter" in pot_state_filtered and "pickup_onion_from_counter" not in final_filtered:
                        final_filtered.append("pickup_onion_from_counter")
                        print(f"    ➕ Added pickup_onion_from_counter (Role 3 priority)")
            
            # Case 2: Holding onion - HIGHEST PRIORITY: put in pot!
            elif ctx.holding == "onion":
                print(f"    🧅 Holding onion in empty pot situation - PUT IN POT IMMEDIATELY!")
                # Highest priority: put onion in pot if agent has pot capability
                if caps.get("pot", False):
                    if "put_onion_in_pot" in pot_state_filtered and "put_onion_in_pot" not in final_filtered:
                        final_filtered.append("put_onion_in_pot")
                        print(f"    ➕ Added put_onion_in_pot (HIGHEST PRIORITY - start cooking!)")
                # Fallback: place on counter if can't access pot
                else:
                    if "place_onion_on_counter" in pot_state_filtered and "place_onion_on_counter" not in final_filtered:
                        final_filtered.append("place_onion_on_counter")
                        print(f"    ➕ Added place_onion_on_counter (no pot access - share onion)")
            
            # Case 3: Holding dish - need to clear hands for onion
            elif ctx.holding == "dish":
                print(f"    🍽️ Holding dish in empty pot situation - need to clear hands")
                # Clear counter for onion if agent has onion_disp capability
                if caps.get("onion_disp", False):
                    if "clear_counter_for_onion" in pot_state_filtered and "clear_counter_for_onion" not in final_filtered:
                        final_filtered.append("clear_counter_for_onion")
                        print(f"    ➕ Added clear_counter_for_onion (clear hands for onion)")
                # Place dish on any available counter
                if "place_dish_on_counter" in pot_state_filtered and "place_dish_on_counter" not in final_filtered:
                    final_filtered.append("place_dish_on_counter")
                    print(f"    ➕ Added place_dish_on_counter (clear hands for onion)")
        
        # Add remaining subgoals with basic filtering
        for g in pot_state_filtered:
            # Don't place soup on counter if agent can serve directly
            if g == "place_soup_on_counter" and caps.get("serve", False):
                continue
            
            # Add if not already added by priority logic
            if g not in final_filtered:
                final_filtered.append(g)
        
        # Step 5: Additional situational adjustments (soup delivery priority)
        if ctx.holding == "soup" and caps.get("serve", False):
            if "deliver_to_serve" not in final_filtered:
                final_filtered.append("deliver_to_serve")
                print(f"    ➕ Added deliver_to_serve (holding soup + serve capability)")
        
        # Step 6: Final validation and cleanup
        print(f"    📋 Before priority sorting: {final_filtered}")
        
        # Step 7: Sort by priority (higher priority first)
        final_filtered.sort(key=lambda x: self.SUBGOAL_PRIORITY.get(x, 0), reverse=True)
        print(f"    ✅ Final feasible subgoals (priority sorted): {final_filtered}")
        
        return final_filtered

    def _select_target_for(self, ctx: AgentContext, feasible: List[str], state: OvercookedState, mdp: OvercookedGridworld, is_stuck: bool = False) -> Tuple[int, int]:
        """Select a concrete target location from feasible subgoals."""
        
        # feasible 리스트는 이미 우선순위가 적용된 상태
        for subgoal in feasible:
            target = self._target_for(subgoal, ctx, state, mdp)
            if target is not None:
                print(f"    ✅ Selected '{subgoal}' → Target: {target}")
                setattr(self, f'_prev_subgoal_{ctx.idx}', subgoal)
                return target
        
        print(f"    ⏸️  No valid target found")
        return None


    def _get_action_name(self, action_idx: int) -> str:
        """Convert action index to readable name."""
        action_names = {
            0: "NORTH",
            1: "SOUTH", 
            2: "EAST",
            3: "WEST",
            4: "STAY",
            5: "INTERACT"
        }
        return action_names.get(action_idx, f"UNKNOWN({action_idx})")

    def _print_environment_state(self, state: OvercookedState, mdp: OvercookedGridworld) -> None:
        """Print current environment state for debugging."""
        print(f"\n🌍 ENVIRONMENT STATE:")
        
        # Layout information
        layout_name = getattr(mdp, "layout_name", "unknown_layout")
        print(f"  Layout: {layout_name}")
        
        # Static object locations
        print(f"  🏭 Static Objects:")
        print(f"    Onion Dispensers: {list(mdp.get_onion_dispenser_locations())}")
        print(f"    Dish Dispensers: {list(mdp.get_dish_dispenser_locations())}")
        print(f"    Pot Locations: {list(mdp.get_pot_locations())}")
        print(f"    Serving Locations: {list(mdp.get_serving_locations())}")
        
        # Detailed pot states
        print(f"  🍲 Detailed Pot States:")
        pot_locations = mdp.get_pot_locations()
        for pot_pos in pot_locations:
            if state.has_object(pot_pos):
                soup = state.get_object(pot_pos)
                ingredients = getattr(soup, "ingredients", [])
                is_cooking = getattr(soup, "is_cooking", False)
                is_ready = getattr(soup, "is_ready", False)
                
                # Determine soup state and get appropriate information
                num_ingredients = len(ingredients)
                
                if is_ready:
                    # Ready soup - no cook time needed
                    status = "READY"
                    print(f"    Pot at {pot_pos}: {status} (ingredients: {num_ingredients})")
                elif is_cooking:
                    # Cooking soup - has cook time
                    status = "COOKING"
                    cook_time = getattr(soup, "cook_time", 0)
                    print(f"    Pot at {pot_pos}: {status} (ingredients: {num_ingredients}, cook_time: {cook_time})")
                else:
                    # Partial soup (1-2 ingredients) - no cook time
                    status = f"PARTIAL ({num_ingredients} ingredients)"
                    print(f"    Pot at {pot_pos}: {status} (ingredients: {num_ingredients})")
            else:
                print(f"    Pot at {pot_pos}: EMPTY")
        
        # Summary
        pot_states = mdp.get_pot_states(state)
        ready_pots = mdp.get_ready_pots(pot_states)
        cooking_pots = mdp.get_cooking_pots(pot_states)
        empty_pots = mdp.get_empty_pots(pot_states)
        partially_full_pots = mdp.get_partially_full_pots(pot_states)
        
        print(f"  📊 Pot Summary - Ready: {len(ready_pots)}, Cooking: {len(cooking_pots)}, Partial: {len(partially_full_pots)}, Empty: {len(empty_pots)}")
            
        # Counter objects
        counter_objects = mdp.get_counter_objects_dict(state)
        if counter_objects:
            print(f"  📦 Counter Objects:")
            for obj_type, positions in counter_objects.items():
                if positions:
                    print(f"    {obj_type}: {positions}")
        
        # Player positions and holdings
        print(f"  👥 Players:")
        for i, player in enumerate(state.players):
            holding = "None"
            if player.has_object():
                holding = getattr(player.get_object(), "name", "unknown")
            print(f"    Agent {i}: pos={tuple(player.position)}, holding={holding}, orientation={player.orientation}")

    def _find_any_adjacent_empty_position(self, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld, 
                                         all_agent_plans: Optional[List[Dict]] = None) -> Tuple[int, int]:
        """
        Stuck 상황에서 벗어나기 위한 안전한 인접 위치 찾기
        
        Args:
            ctx: 현재 에이전트 컨텍스트
            state: 게임 상태
            mdp: 게임 맵 정보
            all_agent_plans: 모든 에이전트의 계획 (있으면 더 정확한 예측 가능)
        """
        return self._find_safe_position_with_next_step_prediction(ctx, state, mdp, all_agent_plans)
    
    def _find_safe_position_with_next_step_prediction(self, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld, 
                                                    all_agent_plans: Optional[List[Dict]] = None,
                                                    exclude_positions: Optional[Set[Tuple[int, int]]] = None) -> Tuple[int, int]:
        """
        다른 에이전트들의 다음 1스텝 위치를 정확히 예측하여 안전한 위치 찾기
        
        Args:
            ctx: 현재 에이전트의 컨텍스트
            state: 현재 게임 상태
            mdp: 게임 맵 정보
            all_agent_plans: 모든 에이전트의 계획 (있으면 사용, 없으면 예측)
            exclude_positions: 추가로 제외할 위치들 (선택사항)
            
        Returns:
            안전한 위치 (최악의 경우 현재 위치)
        """
        valid_positions = set(mdp.get_valid_player_positions())
        
        # 1. 현재 점유된 위치들
        current_occupied = {tuple(p.position) for p in state.players}
        
        # 2. 다른 에이전트들의 다음 스텝 위치 예측
        next_step_occupied = set()
        
        if all_agent_plans:
            # 실제 계획이 있으면 그것을 사용 (더 정확함)
            next_step_occupied = self._get_next_positions_from_plans(all_agent_plans, ctx.idx)
        else:
            # 계획이 없으면 예측 (fallback)
            next_step_occupied = self._predict_next_step_positions(state, mdp, ctx.idx)
        
        # 3. 추가 제외 위치들
        if exclude_positions:
            next_step_occupied.update(exclude_positions)
        
        # 4. 모든 위험 위치 통합 (현재 + 다음 스텝)
        all_dangerous_positions = current_occupied | next_step_occupied
        
        print(f"    🔍 Agent {ctx.idx} danger analysis:")
        print(f"      Current occupied: {current_occupied}")
        print(f"      Next step occupied: {next_step_occupied}")
        print(f"      Additional excludes: {exclude_positions or set()}")
        
        # 5. 현재 위치에서 인접한 위치들 확인
        x, y = ctx.position
        adjacent_positions = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
        
        # 6. 안전한 인접 위치 찾기
        for pos in adjacent_positions:
            if (pos in valid_positions and 
                pos not in all_dangerous_positions):
                print(f"    🛡️ Found safe adjacent position: {pos} for Agent {ctx.idx}")
                return pos
        
        # 7. 인접 위치가 모두 위험하면 현재 위치가 최선
        
        # 8. 최후의 수단: 현재 위치
        print(f"    ⚠️ No safe position found, staying at current position: {ctx.position} for Agent {ctx.idx}")
        return ctx.position
    
    def _get_next_positions_from_plans(self, all_agent_plans: List[Dict], exclude_agent_idx: int) -> Set[Tuple[int, int]]:
        """
        실제 에이전트 계획들로부터 다음 스텝 위치들을 추출
        
        Args:
            all_agent_plans: 모든 에이전트의 계획
            exclude_agent_idx: 제외할 에이전트 인덱스
            
        Returns:
            다른 에이전트들의 다음 스텝 위치들
        """
        next_positions = set()
        
        for plan in all_agent_plans:
            agent_idx = plan.get('agent_idx', -1)
            if agent_idx == exclude_agent_idx:
                continue
                
            ctx = plan.get('context')
            target = plan.get('target')
            
            if ctx and target:
                # 목표까지의 경로에서 다음 스텝 위치 계산
                # 간단한 방향 계산 (1스텝만 필요하므로)
                x, y = ctx.position
                tx, ty = target
                
                # 목표 방향으로 1스텝 이동
                if abs(tx - x) > abs(ty - y):
                    # x축 우선 이동
                    next_x = x + (1 if tx > x else -1)
                    next_positions.add((next_x, y))
                elif abs(ty - y) > 0:
                    # y축 이동
                    next_y = y + (1 if ty > y else -1)
                    next_positions.add((x, next_y))
                else:
                    # 이미 목표에 도달
                    next_positions.add(ctx.position)
            elif ctx:
                next_positions.add(ctx.position)  # 계획이 없으면 제자리
                
        return next_positions
    
    def _predict_next_step_positions(self, state: OvercookedState, mdp: OvercookedGridworld, exclude_agent_idx: int) -> Set[Tuple[int, int]]:
        """
        다른 에이전트들의 다음 스텝 위치를 예측 (fallback 방식)
        
        Args:
            state: 현재 게임 상태
            mdp: 게임 맵 정보
            exclude_agent_idx: 제외할 에이전트 인덱스
            
        Returns:
            예측된 다음 스텝 위치들
        """
        next_positions = set()
        
        for i, player in enumerate(state.players):
            if i == exclude_agent_idx:
                continue
                
            # 간단한 예측: 현재 위치 + 인접 위치들을 모두 위험으로 간주
            current_pos = tuple(player.position)
            next_positions.add(current_pos)  # 제자리 가능성
            
            # 인접 위치들도 가능성으로 추가 (보수적 접근)
            x, y = current_pos
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                adjacent_pos = (x + dx, y + dy)
                if adjacent_pos in mdp.get_valid_player_positions():
                    next_positions.add(adjacent_pos)
                    
        return next_positions
    
    
    
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """맨하탄 거리 계산"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # ---------- Targeting helpers (skeletons to be refined) ----------

    def _target_for(self, goal: str, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld) -> Optional[Tuple[int, int]]:
        """서브골에 따른 구체적인 타겟 위치 반환 - 리팩토링된 버전"""
        
        # 1. 디스펜서에서 픽업
        if goal == "pickup_onion_from_dispenser":
            return self._get_nearest_adjacent(ctx, self._get_onion_dispensers(mdp), mdp)
        if goal == "pickup_dish_from_dispenser":
            return self._get_nearest_adjacent(ctx, self._get_dish_dispensers(mdp), mdp)
        
        # 2. 카운터에서 픽업
        if goal == "pickup_onion_from_counter":
            return self._get_nearest_adjacent(ctx, self._get_counters_with_item(state, mdp, "onion"), mdp)
        if goal == "pickup_dish_from_counter":
            return self._handle_pickup_dish_from_counter(ctx, state, mdp)
        if goal == "pickup_soup_from_counter":
            return self._handle_pickup_soup_from_counter(ctx, state, mdp)
        
        # 3. 카운터에 놓기
        if goal == "place_onion_on_counter":
            return self._get_empty_shared_counter_target(ctx, state, mdp, "onion")
        if goal == "place_dish_on_counter":
            return self._get_empty_shared_counter_target(ctx, state, mdp, "dish")
        if goal == "place_soup_on_counter":
            return self._handle_place_soup_on_counter(ctx, state, mdp)
        
        # 4. 냄비 관련
        if goal == "put_onion_in_pot":
            return self._best_reachable_pot_for_onion(ctx, state, mdp)
        if goal == "pickup_onion_soup":
            return self._handle_pickup_onion_soup(ctx, state, mdp)
        
        # 5. 서빙 및 정리
        if goal == "deliver_to_serve":
            return self._get_nearest_adjacent(ctx, self._get_serve_tiles(mdp), mdp)
        if goal == "clear_counter_for_dish":
            return self._get_counter_clearing_target(ctx, state, mdp, self._get_layout_name(mdp), "dish")
        if goal == "clear_counter_for_onion":
            return self._get_counter_clearing_target(ctx, state, mdp, self._get_layout_name(mdp), "onion")
        
        return None
    
    # ---------- 리팩토링된 헬퍼 메소드들 ----------
    
    def _get_nearest_adjacent(self, ctx: AgentContext, targets: List[Tuple[int, int]], mdp: OvercookedGridworld) -> Optional[Tuple[int, int]]:
        """타겟들 중 가장 가까운 곳의 인접 위치 반환"""
        return self._nearest_reachable_adjacent(ctx, targets, mdp)
    
    def _get_layout_name(self, mdp: OvercookedGridworld) -> str:
        """레이아웃 이름 안전하게 가져오기"""
        return getattr(mdp, "layout_name", "unknown_layout")
    
    def _get_empty_shared_counter_target(self, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld, item_type: str) -> Optional[Tuple[int, int]]:
        """빈 공유 카운터 타겟 반환 (place_onion_on_counter, place_dish_on_counter 통합)"""
        layout_name = self._get_layout_name(mdp)
        shared_counters = self._get_all_shared_counters_for_pot_agents(ctx, layout_name, len(state.players))
        empty_shared_counters = [c for c in shared_counters if not self._is_occupied(state, c)]
        
        print(f"    📦 Empty shared counters for {item_type}: {empty_shared_counters}")
        
        if empty_shared_counters:
            target_counter = self._nearest_of(ctx.position, empty_shared_counters)
            print(f"    🎯 Selected counter for place_{item_type}_on_counter: {target_counter}")
            return self._get_nearest_adjacent(ctx, [target_counter], mdp)
        else:
            print(f"    ❌ No empty shared counter found for place_{item_type}_on_counter")
            return None
    
    def _handle_pickup_dish_from_counter(self, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld) -> Optional[Tuple[int, int]]:
        """접시 픽업 처리 - 복잡한 로직 포함"""
        dish_counters = self._get_counters_with_item(state, mdp, "dish")
        if dish_counters:
            return self._get_nearest_adjacent(ctx, dish_counters, mdp)
        
        # 접시가 없으면 ready pot 확인 후 대기
        pot_states = mdp.get_pot_states(state)
        ready_pots = mdp.get_ready_pots(pot_states)
        if len(ready_pots) > 0:
            shared_counter = self._get_shared_counter_for_pot_agents(ctx, state, mdp, self._get_layout_name(mdp))
            if shared_counter:
                print(f"    🔄 No dish on counter, waiting near shared counter: {shared_counter}")
                return self._get_nearest_adjacent(ctx, [shared_counter], mdp)
        return None
    
    def _handle_pickup_soup_from_counter(self, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld) -> Optional[Tuple[int, int]]:
        """수프 픽업 처리"""
        counter = self._get_soup_from_shared_counter(ctx, state, mdp, self._get_layout_name(mdp))
        return self._get_nearest_adjacent(ctx, [counter] if counter else [], mdp)
    
    def _handle_place_soup_on_counter(self, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld) -> Optional[Tuple[int, int]]:
        """수프 놓기 처리"""
        counter = self._get_shared_counter_for_serve_agents(ctx, state, mdp, self._get_layout_name(mdp))
        return self._get_nearest_adjacent(ctx, [counter] if counter else [], mdp)
    
    def _handle_pickup_onion_soup(self, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld) -> Optional[Tuple[int, int]]:
        """양파 수프 픽업 처리 - ready pot만"""
        pot_states = mdp.get_pot_states(state)
        ready_pots = list(mdp.get_ready_pots(pot_states))
        print(f"    🍲 Ready pots for soup pickup: {ready_pots}")
        return self._get_nearest_adjacent(ctx, ready_pots, mdp)

    def _infer_object_for_adjacent(self, adj: Tuple[int, int], ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld, subgoal: Optional[str] = None) -> Optional[Tuple[int, int]]:
        # Identify which object tile this adjacent tile is meant to interact with (the one adjacent to 'adj')
        x, y = adj
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        
        # Get all object locations
        stand = set(mdp.get_valid_player_positions())
        counter = set(mdp.get_counter_locations())
        pots = set(mdp.get_pot_locations())
        serve = set(mdp.get_serving_locations())
        dishd = set(mdp.get_dish_dispenser_locations())
        oniond = set(mdp.get_onion_dispenser_locations())
        
        # Determine target object type based on subgoal
        target_object_types = set()
        if subgoal == "pickup_onion_from_dispenser":
            target_object_types = {"onion_dispenser"}
        elif subgoal == "pickup_dish_from_dispenser":
            target_object_types = {"dish_dispenser"}
        elif subgoal == "pickup_onion_from_counter":
            target_object_types = {"counter"}  # Counter with onion
        elif subgoal == "pickup_dish_from_counter":
            target_object_types = {"counter"}  # Counter with dish
        elif subgoal == "pickup_soup_from_counter":
            target_object_types = {"counter"}  # Counter with soup
        elif subgoal in ["place_onion_on_counter", "place_dish_on_counter", "place_soup_on_counter"]:
            target_object_types = {"counter"}  # Empty counter
        elif subgoal in ["clear_counter_for_dish", "clear_counter_for_onion"]:
            target_object_types = {"counter"}  # Target counter for clearing
        elif subgoal == "put_onion_in_pot":
            target_object_types = {"pot"}
        elif subgoal == "pickup_onion_soup":
            target_object_types = {"pot"}  # Ready pot
        elif subgoal == "deliver_to_serve":
            target_object_types = {"serving"}
        else:
            # Fallback: use priority-based selection
            target_object_types = {"onion_dispenser", "dish_dispenser", "pot", "serving", "counter"}
        
        # print(f"    🎯 Subgoal: {subgoal}, looking for: {target_object_types}")
        
        # Special handling for counter clearing subgoals
        if subgoal in ["clear_counter_for_dish", "clear_counter_for_onion"]:
            # For counter clearing, we need to find the specific target counter
            # The target should have been calculated by the clearing target function
            # We need to find which counter the agent is supposed to interact with
            
            # Get the target counter from the clearing function
            layout_name = getattr(mdp, "layout_name", "unknown_layout") if hasattr(mdp, "layout_name") else "unknown_layout"
            
            if subgoal == "clear_counter_for_dish":
                # Find regular counters (non-shared) for placing current onion
                all_counters_set = set(mdp.get_counter_locations())
                pot_capable_agents = self._find_agents_with_pot_capability(layout_name, len(state.players))
                shared_counters_set = set()
                for pot_agent_idx in pot_capable_agents:
                    if pot_agent_idx != ctx.idx:
                        counters = self._get_shared_counters_with_agent(layout_name, ctx.idx, pot_agent_idx)
                        shared_counters_set.update(counters)
                regular_counters = all_counters_set - shared_counters_set
                
                # Find the nearest regular counter that's adjacent to current position
                for c in candidates:
                    if c in regular_counters:
                        return c
                        
            elif subgoal == "clear_counter_for_onion":
                # Find regular counters (non-shared) for placing current dish
                all_counters_set = set(mdp.get_counter_locations())
                pot_capable_agents = self._find_agents_with_pot_capability(layout_name, len(state.players))
                shared_counters_set = set()
                for pot_agent_idx in pot_capable_agents:
                    if pot_agent_idx != ctx.idx:
                        counters = self._get_shared_counters_with_agent(layout_name, ctx.idx, pot_agent_idx)
                        shared_counters_set.update(counters)
                regular_counters = all_counters_set - shared_counters_set
                
                # Find the nearest regular counter that's adjacent to current position
                for c in candidates:
                    if c in regular_counters:
                        return c
        
        # Find the first adjacent object that matches the subgoal
        for c in candidates:
            if c in stand:
                continue
            
            # Check if this candidate matches the target object type
            if "onion_dispenser" in target_object_types and c in oniond:
                return c
            elif "dish_dispenser" in target_object_types and c in dishd:
                return c
            elif "pot" in target_object_types and c in pots:
                return c
            elif "serving" in target_object_types and c in serve:
                return c
            elif "counter" in target_object_types and c in counter:
                return c
        
        return None

    def _build_motion_plan(self, ctx: AgentContext, adj_target: Tuple[int, int], obj_pos: Optional[Tuple[int, int]], do_interact: bool, mdp: OvercookedGridworld) -> List[int]:
        # Build BFS path on standable tiles
        path = self._shortest_path(ctx.position, adj_target, mdp)
        actions: List[int] = []
        for i in range(1, len(path)):
            prev = path[i - 1]
            cur = path[i]
            dx = cur[0] - prev[0]
            dy = cur[1] - prev[1]
            if dx == 1 and dy == 0:
                actions.append(Action.ACTION_TO_INDEX[Direction.EAST])
            elif dx == -1 and dy == 0:
                actions.append(Action.ACTION_TO_INDEX[Direction.WEST])
            elif dx == 0 and dy == 1:
                actions.append(Action.ACTION_TO_INDEX[Direction.SOUTH])
            elif dx == 0 and dy == -1:
                actions.append(Action.ACTION_TO_INDEX[Direction.NORTH])
        # Orient towards the object tile if available
        if obj_pos is not None and path:
            cx, cy = (path[-1] if len(path) > 0 else ctx.position)
            ox, oy = obj_pos
            dx = ox - cx
            dy = oy - cy
            orient_action = None
            if abs(dx) + abs(dy) == 1:
                if dx == 1:
                    orient_action = Action.ACTION_TO_INDEX[Direction.EAST]
                elif dx == -1:
                    orient_action = Action.ACTION_TO_INDEX[Direction.WEST]
                elif dy == 1:
                    orient_action = Action.ACTION_TO_INDEX[Direction.SOUTH]
                elif dy == -1:
                    orient_action = Action.ACTION_TO_INDEX[Direction.NORTH]
            if orient_action is not None:
                actions.append(orient_action)
        if do_interact:
            actions.append(Action.ACTION_TO_INDEX[Action.INTERACT])
        return actions

    def _shortest_path(self, start: Tuple[int, int], goal: Tuple[int, int], mdp: OvercookedGridworld) -> List[Tuple[int, int]]:
        if start == goal:
            return [start]
        stand = set(mdp.get_valid_player_positions())
        q: deque = deque([start])
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                nxt = (nx, ny)
                if nxt not in stand or nxt in parent:
                    continue
                parent[nxt] = (x, y)
                if nxt == goal:
                    # reconstruct
                    path = [nxt]
                    cur = nxt
                    while cur is not None:
                        cur = parent[cur]
                        if cur is not None:
                            path.append(cur)
                    return list(reversed(path))
                q.append(nxt)
        # no path; stay
        return [start]


    # ---------- Low-level movement ----------

    def _move_or_interact(self, src: Tuple[int, int], dst: Tuple[int, int], do_interact: bool) -> int:
        if src == dst:
            return Action.ACTION_TO_INDEX[Action.INTERACT] if do_interact else Action.ACTION_TO_INDEX[Action.STAY]
        sx, sy = src
        dx, dy = dst
        if abs(dx - sx) > abs(dy - sy):
            return Action.ACTION_TO_INDEX[Action.EAST] if dx > sx else Action.ACTION_TO_INDEX[Action.WEST]
        if dy != sy:
            return Action.ACTION_TO_INDEX[Action.SOUTH] if dy > sy else Action.ACTION_TO_INDEX[Action.NORTH]
        return Action.ACTION_TO_INDEX[Action.STAY]

    # ---------- Environment queries (stubs to refine) ----------

    def _get_onion_dispensers(self, mdp: OvercookedGridworld) -> List[Tuple[int, int]]:
        # OAI API
        try:
            return list(mdp.get_onion_dispenser_locations())
        except Exception:
            return []

    def _get_dish_dispensers(self, mdp: OvercookedGridworld) -> List[Tuple[int, int]]:
        try:
            return list(mdp.get_dish_dispenser_locations())
        except Exception:
            return []

    def _get_serve_tiles(self, mdp: OvercookedGridworld) -> List[Tuple[int, int]]:
        try:
            return list(mdp.get_serving_locations())
        except Exception:
            return []

    def _get_counters_with_item(self, state: OvercookedState, mdp: OvercookedGridworld, item_name: str) -> List[Tuple[int, int]]:
        # OAI API: positions of objects on counters grouped by name
        try:
            by_type = mdp.get_counter_objects_dict(state)
            return list(by_type.get(item_name, []))
        except Exception:
            return []



    def _find_escape_position(self, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld, 
                            winner_next_pos: Tuple[int, int], all_agent_plans: Optional[List[Dict]] = None) -> Optional[Tuple[int, int]]:
        """
        데드락 상황에서 에이전트가 회피할 수 있는 안전한 위치를 찾습니다.
        
        Args:
            ctx: 현재 에이전트(loser)의 컨텍스트
            state: 현재 게임 상태
            mdp: 게임 맵 정보
            winner_next_pos: 우선순위가 높은 에이전트(winner)가 다음 턴에 이동할 위치
            all_agent_plans: 모든 에이전트의 계획 (더 정확한 예측을 위해)
            
        Returns:
            회피 가능한 위치 (없으면 None)
        """
        # winner의 다음 위치를 추가 제외 위치로 설정
        exclude_positions = {winner_next_pos}
        
        # 정확한 다음 스텝 예측 시스템 사용
        safe_position = self._find_safe_position_with_next_step_prediction(
            ctx, state, mdp, 
            all_agent_plans=all_agent_plans,
            exclude_positions=exclude_positions
        )
        
        # 현재 위치와 다른 위치만 반환 (실제로 이동해야 함)
        if safe_position != ctx.position:
            return safe_position
        else:
            return None

    def _resolve_movement_deadlocks(self, agent_plans: List[Dict], state: OvercookedState, mdp: OvercookedGridworld) -> List[Dict]:
        """
        모든 에이전트의 실제 계획된 액션을 기반으로 포괄적인 충돌을 감지하고 해결합니다.
        단순한 2명 위치 교환뿐만 아니라 다중 에이전트 순환, 동일 목표 충돌 등을 모두 처리합니다.
        """
        if len(agent_plans) < 2:
            return agent_plans

        print(f"\n🔍 포괄적 충돌 감지 시작 (총 {len(agent_plans)}명 에이전트)")
        
        # 1. 모든 에이전트의 다음 위치 및 액션 계산
        movement_info = self._calculate_all_agent_movements(agent_plans, mdp)
        
        # 2. 모든 종류의 충돌 감지
        conflicts = self._detect_all_conflicts(movement_info, agent_plans)
        
        # 3. 충돌 해결
        if conflicts:
            print(f"🚨 총 {len(conflicts)}개의 충돌 감지됨")
            resolved_plans = self._resolve_all_conflicts(conflicts, agent_plans, state, mdp)
            return resolved_plans
        else:
            print("✅ 충돌 없음 - 모든 에이전트가 안전하게 이동 가능")
            return agent_plans
    
    def _calculate_all_agent_movements(self, agent_plans: List[Dict], mdp: OvercookedGridworld) -> Dict[int, Dict]:
        """
        모든 에이전트의 이동 정보를 계산합니다.
        
        Returns:
            Dict[agent_idx, {
                'current_pos': Tuple[int, int],
                'target_pos': Tuple[int, int], 
                'next_pos': Tuple[int, int],
                'planned_action': str,
                'subgoal': str,
                'priority': int
            }]
        """
        movement_info = {}
        
        for plan in agent_plans:
            ctx = plan['context']
            target = plan['target']
            subgoal = plan['subgoal']
            
            # 다음 위치 계산
            if target is None or ctx.position == target:
                next_pos = ctx.position
                planned_action = "STAY"
            else:
                # 실제 다음 스텝 계산 (간단한 방향 기반)
                x, y = ctx.position
                tx, ty = target
                
                if abs(tx - x) > abs(ty - y):
                    next_pos = (x + (1 if tx > x else -1), y)
                    planned_action = "EAST" if tx > x else "WEST"
                elif abs(ty - y) > 0:
                    next_pos = (x, y + (1 if ty > y else -1))
                    planned_action = "SOUTH" if ty > y else "NORTH"
                else:
                    next_pos = ctx.position
                    planned_action = "INTERACT"
            
            movement_info[ctx.idx] = {
                'current_pos': ctx.position,
                'target_pos': target,
                'next_pos': next_pos,
                'planned_action': planned_action,
                'subgoal': subgoal,
                'priority': self.SUBGOAL_PRIORITY.get(subgoal, 0)
            }
            
            print(f"  Agent {ctx.idx}: {ctx.position} → {next_pos} ({planned_action}) [목표: {target}]")
        
        return movement_info
    
    def _detect_all_conflicts(self, movement_info: Dict[int, Dict], agent_plans: List[Dict]) -> List[Dict]:
        """
        모든 종류의 충돌을 감지합니다.
        
        Returns:
            List of conflict dictionaries with type and involved agents
        """
        conflicts = []
        agent_indices = list(movement_info.keys())
        
        # 1. 직접적인 위치 교환 (기존 방식)
        conflicts.extend(self._detect_position_swaps(movement_info, agent_indices))
        
        # 2. 동일 목표 지점 충돌
        conflicts.extend(self._detect_same_destination_conflicts(movement_info, agent_indices))
        
        # 3. 순환 데드락 (3명 이상)
        conflicts.extend(self._detect_circular_deadlocks(movement_info, agent_indices))
        
        # 4. 연쇄적 막힘
        conflicts.extend(self._detect_chain_blocks(movement_info, agent_indices))
        
        return conflicts
    
    def _detect_position_swaps(self, movement_info: Dict[int, Dict], agent_indices: List[int]) -> List[Dict]:
        """2명 에이전트의 직접적인 위치 교환 감지"""
        conflicts = []
        
        for i in range(len(agent_indices)):
            for j in range(i + 1, len(agent_indices)):
                agent_i = agent_indices[i]
                agent_j = agent_indices[j]
                
                info_i = movement_info[agent_i]
                info_j = movement_info[agent_j]
                
                # 서로의 위치로 이동하려는 경우
                if (info_i['next_pos'] == info_j['current_pos'] and 
                    info_j['next_pos'] == info_i['current_pos']):
                    
                    conflicts.append({
                        'type': 'position_swap',
                        'agents': [agent_i, agent_j],
                        'description': f"Agent {agent_i}와 {agent_j}가 서로의 위치로 이동하려 함"
                    })
                    print(f"  🔄 위치 교환 충돌: Agent {agent_i} ↔ Agent {agent_j}")
        
        return conflicts
    
    def _detect_same_destination_conflicts(self, movement_info: Dict[int, Dict], agent_indices: List[int]) -> List[Dict]:
        """동일한 목표 지점으로 이동하려는 충돌 감지"""
        conflicts = []
        destination_groups = {}
        
        # 목표 지점별로 에이전트 그룹화
        for agent_idx in agent_indices:
            info = movement_info[agent_idx]
            next_pos = info['next_pos']
            
            if next_pos not in destination_groups:
                destination_groups[next_pos] = []
            destination_groups[next_pos].append(agent_idx)
        
        # 2명 이상이 같은 곳으로 가려는 경우
        for dest_pos, agents in destination_groups.items():
            if len(agents) > 1:
                # 현재 그 위치에 있는 에이전트는 제외 (제자리)
                moving_agents = [a for a in agents if movement_info[a]['current_pos'] != dest_pos]
                
                if len(moving_agents) > 1:
                    conflicts.append({
                        'type': 'same_destination',
                        'agents': moving_agents,
                        'destination': dest_pos,
                        'description': f"{len(moving_agents)}명이 {dest_pos}로 동시 이동 시도"
                    })
                    print(f"  🎯 동일 목표 충돌: {moving_agents} → {dest_pos}")
        
        return conflicts
    
    def _detect_circular_deadlocks(self, movement_info: Dict[int, Dict], agent_indices: List[int]) -> List[Dict]:
        """3명 이상의 순환 데드락 감지 (A→B→C→A)"""
        conflicts = []
        
        # 간단한 순환 감지 (3-4명 정도까지)
        for start_agent in agent_indices:
            visited = set()
            current = start_agent
            path = []
            
            while current not in visited and current in movement_info:
                visited.add(current)
                path.append(current)
                
                # 현재 에이전트가 이동하려는 위치에 있는 다른 에이전트 찾기
                current_next_pos = movement_info[current]['next_pos']
                next_agent = None
                
                for other_agent in agent_indices:
                    if (other_agent != current and 
                        movement_info[other_agent]['current_pos'] == current_next_pos):
                        next_agent = other_agent
                        break
                
                if next_agent is None:
                    break
                    
                current = next_agent
                
                # 순환 감지
                if current == start_agent and len(path) >= 3:
                    conflicts.append({
                        'type': 'circular_deadlock',
                        'agents': path,
                        'description': f"순환 데드락: {' → '.join(map(str, path))} → {start_agent}"
                    })
                    print(f"  🔄 순환 데드락: {' → '.join(map(str, path))} → {start_agent}")
                    break
        
        return conflicts
    
    def _detect_chain_blocks(self, movement_info: Dict[int, Dict], agent_indices: List[int]) -> List[Dict]:
        """연쇄적 막힘 감지 (A가 B를 막고, B가 C를 막는 상황)"""
        conflicts = []
        
        # 각 에이전트가 누구를 막고 있는지 확인
        blocking_chains = {}
        
        for agent_idx in agent_indices:
            info = movement_info[agent_idx]
            current_pos = info['current_pos']
            
            # 이 에이전트의 현재 위치로 이동하려는 다른 에이전트들 찾기
            blocked_agents = []
            for other_agent in agent_indices:
                if (other_agent != agent_idx and 
                    movement_info[other_agent]['next_pos'] == current_pos and
                    movement_info[other_agent]['current_pos'] != current_pos):
                    blocked_agents.append(other_agent)
            
            if blocked_agents:
                blocking_chains[agent_idx] = blocked_agents
        
        # 연쇄 길이가 2 이상인 경우 충돌로 간주
        for blocker, blocked_list in blocking_chains.items():
            if len(blocked_list) >= 1:
                # 간단한 연쇄 감지 (더 복잡한 로직 필요시 확장 가능)
                for blocked in blocked_list:
                    if blocked in blocking_chains:
                        conflicts.append({
                            'type': 'chain_block',
                            'agents': [blocker, blocked],
                            'description': f"Agent {blocker}가 Agent {blocked}를 막음"
                        })
                        print(f"  ⛓️ 연쇄 막힘: Agent {blocker} → Agent {blocked}")
        
        return conflicts
    
    def _resolve_all_conflicts(self, conflicts: List[Dict], agent_plans: List[Dict], 
                             state: OvercookedState, mdp: OvercookedGridworld) -> List[Dict]:
        """
        감지된 모든 충돌을 우선순위 기반으로 해결합니다.
        """
        resolved_plans = agent_plans.copy()
        
        # 충돌을 심각도 순으로 정렬 (순환 데드락 > 위치 교환 > 동일 목표 > 연쇄 막힘)
        conflict_priority = {
            'circular_deadlock': 4,
            'position_swap': 3,
            'same_destination': 2,
            'chain_block': 1
        }
        
        sorted_conflicts = sorted(conflicts, key=lambda c: conflict_priority.get(c['type'], 0), reverse=True)
        
        for conflict in sorted_conflicts:
            print(f"\n🔧 충돌 해결 중: {conflict['description']}")
            
            if conflict['type'] == 'position_swap':
                resolved_plans = self._resolve_position_swap(conflict, resolved_plans, state, mdp)
            elif conflict['type'] == 'same_destination':
                resolved_plans = self._resolve_same_destination(conflict, resolved_plans, state, mdp)
            elif conflict['type'] == 'circular_deadlock':
                resolved_plans = self._resolve_circular_deadlock(conflict, resolved_plans, state, mdp)
            elif conflict['type'] == 'chain_block':
                resolved_plans = self._resolve_chain_block(conflict, resolved_plans, state, mdp)
        
        return resolved_plans
    
    def _resolve_position_swap(self, conflict: Dict, agent_plans: List[Dict], 
                             state: OvercookedState, mdp: OvercookedGridworld) -> List[Dict]:
        """2명 에이전트의 위치 교환 충돌 해결"""
        agent_i, agent_j = conflict['agents']
        
        # 해당 에이전트들의 계획 찾기
        plan_i = next(p for p in agent_plans if p['context'].idx == agent_i)
        plan_j = next(p for p in agent_plans if p['context'].idx == agent_j)
        
        # 우선순위 비교
        priority_i = self.SUBGOAL_PRIORITY.get(plan_i['subgoal'], 0)
        priority_j = self.SUBGOAL_PRIORITY.get(plan_j['subgoal'], 0)
        
        # 우선순위가 낮은 에이전트가 회피
        if priority_i < priority_j:
            loser_plan, winner_plan = plan_i, plan_j
        elif priority_j < priority_i:
            loser_plan, winner_plan = plan_j, plan_i
        else:
            # 우선순위가 같으면 인덱스가 높은 쪽이 회피
            loser_plan = plan_j if agent_j > agent_i else plan_i
            winner_plan = plan_i if agent_j > agent_i else plan_j

        print(f"  → Agent {loser_plan['context'].idx}가 Agent {winner_plan['context'].idx}를 위해 회피")

        # 회피 위치 찾기
        escape_pos = self._find_escape_position(
            loser_plan['context'], state, mdp, 
            winner_plan['target'], agent_plans
        )
        
        if escape_pos:
            loser_plan['subgoal'] = "evade_deadlock"
            loser_plan['target'] = escape_pos
            print(f"  ✅ Agent {loser_plan['context'].idx} → {escape_pos}로 회피")
        else:
            loser_plan['subgoal'] = "yield_to_deadlock"
            loser_plan['target'] = loser_plan['context'].position
            print(f"  ⚠️ Agent {loser_plan['context'].idx} 제자리 대기")
        
        return agent_plans
    
    def _resolve_same_destination(self, conflict: Dict, agent_plans: List[Dict], 
                                state: OvercookedState, mdp: OvercookedGridworld) -> List[Dict]:
        """동일 목표 지점 충돌 해결"""
        conflicting_agents = conflict['agents']
        destination = conflict['destination']
        
        # 우선순위 순으로 정렬
        agent_priorities = []
        for agent_idx in conflicting_agents:
            plan = next(p for p in agent_plans if p['context'].idx == agent_idx)
            priority = self.SUBGOAL_PRIORITY.get(plan['subgoal'], 0)
            agent_priorities.append((agent_idx, priority, plan))
        
        agent_priorities.sort(key=lambda x: x[1], reverse=True)  # 높은 우선순위부터
        
        # 가장 높은 우선순위 에이전트만 목표 지점으로 이동, 나머지는 회피
        winner_agent, _, winner_plan = agent_priorities[0]
        print(f"  → Agent {winner_agent}가 {destination}에 우선 접근")
        
        for agent_idx, _, plan in agent_priorities[1:]:
            print(f"  → Agent {agent_idx}는 회피")
            
            # 회피 위치 찾기
            escape_pos = self._find_escape_position(
                plan['context'], state, mdp, destination, agent_plans
            )
            
            if escape_pos:
                plan['subgoal'] = "evade_deadlock"
                plan['target'] = escape_pos
                print(f"    ✅ Agent {agent_idx} → {escape_pos}로 회피")
            else:
                plan['subgoal'] = "yield_to_deadlock"
                plan['target'] = plan['context'].position
                print(f"    ⚠️ Agent {agent_idx} 제자리 대기")
        
        return agent_plans
    
    def _resolve_circular_deadlock(self, conflict: Dict, agent_plans: List[Dict], 
                                 state: OvercookedState, mdp: OvercookedGridworld) -> List[Dict]:
        """순환 데드락 해결 - 가장 우선순위가 낮은 에이전트가 순환을 끊음"""
        circular_agents = conflict['agents']
        
        # 순환에 참여하는 에이전트들의 우선순위 확인
        agent_priorities = []
        for agent_idx in circular_agents:
            plan = next(p for p in agent_plans if p['context'].idx == agent_idx)
            priority = self.SUBGOAL_PRIORITY.get(plan['subgoal'], 0)
            agent_priorities.append((agent_idx, priority, plan))
        
        # 가장 낮은 우선순위 에이전트가 순환을 끊음
        agent_priorities.sort(key=lambda x: x[1])  # 낮은 우선순위부터
        breaker_agent, _, breaker_plan = agent_priorities[0]
        
        print(f"  → Agent {breaker_agent}가 순환을 끊기 위해 회피")
        
        # 회피 위치 찾기
        escape_pos = self._find_escape_position(
            breaker_plan['context'], state, mdp, 
            breaker_plan['target'], agent_plans
        )
        
        if escape_pos:
            breaker_plan['subgoal'] = "evade_deadlock"
            breaker_plan['target'] = escape_pos
            print(f"  ✅ Agent {breaker_agent} → {escape_pos}로 회피하여 순환 해결")
        else:
            breaker_plan['subgoal'] = "yield_to_deadlock"
            breaker_plan['target'] = breaker_plan['context'].position
            print(f"  ⚠️ Agent {breaker_agent} 제자리 대기하여 순환 해결")
        
        return agent_plans
    
    def _resolve_chain_block(self, conflict: Dict, agent_plans: List[Dict], 
                           state: OvercookedState, mdp: OvercookedGridworld) -> List[Dict]:
        """연쇄적 막힘 해결"""
        blocker, blocked = conflict['agents']
        
        # 막는 에이전트와 막히는 에이전트의 우선순위 비교
        blocker_plan = next(p for p in agent_plans if p['context'].idx == blocker)
        blocked_plan = next(p for p in agent_plans if p['context'].idx == blocked)
        
        blocker_priority = self.SUBGOAL_PRIORITY.get(blocker_plan['subgoal'], 0)
        blocked_priority = self.SUBGOAL_PRIORITY.get(blocked_plan['subgoal'], 0)
        
        # 우선순위가 낮은 쪽이 회피
        if blocker_priority < blocked_priority:
            # 막는 에이전트가 우선순위가 낮으면 그가 회피
            evader_plan = blocker_plan
            print(f"  → Agent {blocker}가 Agent {blocked}를 위해 회피")
        else:
            # 막히는 에이전트가 우선순위가 낮거나 같으면 그가 회피
            evader_plan = blocked_plan
            print(f"  → Agent {blocked}가 다른 경로 탐색")
        
        # 회피 위치 찾기
        escape_pos = self._find_escape_position(
            evader_plan['context'], state, mdp, 
            evader_plan['target'], agent_plans
        )
        
        if escape_pos:
            evader_plan['subgoal'] = "evade_deadlock"
            evader_plan['target'] = escape_pos
            print(f"  ✅ Agent {evader_plan['context'].idx} → {escape_pos}로 회피")
        else:
            evader_plan['subgoal'] = "yield_to_deadlock"
            evader_plan['target'] = evader_plan['context'].position
            print(f"  ⚠️ Agent {evader_plan['context'].idx} 제자리 대기")

        return agent_plans


    def _get_pots_info(self, state: OvercookedState, mdp: OvercookedGridworld) -> List[Dict]:
        pots: List[Dict] = []
        for pos in mdp.get_pot_locations():
            if state.has_object(pos):
                soup = state.get_object(pos)
                is_cooking_or_done = getattr(soup, "is_cooking", False) or getattr(soup, "is_ready", False)
                onions = len(getattr(soup, "ingredients", []))
                pots.append({"pos": tuple(pos), "cooking_or_done": bool(is_cooking_or_done), "onions": onions})
            else:
                pots.append({"pos": tuple(pos), "cooking_or_done": False, "onions": 0})
        return pots

    def _is_occupied(self, state: OvercookedState, loc: Tuple[int, int]) -> bool:
        try:
            return state.has_object(loc)
        except Exception:
            return False

    def _nearest_of(self, src: Tuple[int, int], targets: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if not targets:
            return None
        return min(targets, key=lambda t: self._manhattan(src, t))

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ---------- Static reachability helpers ----------

    def _bfs_reachable(self, valid_tiles: Set[Tuple[int, int]], start: Tuple[int, int]) -> Set[Tuple[int, int]]:
        q: deque = deque([start])
        visited: Set[Tuple[int, int]] = {start}
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                nxt = (nx, ny)
                if nxt in valid_tiles and nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        return visited

    def _adjacent_stand_tiles(self, pos: Tuple[int, int], mdp: OvercookedGridworld) -> List[Tuple[int, int]]:
        stand = set(mdp.get_valid_player_positions())
        x, y = pos
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [p for p in candidates if p in stand]

    def _nearest_reachable_adjacent(self, ctx: AgentContext, obj_positions: List[Tuple[int, int]], mdp: OvercookedGridworld) -> Optional[Tuple[int, int]]:
        if not obj_positions:
            return None
        reachable = ctx.reachable or set()
        best: Optional[Tuple[int, int]] = None
        best_dist = 10**9
        
        for pos in obj_positions:
            # Check if we're already adjacent to this object
            adjacent_tiles = self._adjacent_stand_tiles(pos, mdp)
            if ctx.position in adjacent_tiles:
                # Already adjacent to this object - return current position for interaction
                return ctx.position
                
            # Otherwise find nearest reachable adjacent tile
            for adj in adjacent_tiles:
                if adj in reachable:
                    d = self._manhattan(ctx.position, adj)
                    if d < best_dist:
                        best_dist = d
                        best = adj
        return best

    def _compute_capabilities(self, reachable: Set[Tuple[int, int]], mdp: OvercookedGridworld) -> Dict[str, bool]:
        caps = {k: False for k in ("onion_disp", "dish_disp", "pot", "serve")}
        def any_adjacent_of(positions: List[Tuple[int, int]]) -> bool:
            for pos in positions:
                for adj in self._adjacent_stand_tiles(pos, mdp):
                    if adj in reachable:
                        return True
            return False
        caps["onion_disp"] = any_adjacent_of(mdp.get_onion_dispenser_locations())
        caps["dish_disp"] = any_adjacent_of(mdp.get_dish_dispenser_locations())
        caps["pot"] = any_adjacent_of(mdp.get_pot_locations())
        caps["serve"] = any_adjacent_of(mdp.get_serving_locations())
        return caps

    def _compute_shared_counters(self, agent_reachable_areas: Dict[int, Set[Tuple[int, int]]], mdp: OvercookedGridworld) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Compute shared counters between all pairs of agents.
        
        Args:
            agent_reachable_areas: {agent_idx: set_of_reachable_positions}
            mdp: OvercookedGridworld instance
            
        Returns:
            {(agent_i, agent_j): [list_of_shared_counter_positions]} where agent_i < agent_j (no duplicates)
        """
        shared_counters = {}
        counter_locations = mdp.get_counter_locations()
        num_agents = len(agent_reachable_areas)
        
        # For each pair of agents (only store upper triangle to avoid duplicates)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                agent_i_reachable = agent_reachable_areas[i]
                agent_j_reachable = agent_reachable_areas[j]
                
                shared_counter_positions = []
                
                # Check each counter location
                for counter_pos in counter_locations:
                    # Get adjacent standable tiles for this counter
                    adjacent_tiles = self._adjacent_stand_tiles(counter_pos, mdp)
                    
                    # Check if both agents can reach at least one adjacent tile
                    agent_i_can_reach = any(adj in agent_i_reachable for adj in adjacent_tiles)
                    agent_j_can_reach = any(adj in agent_j_reachable for adj in adjacent_tiles)
                    
                    if agent_i_can_reach and agent_j_can_reach:
                        shared_counter_positions.append(counter_pos)
                
                # Only store once: (smaller_idx, larger_idx)
                shared_counters[(i, j)] = shared_counter_positions
        
        return shared_counters

    def _get_shared_counters_with_agent(self, layout_name: str, current_agent_idx: int, other_agent_idx: int) -> List[Tuple[int, int]]:
        """
        Get counters that are shared between current_agent and other_agent.
        
        Args:
            layout_name: Current layout name
            current_agent_idx: Index of current agent
            other_agent_idx: Index of other agent
            
        Returns:
            List of shared counter positions
        """
        if layout_name not in self._shared_counters_cache:
            return []
        
        shared_counters = self._shared_counters_cache[layout_name]
        
        # Normalize the key to always have smaller index first
        if current_agent_idx < other_agent_idx:
            key = (current_agent_idx, other_agent_idx)
        else:
            key = (other_agent_idx, current_agent_idx)
            
        return shared_counters.get(key, [])

    def _find_agents_with_pot_capability(self, layout_name: str, num_agents: int) -> List[int]:
        """
        Find agents that can reach pots.
        
        Args:
            layout_name: Current layout name
            num_agents: Total number of agents
            
        Returns:
            List of agent indices that can reach pots
        """
        pot_capable_agents = []
        for i in range(num_agents):
            if layout_name in self._static_cache and i in self._static_cache[layout_name]:
                caps = self._static_cache[layout_name][i].get("capabilities", {})
                if caps.get("pot", False):
                    pot_capable_agents.append(i)
        return pot_capable_agents


    # ---------- New Strategy-based Counter Selection Methods ----------
    
    def _get_agent_possible_roles(self, ctx: AgentContext) -> List[int]:
        """
        Determine which roles this agent can perform based on capabilities only.
        An agent can have multiple roles simultaneously.
        
        Returns:
            List[int]: List of role numbers (1-9) this agent can perform
        """
        caps = ctx.capabilities or {}
        has_onion_disp = caps.get("onion_disp", False)
        has_dish_disp = caps.get("dish_disp", False)
        has_pot = caps.get("pot", False)
        has_serve = caps.get("serve", False)
        
        possible_roles = []
        
        # Role 1: Direct onion supplier (onion_disp: True, pot: True)
        if has_onion_disp and has_pot:
            possible_roles.append(1)
        
        # Role 2: Onion supplier via sharing (onion_disp: True, pot: False)
        if has_onion_disp and not has_pot:
            possible_roles.append(2)
        
        # Role 3: Onion receiver and cooker (onion_disp: False, pot: True)
        if not has_onion_disp and has_pot:
            possible_roles.append(3)
        
        # Role 4: Direct dish supplier and soup picker (dish_disp: True, pot: True)
        if has_dish_disp and has_pot:
            possible_roles.append(4)
        
        # Role 5: Dish supplier via sharing (dish_disp: True, pot: False)
        if has_dish_disp and not has_pot:
            possible_roles.append(5)
        
        # Role 6: Dish receiver and soup picker (dish_disp: False, pot: True)
        if not has_dish_disp and has_pot:
            possible_roles.append(6)
        
        # Role 7: Full service (pot: True, serve: True)
        if has_pot and has_serve:
            possible_roles.append(7)
        
        # Role 8: Soup maker and sharer (pot: True, serve: False)
        if has_pot and not has_serve:
            possible_roles.append(8)
        
        # Role 9: Soup receiver and deliverer (pot: False, serve: True)
        if not has_pot and has_serve:
            possible_roles.append(9)
        
        return possible_roles

    def _get_subgoals_for_roles(self, roles: List[int]) -> List[str]:
        """
        Get all possible subgoals for the given roles.
        
        Args:
            roles: List of role numbers
            
        Returns:
            List of subgoal names that these roles can perform
        """
        # Role to subgoals mapping (capability-based only, no pot status or delivery access checks)
        role_subgoals = {
            1: ["pickup_onion_from_dispenser", "put_onion_in_pot", "place_onion_on_counter", "clear_counter_for_onion", "clear_counter_for_dish"],  # Direct onion supplier
            2: ["pickup_onion_from_dispenser", "place_onion_on_counter", "clear_counter_for_onion", "clear_counter_for_dish"],  # Onion supplier via sharing (onion_disp + dish_disp)
            3: ["pickup_onion_from_counter", "put_onion_in_pot", "place_onion_on_counter"],  # Onion receiver and cooker
            4: ["pickup_dish_from_dispenser", "pickup_onion_soup", "place_dish_on_counter", "clear_counter_for_dish"],  # Direct soup picker (dish access certain)
            5: ["pickup_dish_from_dispenser", "place_dish_on_counter", "clear_counter_for_dish"],  # Dish supplier via sharing (dish access certain)
            6: ["pickup_dish_from_counter", "pickup_onion_soup", "place_dish_on_counter"],  # Dish receiver and soup picker
            7: ["pickup_onion_soup", "place_soup_on_counter", "deliver_to_serve"],  # Full service (dish/delivery access)
            8: ["pickup_onion_soup", "place_soup_on_counter"],  # Soup maker and sharer
            9: ["pickup_soup_from_counter", "deliver_to_serve"]  # Soup receiver (delivery access)
        }
        
        all_subgoals = set()
        for role in roles:
            if role in role_subgoals:
                all_subgoals.update(role_subgoals[role])
        
        return list(all_subgoals)

    def _execute_role_based_subgoal_selection(self, ctx: AgentContext, role: int, state: OvercookedState, mdp: OvercookedGridworld) -> Optional[str]:
        """
        Execute dynamic subgoal selection for a specific role based on holding state and pot status.
        
        Returns:
            Optional[str]: Selected subgoal name, or None if no suitable subgoal for this role
        """
        layout_name = getattr(mdp, "layout_name", "unknown_layout")
        pot_states = mdp.get_pot_states(state)
        ready_pots = mdp.get_ready_pots(pot_states)
        non_ready_pots = []
        for pot_pos in mdp.get_pot_locations():
            if pot_pos not in ready_pots:
                non_ready_pots.append(pot_pos)
        
        has_non_ready_pots = len(non_ready_pots) > 0
        has_ready_pots = len(ready_pots) > 0
        
        # Case 1: Direct onion supplier (onion_disp: True, pot: True)
        if role == 1:
            if ctx.holding == "dish":
                if all_pots_empty:
                    # All pots empty - check if we need to clear onions first
                    layout_name = getattr(mdp, "layout_name", "unknown_layout")
                    pot_capable_agents = self._find_agents_with_pot_capability(layout_name, len(state.players))
                    shared_counters = []
                    for pot_agent_idx in pot_capable_agents:
                        if pot_agent_idx != ctx.idx:
                            counters = self._get_shared_counters_with_agent(layout_name, ctx.idx, pot_agent_idx)
                            shared_counters.extend(counters)
                    shared_counters = list(set(shared_counters))

                    # Check what items are on shared counters
                    onion_counters = set(self._get_counters_with_item(state, mdp, "onion"))
                    onion_on_shared = sum(1 for c in shared_counters if c in onion_counters)
                    empty_shared = [c for c in shared_counters if not self._is_occupied(state, c)]

                    if onion_on_shared >= len(shared_counters) * 0.7:  # 70% 이상이 onion으로 차 있음
                        # Shared counters are full of onions - need to clear onions first
                        caps = ctx.capabilities or {}
                        if caps.get("dish_disp", False):
                            # Can clear onions (by placing dishes) - check if there are regular counters available
                            all_counters_set = set(mdp.get_counter_locations())
                            shared_counters_set = set(shared_counters)
                            regular_counters = all_counters_set - shared_counters_set
                            empty_regular_counters = [c for c in regular_counters if not self._is_occupied(state, c)]

                            if empty_regular_counters:
                                return "clear_counter_for_onion"  # Clear onion to make room for dish
                    # Clear dish to make room for onion (either counters not full or can't clear onions)
                    return "clear_counter_for_onion"  # Clear dish to make room for onion
                elif has_ready_pots:
                    return "pickup_onion_soup"  # Pick up soup from ready pot
                else:
                    return "place_dish_on_counter"  # No ready pots, place dish near pot
            elif ctx.holding == "onion":
                if has_non_ready_pots:
                    return "put_onion_in_pot"  # Put onion in non-ready pot
                else:
                    return "place_onion_on_counter"  # All pots ready, place onion near pot
            else:  # Not holding anything
                if has_non_ready_pots:
                    return "pickup_onion_from_dispenser"  # Get onion for cooking
                # All pots ready - low priority, let other subgoals handle
                return None
        
        # Case 2: Onion supplier via sharing (onion_disp: True, pot: False)
        elif role == 2:
            # Check if this agent also has dish_disp capability
            caps = ctx.capabilities or {}
            has_dish_disp = caps.get("dish_disp", False)
            
            # Check for empty pots
            empty_pots = mdp.get_empty_pots(pot_states)
            all_pots_empty = len(empty_pots) == len(mdp.get_pot_locations())
            
            if ctx.holding == "onion":
                if has_ready_pots and has_dish_disp:
                    # Ready pots exist and we can get dishes - check shared counter status
                    layout_name = getattr(mdp, "layout_name", "unknown_layout")
                    pot_capable_agents = self._find_agents_with_pot_capability(layout_name, len(state.players))
                    shared_counters = []
                    for pot_agent_idx in pot_capable_agents:
                        if pot_agent_idx != ctx.idx:
                            counters = self._get_shared_counters_with_agent(layout_name, ctx.idx, pot_agent_idx)
                            shared_counters.extend(counters)
                    shared_counters = list(set(shared_counters))

                    # Check what items are on shared counters
                    dish_counters = set(self._get_counters_with_item(state, mdp, "dish"))
                    onion_counters = set(self._get_counters_with_item(state, mdp, "onion"))
                    dish_on_shared = sum(1 for c in shared_counters if c in dish_counters)
                    onion_on_shared = sum(1 for c in shared_counters if c in onion_counters)
                    empty_shared = [c for c in shared_counters if not self._is_occupied(state, c)]

                    if dish_on_shared >= len(shared_counters) * 0.7:  # 70% 이상이 dish로 차 있음
                        # Shared counters are full of dishes - need to clear dishes first
                        caps = ctx.capabilities or {}
                        if caps.get("dish_disp", False):
                            # Can clear dishes - check if there are regular counters available
                            all_counters_set = set(mdp.get_counter_locations())
                            shared_counters_set = set(shared_counters)
                            regular_counters = all_counters_set - shared_counters_set
                            empty_regular_counters = [c for c in regular_counters if not self._is_occupied(state, c)]

                            if empty_regular_counters:
                                return "clear_counter_for_dish"  # Clear dish to make room for onion
                        # Can't clear dishes or no empty regular counters - place on shared counter anyway
                        return "place_onion_on_counter"
                    elif empty_shared:
                        return "place_onion_on_counter"  # Clear onion to get dish for ready soup
                    else:
                        # Shared counters have onions or are empty - clear onions first
                        if caps.get("dish_disp", False):
                            # Can clear onions (by placing dishes) - check if there are regular counters available
                            all_counters_set = set(mdp.get_counter_locations())
                            shared_counters_set = set(shared_counters)
                            regular_counters = all_counters_set - shared_counters_set
                            empty_regular_counters = [c for c in regular_counters if not self._is_occupied(state, c)]

                            if empty_regular_counters:
                                return "clear_counter_for_onion"  # Clear onion to make room for dish
                        # Can't clear onions - place on shared counter anyway
                        return "place_onion_on_counter"
                else:
                    return "place_onion_on_counter"  # Normal onion sharing
            elif ctx.holding == "dish":
                if all_pots_empty:
                    # All pots empty - check if we need to clear onions first
                    layout_name = getattr(mdp, "layout_name", "unknown_layout")
                    pot_capable_agents = self._find_agents_with_pot_capability(layout_name, len(state.players))
                    shared_counters = []
                    for pot_agent_idx in pot_capable_agents:
                        if pot_agent_idx != ctx.idx:
                            counters = self._get_shared_counters_with_agent(layout_name, ctx.idx, pot_agent_idx)
                            shared_counters.extend(counters)
                    shared_counters = list(set(shared_counters))

                    # Check what items are on shared counters
                    onion_counters = set(self._get_counters_with_item(state, mdp, "onion"))
                    onion_on_shared = sum(1 for c in shared_counters if c in onion_counters)
                    empty_shared = [c for c in shared_counters if not self._is_occupied(state, c)]

                    if onion_on_shared >= len(shared_counters) * 0.7:  # 70% 이상이 onion으로 차 있음
                        # Shared counters are full of onions - need to clear onions first
                        caps = ctx.capabilities or {}
                        if caps.get("dish_disp", False):
                            # Can clear onions (by placing dishes) - check if there are regular counters available
                            all_counters_set = set(mdp.get_counter_locations())
                            shared_counters_set = set(shared_counters)
                            regular_counters = all_counters_set - shared_counters_set
                            empty_regular_counters = [c for c in regular_counters if not self._is_occupied(state, c)]

                            if empty_regular_counters:
                                return "clear_counter_for_onion"  # Clear onion to make room for dish
                    # Clear dish to make room for onion (either counters not full or can't clear onions)
                    return "clear_counter_for_onion"  # Clear dish to make room for onion
                elif has_ready_pots:
                    return "pickup_onion_soup"  # Pick up soup from ready pot
                else:
                    return "place_dish_on_counter"  # No ready pots, share dish
            else:  # Not holding anything
                if has_ready_pots and has_dish_disp:
                    # High priority: ready pots exist and we can get dishes
                    # But first check if we can actually place the dish (shared counters not full of dishes)
                    layout_name = getattr(mdp, "layout_name", "unknown_layout")
                    pot_capable_agents = self._find_agents_with_pot_capability(layout_name, len(state.players))
                    shared_counters = []
                    for pot_agent_idx in pot_capable_agents:
                        if pot_agent_idx != ctx.idx:
                            counters = self._get_shared_counters_with_agent(layout_name, ctx.idx, pot_agent_idx)
                            shared_counters.extend(counters)
                    shared_counters = list(set(shared_counters))

                    # Check what items are on shared counters
                    dish_counters = set(self._get_counters_with_item(state, mdp, "dish"))
                    dish_on_shared = sum(1 for c in shared_counters if c in dish_counters)
                    empty_shared = [c for c in shared_counters if not self._is_occupied(state, c)]

                    if dish_on_shared >= len(shared_counters) * 0.8:  # 80% 이상이 dish로 차 있음
                        # Shared counters are full of dishes - need to clear dishes first
                        caps = ctx.capabilities or {}
                        if caps.get("dish_disp", False):
                            # Can clear dishes - check if there are regular counters available
                            all_counters_set = set(mdp.get_counter_locations())
                            shared_counters_set = set(shared_counters)
                            regular_counters = all_counters_set - shared_counters_set
                            empty_regular_counters = [c for c in regular_counters if not self._is_occupied(state, c)]

                            if empty_regular_counters:
                                return "clear_counter_for_dish"  # Clear dish to make room
                        # Can't clear dishes or no empty regular counters - fallback to getting onion instead
                        return "pickup_onion_from_dispenser"  # Get onion for sharing
                    else:
                        return "pickup_dish_from_dispenser"  # Get dish for ready soup
                elif has_non_ready_pots:
                    return "pickup_onion_from_dispenser"  # Get onion to share
                # All pots ready but no dish access - low priority
                return None
        
        # Case 3: Onion receiver and cooker (onion_disp: False, pot: True)
        elif role == 3:
            if ctx.holding == "dish":
                if has_ready_pots:
                    return "pickup_onion_soup"  # Pick up soup from ready pot
                else:
                    return "place_dish_on_counter"  # No ready pots, place dish near pot
            elif ctx.holding == "onion":
                if has_non_ready_pots:
                    return "put_onion_in_pot"  # Put onion in non-ready pot
                else:
                    return "place_onion_on_counter"  # All pots ready, place onion near pot
            else:  # Not holding anything
                if has_ready_pots:
                    # Priority: get dish first for ready soup
                    return "pickup_dish_from_counter"  # Get dish from shared counter
                elif has_non_ready_pots:
                    return "pickup_onion_from_counter"  # Get onion from shared counter
                # All pots ready but no dish access - low priority
                return None
        
        # Case 4: Direct dish supplier and soup picker (dish_disp: True, pot: True)
        elif role == 4:
            if ctx.holding == "dish":
                if has_ready_pots:
                    return "pickup_onion_soup"  # Pick up soup from ready pot
                else:
                    return "place_dish_on_counter"  # No ready pots, place dish near pot
            else:  # Not holding dish
                if has_ready_pots:
                    # Check if we can access dish dispenser, if not this will be handled by feasibility filtering
                    caps = ctx.capabilities or {}
                    if caps.get("dish_disp", False):
                        return "pickup_dish_from_dispenser"  # Get dish for soup
                # No ready pots - low priority
                return None
        
        # Case 5: Dish supplier via sharing (dish_disp: True, pot: False)
        elif role == 5:
            # Check for cooking pots (will be ready soon) in addition to ready pots
            pot_states = mdp.get_pot_states(state)
            cooking_pots = mdp.get_cooking_pots(pot_states)
            has_cooking_pots = len(cooking_pots) > 0
            
            # Check for empty pots
            empty_pots = mdp.get_empty_pots(pot_states)
            all_pots_empty = len(empty_pots) == len(mdp.get_pot_locations())
            
            if ctx.holding == "dish":
                if all_pots_empty:
                    # All pots empty - need to clear space for onion
                    return "clear_counter_for_onion"  # Clear dish to make room for onion
                else:
                    return "place_dish_on_counter"  # Share dish
            elif ctx.holding == "onion":
                if has_ready_pots or has_cooking_pots:
                    # Ready or cooking pots exist - need to clear onion to make room for dish
                    # Check if shared counters are available
                    layout_name = getattr(mdp, "layout_name", "unknown_layout")
                    pot_capable_agents = self._find_agents_with_pot_capability(layout_name, len(state.players))
                    shared_counters = []
                    for pot_agent_idx in pot_capable_agents:
                        if pot_agent_idx != ctx.idx:
                            counters = self._get_shared_counters_with_agent(layout_name, ctx.idx, pot_agent_idx)
                            shared_counters.extend(counters)
                    shared_counters = list(set(shared_counters))
                    empty_shared_counters = [c for c in shared_counters if not self._is_occupied(state, c)]
                    
                    if empty_shared_counters:
                        return "place_onion_on_counter"  # Normal onion sharing
                    else:
                        # No empty shared counters - need to clear space for dish
                        return "clear_counter_for_dish"  # Clear counter space first
                else:
                    return "place_onion_on_counter"  # Normal onion sharing
            else:  # Not holding anything
                if has_ready_pots or has_cooking_pots:
                    # High priority when ready/cooking pots exist - need to supply dishes
                    caps = ctx.capabilities or {}
                    if caps.get("dish_disp", False):
                        return "pickup_dish_from_dispenser"  # Get dish to share
                elif has_non_ready_pots:
                    # Only non-ready pots exist - normal onion sharing priority
                    caps = ctx.capabilities or {}
                    if caps.get("onion_disp", False):
                        return "pickup_onion_from_dispenser"  # Get onion to share
                # No pots or can't access dispensers - low priority
                return None
        
        # Case 6: Dish receiver and soup picker (dish_disp: False, pot: True)
        elif role == 6:
            if ctx.holding == "dish":
                if has_ready_pots:
                    return "pickup_onion_soup"  # Pick up soup
                else:
                    return "place_dish_on_counter"  # No ready pots, place dish near pot
            else:  # Not holding dish
                if has_ready_pots:
                    return "pickup_dish_from_counter"  # Get dish from shared counter
                # No ready pots - low priority
                return None
        
        # Case 7: Full service (pot: True, serve: True)
        elif role == 7:
            if ctx.holding == "soup":
                caps = ctx.capabilities or {}
                if caps.get("serve", False):
                    return "deliver_to_serve"  # Deliver soup
            elif ctx.holding == "dish":
                if has_ready_pots:
                    return "pickup_onion_soup"  # Pick up soup from ready pot
                else:
                    return "place_dish_on_counter"  # No ready pots, place dish near pot
            else:  # Not holding anything
                if has_ready_pots:
                    # Priority: get dish first for ready soup
                    caps = ctx.capabilities or {}
                    if caps.get("dish_disp", False):
                        return "pickup_dish_from_dispenser"  # Get dish for soup
                    else:
                        return "pickup_dish_from_counter"  # Get dish from shared counter
                # No ready pots - low priority
                return None
        
        # Case 8: Soup maker and sharer (pot: True, serve: False)
        elif role == 8:
            if ctx.holding == "soup":
                return "place_soup_on_counter"  # Share soup with delivery-capable agent
            else:  # Not holding soup
                if has_ready_pots:
                    return "pickup_onion_soup"  # Get soup to share
                # No ready pots - low priority
                return None
        
        # Case 9: Soup receiver and deliverer (pot: False, serve: True)
        elif role == 9:
            if ctx.holding == "soup":
                caps = ctx.capabilities or {}
                if caps.get("serve", False):
                    return "deliver_to_serve"  # Deliver soup
            else:  # Not holding soup
                return "pickup_soup_from_counter"  # Get soup from shared counter
        
        return None

    def _get_all_shared_counters_for_pot_agents(self, ctx: AgentContext, layout_name: str, num_agents: int = 2) -> List[Tuple[int, int]]:
        """Get all shared counters with pot-capable agents."""
        pot_capable_agents = self._find_agents_with_pot_capability(layout_name, num_agents)
        
        shared_counters = []
        for pot_agent_idx in pot_capable_agents:
            if pot_agent_idx != ctx.idx:
                counters = self._get_shared_counters_with_agent(layout_name, ctx.idx, pot_agent_idx)
                shared_counters.extend(counters)
        
        # Remove duplicates and return
        return list(set(shared_counters))
    
    def _get_shared_counter_for_pot_agents(self, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld, layout_name: str) -> Optional[Tuple[int, int]]:
        """Find the nearest shared counter with pot-capable agents (prefer empty, but allow occupied if needed)."""
        pot_capable_agents = self._find_agents_with_pot_capability(layout_name, len(state.players))
        print(f"    🍲 Pot-capable agents: {pot_capable_agents}")
        
        shared_counters = []
        for pot_agent_idx in pot_capable_agents:
            if pot_agent_idx != ctx.idx:
                counters = self._get_shared_counters_with_agent(layout_name, ctx.idx, pot_agent_idx)
                print(f"    🤝 Shared counters with Agent {pot_agent_idx}: {counters}")
                shared_counters.extend(counters)
        
        # Remove duplicates
        shared_counters = list(set(shared_counters))
        empty_counters = [c for c in shared_counters if not self._is_occupied(state, c)]
        
        print(f"    📍 All shared counters: {shared_counters}")
        print(f"    ✨ Empty shared counters: {empty_counters}")
        
        # Check for ready/cooking pots to determine priority
        pot_states = mdp.get_pot_states(state)
        ready_pots = mdp.get_ready_pots(pot_states)
        cooking_pots = mdp.get_cooking_pots(pot_states)
        has_ready_pots = len(ready_pots) > 0
        has_cooking_pots = len(cooking_pots) > 0
        
        # Prefer empty counters only - don't try to place on occupied counters
        if empty_counters:
            selected_counter = self._nearest_of(ctx.position, empty_counters)
            print(f"    🎯 Selected empty counter: {selected_counter}")
        else:
            # No empty counters available
            selected_counter = None
            if shared_counters:
                print(f"    ❌ No empty shared counters available (all occupied)")
            else:
                print(f"    ❌ No shared counters found")
        
        return selected_counter

    def _get_shared_counter_for_serve_agents(self, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld, layout_name: str) -> Optional[Tuple[int, int]]:
        """Find the nearest empty shared counter with serve-capable agents."""
        serve_capable_agents = self._find_agents_with_serve_capability(layout_name, len(state.players))
        shared_counters = []
        
        for serve_agent_idx in serve_capable_agents:
            if serve_agent_idx != ctx.idx:
                counters = self._get_shared_counters_with_agent(layout_name, ctx.idx, serve_agent_idx)
                shared_counters.extend(counters)
        
        # Remove duplicates and filter for empty counters
        shared_counters = list(set(shared_counters))
        empty_counters = [c for c in shared_counters if not self._is_occupied(state, c)]
        
        return self._nearest_of(ctx.position, empty_counters)

    def _get_onion_from_shared_counter(self, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld, layout_name: str) -> Optional[Tuple[int, int]]:
        """Find shared counter with onion from onion_disp-capable agents."""
        onion_disp_agents = self._find_agents_with_onion_disp_capability(layout_name, len(state.players))
        shared_counters = []
        
        for onion_agent_idx in onion_disp_agents:
            if onion_agent_idx != ctx.idx:
                counters = self._get_shared_counters_with_agent(layout_name, ctx.idx, onion_agent_idx)
                shared_counters.extend(counters)
        
        # Remove duplicates and filter for counters with onions
        shared_counters = list(set(shared_counters))
        counters_with_onion = self._get_counters_with_item(state, mdp, "onion")
        onion_counters = [c for c in shared_counters if c in counters_with_onion]
        
        return self._nearest_of(ctx.position, onion_counters) if onion_counters else None

    def _get_dish_from_shared_counter(self, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld, layout_name: str) -> Optional[Tuple[int, int]]:
        """Find shared counter with dish from dish_disp-capable agents."""
        dish_disp_agents = self._find_agents_with_dish_disp_capability(layout_name, len(state.players))
        shared_counters = []
        
        for dish_agent_idx in dish_disp_agents:
            if dish_agent_idx != ctx.idx:
                counters = self._get_shared_counters_with_agent(layout_name, ctx.idx, dish_agent_idx)
                shared_counters.extend(counters)
        
        # Remove duplicates and filter for counters with dishes
        shared_counters = list(set(shared_counters))
        counters_with_dish = self._get_counters_with_item(state, mdp, "dish")
        dish_counters = [c for c in shared_counters if c in counters_with_dish]
        
        return self._nearest_of(ctx.position, dish_counters) if dish_counters else None

    def _get_soup_from_shared_counter(self, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld, layout_name: str) -> Optional[Tuple[int, int]]:
        """Find shared counter with soup from pot-capable agents."""
        pot_capable_agents = self._find_agents_with_pot_capability(layout_name, len(state.players))
        shared_counters = []
        
        for pot_agent_idx in pot_capable_agents:
            if pot_agent_idx != ctx.idx:
                counters = self._get_shared_counters_with_agent(layout_name, ctx.idx, pot_agent_idx)
                shared_counters.extend(counters)
        
        # Remove duplicates and filter for counters with soup
        shared_counters = list(set(shared_counters))
        counters_with_soup = self._get_counters_with_item(state, mdp, "soup")
        soup_counters = [c for c in shared_counters if c in counters_with_soup]
        
        return self._nearest_of(ctx.position, soup_counters) if soup_counters else None

    def _find_agents_with_onion_disp_capability(self, layout_name: str, num_agents: int) -> List[int]:
        """Find agents that can reach onion dispensers."""
        onion_disp_agents = []
        for i in range(num_agents):
            if layout_name in self._static_cache and i in self._static_cache[layout_name]:
                caps = self._static_cache[layout_name][i].get("capabilities", {})
                if caps.get("onion_disp", False):
                    onion_disp_agents.append(i)
        return onion_disp_agents

    def _find_agents_with_dish_disp_capability(self, layout_name: str, num_agents: int) -> List[int]:
        """Find agents that can reach dish dispensers."""
        dish_disp_agents = []
        for i in range(num_agents):
            if layout_name in self._static_cache and i in self._static_cache[layout_name]:
                caps = self._static_cache[layout_name][i].get("capabilities", {})
                if caps.get("dish_disp", False):
                    dish_disp_agents.append(i)
        return dish_disp_agents

    def _find_agents_with_serve_capability(self, layout_name: str, num_agents: int) -> List[int]:
        """Find agents that can reach serving locations."""
        serve_agents = []
        for i in range(num_agents):
            if layout_name in self._static_cache and i in self._static_cache[layout_name]:
                caps = self._static_cache[layout_name][i].get("capabilities", {})
                if caps.get("serve", False):
                    serve_agents.append(i)
        return serve_agents

    def _best_reachable_pot_for_onion(self, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld) -> Optional[Tuple[int, int]]:
        """
        Find the best pot to put an onion in.
        Priority: 1) Partially full pots, 2) Empty pots
        Avoid full/cooking/ready pots.
        """
        pot_states = mdp.get_pot_states(state)
        pot_locations = mdp.get_pot_locations()
        
        # Get different types of pots
        empty_pots = mdp.get_empty_pots(pot_states)
        partially_full_pots = mdp.get_partially_full_pots(pot_states)
        cooking_pots = mdp.get_cooking_pots(pot_states)
        ready_pots = mdp.get_ready_pots(pot_states)
        
        print(f"    🍲 Pot analysis for onion placement:")
        print(f"      Empty pots: {empty_pots}")
        print(f"      Partially full pots: {partially_full_pots}")
        print(f"      Cooking pots: {cooking_pots}")
        print(f"      Ready pots: {ready_pots}")
        
        # Priority 1: Partially full pots (can add more ingredients)
        fillable_pots = []
        if partially_full_pots:
            fillable_pots.extend(partially_full_pots)
            print(f"    ✅ Found partially full pots to fill: {partially_full_pots}")
        
        # Priority 2: Empty pots
        if empty_pots:
            fillable_pots.extend(empty_pots)
            print(f"    ✅ Found empty pots to fill: {empty_pots}")
        
        if not fillable_pots:
            print(f"    ❌ No fillable pots found (all pots are cooking/ready)")
            return None
        
        # Find the nearest reachable pot
        best_pot = None
        best_distance = float('inf')
        
        for pot_pos in fillable_pots:
            # Find adjacent positions to this pot
            adjacent_positions = self._adjacent_stand_tiles(pot_pos, mdp)
            reachable_adjacent = [pos for pos in adjacent_positions if pos in (ctx.reachable or set())]
            
            print(f"    🔍 Checking pot {pot_pos}:")
            print(f"      Adjacent positions: {adjacent_positions}")
            print(f"      Agent reachable area size: {len(ctx.reachable or set())}")
            print(f"      Reachable adjacent: {reachable_adjacent}")
            
            if reachable_adjacent:
                # Calculate distance to nearest adjacent position
                min_dist_to_pot = min(self._manhattan(ctx.position, adj_pos) for adj_pos in reachable_adjacent)
                print(f"      Distance to pot: {min_dist_to_pot}")
                if min_dist_to_pot < best_distance:
                    best_distance = min_dist_to_pot
                    best_pot = pot_pos
                    print(f"      ✅ New best pot: {best_pot}")
            else:
                print(f"      ❌ No reachable adjacent positions for pot {pot_pos}")
        
        if best_pot:
            print(f"    🎯 Selected best pot for onion: {best_pot} (distance: {best_distance})")
            # Return the adjacent position to move to, not the pot itself
            return self._nearest_reachable_adjacent(ctx, [best_pot], mdp)
        else:
            print(f"    ❌ No reachable fillable pots found")
            return None

    def _get_counter_clearing_target(self, ctx: AgentContext, state: OvercookedState, mdp: OvercookedGridworld, layout_name: str, clear_for: str = "dish") -> Optional[Tuple[int, int]]:
        """
        카운터 정리를 위한 타겟을 찾습니다.
        
        Args:
            clear_for: "dish" 또는 "onion" - 무엇을 위한 공간을 확보할지
            
        로직:
        - clear_for="dish": onion을 들고 있을 때, onion을 일반 카운터에 놓아서 dish 공간 확보
        - clear_for="onion": dish를 들고 있을 때, dish를 일반 카운터에 놓아서 onion 공간 확보
        """
        print(f"    🧹 Finding counter clearing target for {clear_for} for Agent {ctx.idx}")
        
        # 어떤 아이템을 치워야 하는지 결정
        if clear_for == "dish":
            required_holding = "onion"
            item_to_clear = "onion"
        elif clear_for == "onion":
            required_holding = "dish"
            item_to_clear = "dish"
        else:
            print(f"    ❌ Invalid clear_for parameter: {clear_for}")
            return None
        
        # 필요한 아이템을 들고 있는지 확인
        if ctx.holding != required_holding:
            print(f"    ❌ Agent not holding {required_holding}, cannot clear counter for {clear_for}")
            return None

        # 모든 카운터 위치 가져오기
        all_counters = set(mdp.get_counter_locations())

        # 공유 카운터 찾기 (pot-capable agents와 공유하는 카운터)
        pot_capable_agents = self._find_agents_with_pot_capability(layout_name, len(state.players))
        shared_counters = set()
        for pot_agent_idx in pot_capable_agents:
            if pot_agent_idx != ctx.idx:
                counters = self._get_shared_counters_with_agent(layout_name, ctx.idx, pot_agent_idx)
                shared_counters.update(counters)

        # 일반 카운터 = 전체 카운터 - 공유 카운터
        regular_counters = all_counters - shared_counters

        # 빈 일반 카운터 찾기
        empty_regular_counters = [c for c in regular_counters if not self._is_occupied(state, c)]

        print(f"    📦 All counters: {len(all_counters)}")
        print(f"    🤝 Shared counters: {len(shared_counters)}")
        print(f"    📋 Regular counters: {len(regular_counters)}")
        print(f"    ✨ Empty regular counters: {len(empty_regular_counters)}")

        # 1. 빈 일반 카운터 우선 사용
        if empty_regular_counters:
            target_counter = self._nearest_of(ctx.position, empty_regular_counters)
            print(f"    🎯 Using empty regular counter: {target_counter}")
        else:
            # 2. 빈 일반 카운터가 없으면, dispenser 위치를 타겟으로 사용
            print(f"    ⚠️ No empty regular counters, using dispenser position directly...")

            # clear_for에 따라 적절한 dispenser 찾기
            if clear_for == "dish":
                # dish dispenser 사용
                dispenser_locations = self._get_dish_dispensers(mdp)
                print(f"    🍽️ Using dish dispenser positions: {dispenser_locations}")
            elif clear_for == "onion":
                # onion dispenser 사용
                dispenser_locations = self._get_onion_dispensers(mdp)
                print(f"    🧅 Using onion dispenser positions: {dispenser_locations}")
            else:
                dispenser_locations = []

            if dispenser_locations:
                # dispenser 위치를 타겟으로 사용
                target_counter = self._nearest_of(ctx.position, dispenser_locations)
                print(f"    🎯 Using dispenser position: {target_counter}")
            else:
                # 3. dispenser도 없으면, 임의의 빈 카운터 사용
                print(f"    ⚠️ No dispenser found, looking for any empty counter...")
                all_empty_counters = [c for c in all_counters if not self._is_occupied(state, c)]

                if all_empty_counters:
                    target_counter = self._nearest_of(ctx.position, all_empty_counters)
                    print(f"    🎯 Using any empty counter: {target_counter}")
                else:
                    # 4. 빈 카운터가 하나도 없으면 현재 위치에서 상호작용 (강제 배치)
                    print(f"    ❌ No empty counters at all, will place at occupied counter")
                    target_counter = None

        if target_counter:
            # 이미 인접한지 확인 (onion용 로직 통합)
            adjacent_positions = self._adjacent_stand_tiles(target_counter, mdp)
            if ctx.position in adjacent_positions:
                # 이미 인접하면 현재 위치에서 상호작용
                print(f"    🎯 Already adjacent to target counter {target_counter}, staying at {ctx.position}")
                return ctx.position
            else:
                # 인접 위치로 이동 필요
                target_adjacent = self._nearest_reachable_adjacent(ctx, [target_counter], mdp)
                print(f"    🎯 Selected counter for {item_to_clear}: {target_counter} → adjacent: {target_adjacent}")
                return target_adjacent
        else:
            # 빈 카운터가 없어 강제 배치
            return ctx.position



