#!/usr/bin/env python3
"""
Simple test script for the layout generator
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pymarlzooplus'))

from pymarlzooplus.envs.overcooked_ai.src.overcooked_ai_py.mdp.layout_generator import (
    LayoutGenerator, MDPParamsGenerator, DEFAULT_FEATURE_TYPES,
    POT, ONION_DISPENSER, DISH_DISPENSER, SERVING_LOC
)

def test_basic_layout_generation():
    """Test basic layout generation with fixed parameters"""
    print("=== Testing Basic Layout Generation ===")
    
    # Create parameter generator with fixed parameters
    params_gen = MDPParamsGenerator.from_fixed_param({
        "inner_shape": (8, 6),
        "prop_empty": 0.7,
        "prop_feats": 0.15,
        "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
        "recipe_values": [20],
        "recipe_times": [20],
        "display": True,
    })
    
    # Create layout generator
    generator = LayoutGenerator(params_gen, outer_shape=(8, 6))
    
    try:
        # Generate MDP
        mdp = generator.generate_padded_mdp({})
        print("✓ Layout generation successful!")
        print(f"  Grid shape: {len(mdp.terrain_mtx[0])} x {len(mdp.terrain_mtx)}")
        # Check if start_player_positions exists
        if hasattr(mdp, 'start_player_positions'):
            print(f"  Number of players: {len(mdp.start_player_positions)}")
        else:
            print("  Player positions: Not available")
        
        # Display the generated layout
        print("\nGenerated Layout:")
        print(mdp.terrain_mtx)
        
    except Exception as e:
        print(f"✗ Layout generation failed: {e}")
        import traceback
        traceback.print_exc()

def test_dynamic_layout_generation():
    """Test dynamic layout generation with varying parameters"""
    print("\n=== Testing Dynamic Layout Generation ===")
    
    def dynamic_params_fn(outside_info):
        episode = outside_info.get("episode", 0)
        
        # Vary layout based on episode number
        if episode < 5:
            return {
                "inner_shape": (5, 4),
                "prop_empty": 0.9,
                "prop_feats": 0.1,
                "start_all_orders": [{"ingredients": ["onion", "onion"]}],
                "recipe_values": [15],
                "recipe_times": [15],
                "display": True,  # 레이아웃 출력 활성화
            }
        else:
            return {
                "inner_shape": (8, 6),
                "prop_empty": 0.7,
                "prop_feats": 0.3,
                "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
                "recipe_values": [25],
                "recipe_times": [25],
                "display": True,  # 레이아웃 출력 활성화
            }
    
    # Create dynamic parameter generator
    params_gen = MDPParamsGenerator(dynamic_params_fn)
    
    try:
        # Test different episodes
        for episode in [1, 3, 7, 10]:
            print(f"\nEpisode {episode}:")
            
            # Create generator with appropriate outer shape for each episode
            if episode < 5:
                outer_shape = (6, 5)  # smaller outer shape
            else:
                outer_shape = (10, 8)  # larger outer shape
                
            generator = LayoutGenerator(params_gen, outer_shape=outer_shape)
            mdp = generator.generate_padded_mdp({"episode": episode})
            print(f"  Grid shape: {len(mdp.terrain_mtx[0])} x {len(mdp.terrain_mtx)}")
            # Check available attributes
            if hasattr(mdp, 'recipe_values'):
                print(f"  Recipe value: {mdp.recipe_values[0]}")
            elif hasattr(mdp, 'start_all_orders'):
                print(f"  Number of orders: {len(mdp.start_all_orders)}")
            else:
                print(f"  Available attributes: {[attr for attr in dir(mdp) if not attr.startswith('_')]}")
            
            # Display the generated layout
            print(f"  Generated Layout for Episode {episode}:")
            for row in mdp.terrain_mtx:
                print(f"  {row}")
            
    except Exception as e:
        print(f"✗ Dynamic layout generation failed: {e}")
        import traceback
        traceback.print_exc()


def test_forced_coordination():
    """Test forced coordination layout generation"""
    print("\n=== Testing Forced Coordination Layout ===")
    
    # Forced coordination parameters
    forced_params = {
        "coordination_mode": "forced",
        "n_sets": 2,
        "inner_shape": [(4, 3), (4, 3)],  # Two rooms of size 4x3
        "room_connectivity": [(0, 1)],  # Room 0 connected to Room 1
        "prop_empty": 0.7,
        "shared_counter_prop": 0.1,
        "feature_distribution": {
            0: [POT, ONION_DISPENSER],  # Room 0: cooking facilities
            1: [DISH_DISPENSER, SERVING_LOC],  # Room 1: serving facilities
        },
        "player_distribution": {
            0: [0],  # Player 0 in room 0
            1: [1],  # Player 1 in room 1
        },
        "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
        "recipe_values": [20],
        "recipe_times": [20],
        "display": True,
    }
    
    params_gen = MDPParamsGenerator.from_fixed_param(forced_params)
    generator = LayoutGenerator(params_gen, outer_shape=(15, 8))  # Much larger outer shape for two rooms + shared space
    
    try:
        print("  Generating forced coordination layout...")
        print(f"  Parameters: n_sets={forced_params['n_sets']}, inner_shapes={forced_params['inner_shape']}")
        print(f"  Room connectivity: {forced_params['room_connectivity']}")
        print(f"  Outer shape: (15, 8)")
        
        mdp = generator.generate_padded_mdp({})
        print("✓ Forced coordination layout generation successful!")
        print(f"  Grid shape: {len(mdp.terrain_mtx[0])} x {len(mdp.terrain_mtx)}")
        
        # Check for player positions
        if hasattr(mdp, 'start_player_positions'):
            print(f"  Number of players: {len(mdp.start_player_positions)}")
            print(f"  Player positions: {mdp.start_player_positions}")
        
        # Display the generated layout
        print("  Generated Forced Coordination Layout:")
        for row in mdp.terrain_mtx:
            print(f"  {row}")
            
    except Exception as e:
        print(f"✗ Forced coordination layout generation failed: {e}")
        import traceback
        traceback.print_exc()


def test_graph_based_forced_coordination():
    """Test graph-based forced coordination layout generation"""
    print("\n=== Testing Graph-Based Forced Coordination Layout ===")
    
    # Graph-based forced coordination parameters
    graph_params = {
        "coordination_mode": "forced",
        "n_sets": 3,
        "inner_shape": [(5, 4), (4, 3), (6, 3)],  # Three rooms of different sizes
        "room_connectivity": [(0, 1), (1, 2)],  # Linear chain: 0-1-2
        "prop_empty": 0.7,
        "feature_distribution": {
            0: [POT, ONION_DISPENSER],  # Room 0: cooking facilities
            1: [DISH_DISPENSER],  # Room 1: intermediate facilities
            2: [SERVING_LOC],  # Room 2: serving facilities
        },
        "player_distribution": {
            0: [0],  # Player 0 in room 0
            1: [1],  # Player 1 in room 1
            2: [2],  # Player 2 in room 2
        },
        "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
        "recipe_values": [20],
        "recipe_times": [20],
        "display": True,
    }
    
    params_gen = MDPParamsGenerator.from_fixed_param(graph_params)
    generator = LayoutGenerator(params_gen, outer_shape=(20, 10))  # Large outer shape for three rooms + shared spaces
    
    try:
        print("  Generating graph-based forced coordination layout...")
        print(f"  Parameters: n_sets={graph_params['n_sets']}, inner_shapes={graph_params['inner_shape']}")
        print(f"  Room connectivity: {graph_params['room_connectivity']}")
        print(f"  Outer shape: (20, 10)")
        
        mdp = generator.generate_padded_mdp({})
        print("✓ Graph-based forced coordination layout generation successful!")
        print(f"  Grid shape: {len(mdp.terrain_mtx[0])} x {len(mdp.terrain_mtx)}")
        
        # Check for player positions
        if hasattr(mdp, 'start_player_positions'):
            print(f"  Number of players: {len(mdp.start_player_positions)}")
            print(f"  Player positions: {mdp.start_player_positions}")
        
        # Display the generated layout
        print("  Generated Graph-Based Forced Coordination Layout:")
        for row in mdp.terrain_mtx:
            print(f"  {row}")
            
    except Exception as e:
        print(f"✗ Graph-based forced coordination layout generation failed: {e}")
        import traceback
        traceback.print_exc()


def test_complex_graph_connectivity():
    """Test complex graph connectivity patterns"""
    print("\n=== Testing Complex Graph Connectivity ===")
    
    # Complex graph with branching
    complex_params = {
        "coordination_mode": "forced",
        "n_sets": 4,
        "inner_shape": [(4, 4), (3, 3), (3, 3), (5, 3)],  # Four rooms
        "room_connectivity": [(0, 1), (0, 2), (1, 3), (2, 3)],  # Star pattern: 0 connects to 1,2; 1,2 connect to 3
        "prop_empty": 0.7,
        "feature_distribution": {
            0: [POT, ONION_DISPENSER],  # Central room: cooking
            1: [DISH_DISPENSER],  # Branch 1: preparation
            2: [DISH_DISPENSER],  # Branch 2: preparation
            3: [SERVING_LOC],  # Final room: serving
        },
        "player_distribution": {
            0: [0],  # Player 0 in central room
            1: [1],  # Player 1 in branch 1
            2: [2],  # Player 2 in branch 2
            3: [3],  # Player 3 in final room
        },
        "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
        "recipe_values": [20],
        "recipe_times": [20],
        "display": True,
    }
    
    params_gen = MDPParamsGenerator.from_fixed_param(complex_params)
    generator = LayoutGenerator(params_gen, outer_shape=(25, 12))  # Very large outer shape for complex layout
    
    try:
        print("  Generating complex graph connectivity layout...")
        print(f"  Parameters: n_sets={complex_params['n_sets']}, inner_shapes={complex_params['inner_shape']}")
        print(f"  Room connectivity: {complex_params['room_connectivity']}")
        print(f"  Outer shape: (25, 12)")
        
        mdp = generator.generate_padded_mdp({})
        print("✓ Complex graph connectivity layout generation successful!")
        print(f"  Grid shape: {len(mdp.terrain_mtx[0])} x {len(mdp.terrain_mtx)}")
        
        # Check for player positions
        if hasattr(mdp, 'start_player_positions'):
            print(f"  Number of players: {len(mdp.start_player_positions)}")
            print(f"  Player positions: {mdp.start_player_positions}")
        
        # Display the generated layout
        print("  Generated Complex Graph Connectivity Layout:")
        for row in mdp.terrain_mtx:
            print(f"  {row}")
            
    except Exception as e:
        print(f"✗ Complex graph connectivity layout generation failed: {e}")
        import traceback
        traceback.print_exc()


def test_factory_method():
    """Test the factory method approach"""
    print("\n=== Testing Factory Method ===")
    
    try:
        # Use the factory method
        mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(
            mdp_params={
                "inner_shape": (6, 5),
                "prop_empty": 0.7,
                "prop_feats": 0.3,
                "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
                "recipe_values": [20],
                "recipe_times": [20],
                "display": True,
            },
            outer_shape=(6, 5)
        )
        
        # Generate MDP using the function
        mdp = mdp_fn({})
        print("✓ Factory method successful!")
        print(f"  Grid shape: {len(mdp.terrain_mtx[0])} x {len(mdp.terrain_mtx)}")
        
    except Exception as e:
        print(f"✗ Factory method failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    print("Layout Generator Test Suite")
    print("=" * 50)
    
    test_basic_layout_generation()
    test_dynamic_layout_generation()
    test_forced_coordination()
    test_graph_based_forced_coordination()
    test_complex_graph_connectivity()
    test_factory_method()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")

if __name__ == "__main__":
    main()
