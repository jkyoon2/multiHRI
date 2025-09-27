#!/bin/bash

echo "🚀 Rule-Based Planner Test Suite"
echo "================================="

# Set Python path
export PYTHONPATH="/home/juliecandoit98/pymarlzooplus:$PYTHONPATH"

cd /home/juliecandoit98/pymarlzooplus

echo ""
echo "1. 🔍 Quick Single Layout Test (3_chefs_smartfactory)"
echo "----------------------------------------------------"
python -m pymarlzooplus.scripts.test_rule_based --single --layout 3_chefs_smartfactory --episodes 1 --steps 50

echo ""
echo "2. 🔍 Extended Test with Rendering (3_chefs_smartfactory)" 
echo "--------------------------------------------------------"
python -m pymarlzooplus.scripts.test_rule_based --single --layout 3_chefs_smartfactory --episodes 2 --steps 100 --render

echo ""
echo "3. 🏆 Comprehensive Test Suite (Multiple Scenarios)"
echo "---------------------------------------------------"
python -m pymarlzooplus.scripts.test_rule_based --episodes 2 --steps 150

echo ""
echo "✅ All tests completed!"
echo ""
echo "📋 Available test options:"
echo "  --single                : Run single scenario test"
echo "  --layout LAYOUT         : Specify layout (cramped_room, asymmetric_advantages, etc.)"
echo "  --episodes N            : Number of episodes per scenario"
echo "  --steps N               : Max steps per episode"
echo "  --render                : Enable visual rendering"
echo "  --debug                 : Enable debug mode"
echo "  --seed N                : Set random seed"
echo ""
echo "📝 Example commands:"
echo "  python -m pymarlzooplus.scripts.test_rule_based --single --layout 3_chefs_smartfactory --render"
echo "  python -m pymarlzooplus.scripts.test_rule_based --single --layout cramped_room --episodes 3"
echo "  python -m pymarlzooplus.scripts.test_rule_based --episodes 3 --steps 200 --debug"
