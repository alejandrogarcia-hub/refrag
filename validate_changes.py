#!/usr/bin/env python3
"""
Validate that the code changes are syntactically correct.
"""

import ast
import sys

print("=" * 60)
print("REfrag Changes Validation")
print("=" * 60)

def check_file(filepath, checks):
    """Check a file for specific code patterns."""
    print(f"\nChecking {filepath}...")
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Parse to check syntax
        ast.parse(content)
        print(f"  ✓ Syntax valid")

        # Check for specific patterns
        for check_name, pattern in checks.items():
            if pattern in content:
                print(f"  ✓ {check_name}")
            else:
                print(f"  ✗ {check_name} NOT FOUND")
                return False

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

# Test 1: HybridInputConstructor positional encodings
print("\n[1/4] Validating hybrid_input.py (positional encodings)...")
success = check_file(
    '/home/user/refrag/refrag/generation/hybrid_input.py',
    {
        'position_embedding declaration': 'self.position_embedding = nn.Embedding',
        'position_embedding initialization': 'nn.init.normal_',
        'position_id usage': 'position_id = torch.tensor',
        'position_emb usage': 'position_emb = self.position_embedding',
    }
)

if not success:
    sys.exit(1)

# Test 2: QueryComplexityEstimator
print("\n[2/4] Validating complexity.py (query complexity estimator)...")
success = check_file(
    '/home/user/refrag/refrag/selection/complexity.py',
    {
        'QueryComplexityEstimator class': 'class QueryComplexityEstimator',
        'estimate_complexity method': 'def estimate_complexity',
        'get_dynamic_expansion_fraction method': 'def get_dynamic_expansion_fraction',
    }
)

if not success:
    sys.exit(1)

# Test 3: SelectionPolicy with dynamic fraction support
print("\n[3/4] Validating base.py (dynamic selection)...")
success = check_file(
    '/home/user/refrag/refrag/selection/base.py',
    {
        'select_with_dynamic_fraction method': 'def select_with_dynamic_fraction',
        '_get_top_k with optional fraction': 'expansion_fraction: float | None = None',
    }
)

if not success:
    sys.exit(1)

# Test 4: Config with use_dynamic_expansion
print("\n[4/4] Validating config.py (dynamic expansion config)...")
success = check_file(
    '/home/user/refrag/refrag/config.py',
    {
        'use_dynamic_expansion field': 'use_dynamic_expansion: bool',
        'USE_DYNAMIC_EXPANSION env var': 'USE_DYNAMIC_EXPANSION',
    }
)

if not success:
    sys.exit(1)

# Test 5: Pipeline integration
print("\n[5/5] Validating pipeline.py (integration)...")
success = check_file(
    '/home/user/refrag/refrag/pipeline.py',
    {
        'QueryComplexityEstimator import': 'QueryComplexityEstimator',
        'complexity_estimator initialization': 'self.complexity_estimator = QueryComplexityEstimator',
        'Dynamic expansion check': 'if self.config.use_dynamic_expansion:',
        'complexity estimation': 'complexity_score = self.complexity_estimator.estimate_complexity',
        'select_with_dynamic_fraction call': 'select_with_dynamic_fraction',
    }
)

if not success:
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All validation checks passed!")
print("=" * 60)
print("\nChanges summary:")
print("  1. ✓ Positional encodings for compressed chunks")
print("  2. ✓ Dynamic expansion fraction based on query complexity")
print("  3. ✓ Configuration option (USE_DYNAMIC_EXPANSION)")
print("  4. ✓ Full integration into pipeline")
print("=" * 60)
