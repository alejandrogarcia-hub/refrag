#!/usr/bin/env python3
"""
Quick sanity check for REfrag improvements.

Tests:
1. Positional encodings in HybridInputConstructor
2. Dynamic expansion with QueryComplexityEstimator
3. Configuration option for dynamic expansion
"""

import sys

print("=" * 60)
print("REfrag Improvements Sanity Check")
print("=" * 60)

# Test 1: Import all new modules
print("\n[1/5] Testing imports...")
try:
    from refrag.selection import QueryComplexityEstimator
    from refrag.generation import HybridInputConstructor
    from refrag.config import Config
    from refrag.selection.base import SelectionPolicy
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check Config has new parameter
print("\n[2/5] Testing Config with use_dynamic_expansion parameter...")
try:
    config = Config(use_dynamic_expansion=True)
    assert hasattr(config, 'use_dynamic_expansion'), "use_dynamic_expansion attribute missing"
    assert config.use_dynamic_expansion == True, "use_dynamic_expansion not set correctly"

    config_dict = config.to_dict()
    assert 'use_dynamic_expansion' in config_dict, "use_dynamic_expansion missing from to_dict()"
    print("✓ Config with use_dynamic_expansion works")
except Exception as e:
    print(f"✗ Config test failed: {e}")
    sys.exit(1)

# Test 3: Test QueryComplexityEstimator
print("\n[3/5] Testing QueryComplexityEstimator...")
try:
    import numpy as np

    estimator = QueryComplexityEstimator()

    # Create dummy data
    query = "What is machine learning?"
    chunks = ["Machine learning is...", "Deep learning uses...", "Neural networks are..."]
    chunk_embeddings = np.random.randn(3, 768)  # 3 chunks, 768-dim
    query_embedding = np.random.randn(768)

    # Test complexity estimation
    complexity = estimator.estimate_complexity(query, chunks, chunk_embeddings, query_embedding)
    assert 0.0 <= complexity <= 1.0, f"Complexity score {complexity} out of range"

    # Test dynamic fraction calculation
    fraction = estimator.get_dynamic_expansion_fraction(complexity)
    assert 0.1 <= fraction <= 0.5, f"Expansion fraction {fraction} out of range"

    print(f"✓ QueryComplexityEstimator works (complexity={complexity:.3f}, fraction={fraction:.3f})")
except Exception as e:
    print(f"✗ QueryComplexityEstimator test failed: {e}")
    sys.exit(1)

# Test 4: Test SelectionPolicy.select_with_dynamic_fraction
print("\n[4/5] Testing SelectionPolicy.select_with_dynamic_fraction...")
try:
    from refrag.selection import SimilarityPolicy

    policy = SimilarityPolicy(expansion_fraction=0.25)

    # Test that the method exists
    assert hasattr(policy, 'select_with_dynamic_fraction'), "select_with_dynamic_fraction method missing"

    # Test that the method works
    selected = policy.select_with_dynamic_fraction(
        chunks=chunks,
        query=query,
        chunk_embeddings=chunk_embeddings,
        query_embedding=query_embedding,
        dynamic_fraction=0.3
    )

    assert isinstance(selected, list), "select_with_dynamic_fraction should return a list"
    assert len(selected) > 0, "select_with_dynamic_fraction should select at least one chunk"

    # Verify original fraction is restored
    assert policy.expansion_fraction == 0.25, "Original expansion fraction not restored"

    print(f"✓ SelectionPolicy.select_with_dynamic_fraction works (selected {len(selected)}/{len(chunks)} chunks)")
except Exception as e:
    print(f"✗ SelectionPolicy test failed: {e}")
    sys.exit(1)

# Test 5: Test HybridInputConstructor has position_embedding
print("\n[5/5] Testing HybridInputConstructor positional encodings...")
try:
    # We can't fully test this without loading models, but we can check the code
    import inspect
    source = inspect.getsource(HybridInputConstructor.__init__)

    assert 'position_embedding' in source, "position_embedding not found in __init__"
    assert 'nn.Embedding' in source, "nn.Embedding not used for position_embedding"

    # Check that construct method uses positional encodings
    construct_source = inspect.getsource(HybridInputConstructor.construct)
    assert 'position_emb' in construct_source or 'position_id' in construct_source, \
        "Positional encoding not used in construct method"

    print("✓ HybridInputConstructor has positional encoding support")
except Exception as e:
    print(f"✗ HybridInputConstructor test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All sanity checks passed!")
print("=" * 60)
