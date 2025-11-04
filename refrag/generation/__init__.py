"""
Generation module for REfrag.

This module provides LLM interface and hybrid input construction for generation.
"""

from refrag.generation.hybrid_input import HybridInputConstructor
from refrag.generation.llm_interface import LLMInterface

__all__ = ["LLMInterface", "HybridInputConstructor"]
