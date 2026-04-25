"""
Resonance Transformer - Synthetic Data Generator
================================================

Modules for generating synthetic formal language data to train the
Resonance Transformer in its dual-representation (semantic + phase) regime.

Modules
-------
lambda_generator       : Well-typed lambda calculus term generation
proof_walk_generator     : Proof walk dataset for contrastive groups
mutation_engine          : Semantic-preserving bisimilar mutations
contrastive_trainer      : Training harness with group-based contrastive loss

Usage
-----rom synthetic import lambda_generator, proof_walk_generator

See scripts/train_synthetic.py for the full training pipeline.
"""

__version__ = "0.1.0"

from .lambda_generator import (
    LambdaGenerator,
    Term,
    Var,
    Abs,
    App,
    ArrowType,
    BaseType,
    T_INT,
    T_BOOL,
    T_UNIT,
    BASE_TYPES,
    typecheck,
    is_well_typed,
    normal_form,
    enumerate_terms,
    enumerate_all,
)

from .proof_walk_generator import (
    ProofWalkDataset,
    CrossGroupSampler,
    Statement,
    ProofStrategy,
    DEFAULT_STRATEGIES,
)

from .mutation_engine import (
    BisimilarMutation,
    MutatedProgram,
    MutationRecord,
    training_schedule,
    MUTATION_REGISTRY,
)

from .contrastive_trainer import (
    ContrastiveGroupLoss,
    DualContrastiveLoss,
    ProgramTokenizer,
    MinimalResonanceTransformer,
    TrainingConfig,
    train_contrastive,
)

__all__ = [
    # Lambda generator
    "LambdaGenerator",
    "Term",
    "Var",
    "Abs",
    "App",
    "ArrowType",
    "BaseType",
    "T_INT",
    "T_BOOL",
    "T_UNIT",
    "BASE_TYPES",
    "typecheck",
    "is_well_typed",
    "normal_form",
    "enumerate_terms",
    "enumerate_all",
    # Proof walks
    "ProofWalkDataset",
    "CrossGroupSampler",
    "Statement",
    "ProofStrategy",
    "DEFAULT_STRATEGIES",
    # Mutations
    "BisimilarMutation",
    "MutatedProgram",
    "MutationRecord",
    "training_schedule",
    "MUTATION_REGISTRY",
    # Trainer
    "ContrastiveGroupLoss",
    "DualContrastiveLoss",
    "ProgramTokenizer",
    "MinimalResonanceTransformer",
    "TrainingConfig",
    "train_contrastive",
]
