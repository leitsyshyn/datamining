"""Collaborative-filtering recommender for BoardGameGeek ratings.

    config       every decision of the experiment
    data         loading and cleaning the snapshots, comparing them, the experimental frame
    models       Surprise rating models, list models, and the catalogue that names and selects them
    tuning       hyperparameter search (cached)
    metrics      metric definitions
    evaluation   evaluation protocols: both test sets, full-catalogue ranking, sub-groups, robustness
    analysis     descriptive tables for the EDA and the interpretation
    report       chart theme, figures and table formatting
"""
