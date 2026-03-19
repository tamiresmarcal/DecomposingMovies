"""
cinematic_surprise
==================
Per-second hierarchical Bayesian surprise and uncertainty measurement in movies.

References
----------
Itti, L., & Baldi, P. (2009). Bayesian surprise attracts human attention.
    Vision Research, 49(10), 1295-1306.

Cheung, V.K.M., et al. (2019). Uncertainty and Surprise Jointly Predict
    Musical Pleasure and Amygdala, Hippocampus, and Auditory Cortex Activity.
    Current Biology, 29, 4084-4092.

Yamins, D.L., & DiCarlo, J.J. (2016). Using goal-driven deep learning
    models to understand sensory cortex. Nature Neuroscience, 19(3), 356-365.
"""

from cinematic_surprise.pipeline import CinematicSurprisePipeline
from cinematic_surprise.uncertainty_and_surprise.estimator import OnlineGaussianEstimator

__version__ = "0.2.0"
__all__ = ["CinematicSurprisePipeline", "OnlineGaussianEstimator"]
