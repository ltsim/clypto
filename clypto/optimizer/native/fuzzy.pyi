from _typeshed import Incomplete

class FuzzySystem:
    """Fuzzy System for hierarchical pyramid weights"""
    pyramid_type: Incomplete
    def __init__(self, pyramid_type: str = 'increase') -> None:
        """
        Args:
            pyramid_type: 'increase' or 'decrease'
        """
    def triangular_membership(self, x, a, b, c):
        """Triangular membership function"""
    def fuzzify_iterations(self, iteration_percent):
        """Fuzzify input iterations (0-1)"""
    def defuzzify_centroid(self, membership_values):
        """Defuzzification using centroid method"""
    def get_fuzzy_weights(self, current_iteration, max_iterations):
        """Get fuzzy weights for alpha, beta, delta"""
