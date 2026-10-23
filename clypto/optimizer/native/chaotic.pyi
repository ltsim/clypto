class ChaoticMap:
    """
    Implementation of 10 chaotic maps
    """
    @staticmethod
    def bernoulli_map(x: float, a: float = 0.5) -> float:
        """Bernoulli map"""
    @staticmethod
    def logistic_map(x: float, a: float = 4.0) -> float:
        """Logistic map"""
    @staticmethod
    def chebyshev_map(x: float, a: float = 4.0) -> float:
        """Chebyshev map"""
    @staticmethod
    def circle_map(x: float, a: float = 0.5, b: float = 0.2) -> float:
        """Circle map"""
    @staticmethod
    def cubic_map(x: float, q: float = 2.59) -> float:
        """Cubic map"""
    @staticmethod
    def icmic_map(x: float, a: float = 0.7) -> float:
        """Iterative chaotic map with infinite collapses"""
    @staticmethod
    def piecewise_map(x: float, a: float = 0.4) -> float:
        """Piecewise map"""
    @staticmethod
    def singer_map(x: float, a: float = 1.07) -> float:
        """Singer map"""
    @staticmethod
    def sinusoidal_map(x: float, a: float = 2.3) -> float:
        """Sinusoidal map"""
    @staticmethod
    def tent_map(x: float) -> float:
        """Tent map"""
