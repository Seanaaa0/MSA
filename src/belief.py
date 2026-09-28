from dataclasses import dataclass
from typing import Tuple


Position = Tuple[int, int]


@dataclass(frozen=True)
class BeliefState:
    """
    Agent-side belief about its current state.

    Important:
        This is NOT the true environment state.

    Current V0 belief representation:
        - estimated_position

    Future versions may add:
        - uncertainty
        - confidence
        - covariance
        - hard support
        - soft distribution
    """

    estimated_position: Position


class StateEstimator:
    """
    Converts observations into an agent belief.

    V0 implementation:
        observation == true position

    This means there is currently no estimation error.

    Perturbation will be introduced later as a separate module.
    """

    def estimate(self, observation: Position) -> BeliefState:
        """
        Convert an observation into a BeliefState.

        Args:
            observation:
                observed position, represented as (x, y)

        Returns:
            BeliefState
        """

        if not self._is_valid_position(observation):
            raise ValueError(
                f"Invalid observation: {observation}"
            )

        return BeliefState(
            estimated_position=observation
        )

    @staticmethod
    def _is_valid_position(position) -> bool:
        """
        Basic structural validation.

        This does NOT check whether the position is walkable.
        That responsibility belongs to the environment/planner.
        """

        if not isinstance(position, tuple):
            return False

        if len(position) != 2:
            return False

        x, y = position

        return (
            isinstance(x, int)
            and isinstance(y, int)
        )


if __name__ == "__main__":
    estimator = StateEstimator()

    observation = (4, 3)

    belief = estimator.estimate(
        observation
    )

    print("=== Belief Test ===")
    print(f"Observation: {observation}")
    print(
        f"Estimated position: "
        f"{belief.estimated_position}"
    )