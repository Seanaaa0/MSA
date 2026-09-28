from dataclasses import dataclass

from src.belief import BeliefState


@dataclass(frozen=True)
class PositionOffset:
    """
    Deterministic position perturbation.

    Example:
        dx = 2
        dy = 0

    transforms:

        (4, 3) -> (6, 3)

    Important:
        This modifies the agent belief only.
        It does NOT modify the true environment state.
    """

    dx: int = 0
    dy: int = 0


class BeliefPerturbation:
    """
    Applies controlled error to a BeliefState.

    This module belongs to the experiment apparatus,
    not to the real agent itself.

    Its purpose is to generate reproducible belief errors.
    """

    def __init__(self, offset: PositionOffset):
        self.offset = offset

    def apply(self, belief: BeliefState) -> BeliefState:
        """
        Apply deterministic positional error.

        Args:
            belief:
                Clean BeliefState.

        Returns:
            A NEW perturbed BeliefState.
        """

        x, y = belief.estimated_position

        perturbed_position = (
            x + self.offset.dx,
            y + self.offset.dy,
        )

        return BeliefState(
            estimated_position=perturbed_position
        )


if __name__ == "__main__":
    clean_belief = BeliefState(
        estimated_position=(4, 3)
    )

    perturbation = BeliefPerturbation(
        PositionOffset(
            dx=2,
            dy=0,
        )
    )

    perturbed_belief = perturbation.apply(
        clean_belief
    )

    print("=== Perturbation Test ===")
    print(
        f"Clean belief: "
        f"{clean_belief.estimated_position}"
    )
    print(
        f"Offset: "
        f"({perturbation.offset.dx}, "
        f"{perturbation.offset.dy})"
    )
    print(
        f"Perturbed belief: "
        f"{perturbed_belief.estimated_position}"
    )