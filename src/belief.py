from dataclasses import dataclass
from typing import Tuple


# ======================================================
# Shared types
# ======================================================

Position = Tuple[int, int]


# ======================================================
# Legacy V1 / V2 belief
# ======================================================

@dataclass(frozen=True)
class BeliefState:
    """
    Agent-side belief about its current position.

    This is used by the older controlled-perturbation
    experiments.

    Important:
        This is NOT the true environment state.

    Current representation:
        - estimated_position
    """

    estimated_position: Position


class StateEstimator:
    """
    Legacy exact-position estimator.

    V1/V2 implementation:
        observation == estimated position

    Controlled error is added separately by
    perturbation.py.
    """

    def estimate(
        self,
        observation: Position,
    ) -> BeliefState:

        if not self._is_valid_position(
            observation
        ):
            raise ValueError(
                f"Invalid observation: "
                f"{observation}"
            )

        return BeliefState(
            estimated_position=observation
        )

    @staticmethod
    def _is_valid_position(
        position,
    ) -> bool:

        if not isinstance(
            position,
            tuple,
        ):
            return False

        if len(position) != 2:
            return False

        x, y = position

        return (
            isinstance(x, int)
            and isinstance(y, int)
        )


# ======================================================
# Level-2 partial world belief
# ======================================================

UNKNOWN = "?"
FREE = "."
WALL = "#"


class OccupancyBelief:
    """
    Level-2 agent-side world belief.

    Unlike BeliefState, this represents
    partial knowledge of the MAP itself.

    The agent knows:
        - map dimensions
        - its own current position
        - task goal

    The agent does NOT initially know:
        - obstacle layout

    Unknown cells are represented by:
        ?
    """

    def __init__(
        self,
        width: int,
        height: int,
        goal_position: Position,
    ):
        self.width = width
        self.height = height

        self.goal_position = (
            goal_position
        )

        self.estimated_position = None

        self.grid = [
            [
                UNKNOWN
                for _ in range(width)
            ]
            for _ in range(height)
        ]

    # ==================================================
    # Observation update
    # ==================================================

    def update(
        self,
        observation,
    ):
        """
        Merge a LocalObservation into memory.

        observation.position:
            agent's current estimated position

        observation.cells:
            newly observed map cells
        """

        self.estimated_position = (
            observation.position
        )

        for position, value in (
            observation.cells.items()
        ):

            x, y = position

            self.grid[y][x] = value

    # ==================================================
    # Grid access
    # ==================================================

    def is_inside(
        self,
        position: Position,
    ) -> bool:

        x, y = position

        return (
            0 <= x < self.width
            and
            0 <= y < self.height
        )

    def cell(
        self,
        position: Position,
    ) -> str:

        if not self.is_inside(
            position
        ):
            return WALL

        x, y = position

        return self.grid[y][x]

    def is_unknown(
        self,
        position: Position,
    ) -> bool:

        return (
            self.cell(position)
            == UNKNOWN
        )

    # ==================================================
    # Uncertainty helpers
    # ==================================================

    def unknown_positions(
        self,
        center=None,
        radius=None,
    ):
        """
        Return currently unknown cells.

        Optionally restrict them to a
        Manhattan radius around center.
        """

        positions = []

        for y in range(
            self.height
        ):

            for x in range(
                self.width
            ):

                if (
                    self.grid[y][x]
                    != UNKNOWN
                ):
                    continue

                position = (
                    x,
                    y,
                )

                if (
                    center is not None
                    and radius is not None
                ):

                    distance = (
                        abs(
                            x
                            - center[0]
                        )
                        +
                        abs(
                            y
                            - center[1]
                        )
                    )

                    if (
                        distance
                        > radius
                    ):
                        continue

                positions.append(
                    position
                )

        return positions

    def unknown_fraction(
        self,
    ) -> float:
        """
        Fraction of map cells that remain unknown.
        """

        total = (
            self.width
            * self.height
        )

        unknown = sum(
            cell == UNKNOWN
            for row in self.grid
            for cell in row
        )

        return (
            unknown / total
            if total
            else 0.0
        )

    # ==================================================
    # Planner interface
    # ==================================================

    def planning_grid(
        self,
        unknown_as_free=True,
    ):
        """
        Convert current belief into a grid that
        can be passed to AStarPlanner.

        Level-2 default:
            optimistic planning

        UNKNOWN cells are treated as free
        until evidence shows otherwise.

        Later experiments can compare against:
            unknown_as_free=False
        """

        planning_grid = []

        for row in self.grid:

            planning_row = []

            for cell in row:

                if (
                    cell == UNKNOWN
                ):

                    planning_row.append(
                        FREE
                        if unknown_as_free
                        else WALL
                    )

                else:

                    planning_row.append(
                        cell
                    )

            planning_grid.append(
                planning_row
            )

        return planning_grid


# ======================================================
# Manual tests
# ======================================================

def main():

    # ----------------------------------------------
    # Legacy belief
    # ----------------------------------------------

    estimator = (
        StateEstimator()
    )

    belief = estimator.estimate(
        (4, 3)
    )

    print(
        "=== Legacy Belief Test ==="
    )

    print(
        f"Estimated position: "
        f"{belief.estimated_position}"
    )

    print()

    # ----------------------------------------------
    # Level-2 belief
    # ----------------------------------------------

    occupancy = (
        OccupancyBelief(
            width=10,
            height=7,
            goal_position=(8, 5),
        )
    )

    print(
        "=== Level-2 Occupancy "
        "Belief Test ==="
    )

    print(
        f"Goal: "
        f"{occupancy.goal_position}"
    )

    print(
        f"Unknown fraction: "
        f"{occupancy.unknown_fraction():.3f}"
    )

    print(
        f"Unknown cells: "
        f"{len(occupancy.unknown_positions())}"
    )


if __name__ == "__main__":
    main()