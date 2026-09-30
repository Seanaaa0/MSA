import numpy as np

from src.planner import AStarPlanner


class PathCentricRepresentation:
    """
    Agent-visible route-centric representation.

    Instead of asking:

        "What does the area around me look like?"

    ask:

        "What does the route I intend to follow look like?"

    The representation follows the agent's current
    optimistic planned path for a fixed number of
    future waypoints.

    IMPORTANT:
        Uses belief only.

        Does NOT use:
            true map
            true state
            oracle action
            regret
            adequacy label
    """

    OCCUPANCY_NAMES = [
        "free",
        "wall",
        "unknown",
    ]

    STEP_FEATURE_NAMES = [
        "valid",

        "center_free",
        "center_wall",
        "center_unknown",

        "left_free",
        "left_wall",
        "left_unknown",

        "right_free",
        "right_wall",
        "right_unknown",

        "local_unknown_ratio",
        "local_wall_ratio",

        "branching_factor_norm",

        "turn_ahead",

        "goal_here",
    ]

    def __init__(
        self,
        horizon=16,
        local_radius=1,
    ):
        if horizon <= 0:
            raise ValueError(
                "horizon must be positive."
            )

        if local_radius < 0:
            raise ValueError(
                "local_radius must be >= 0."
            )

        self.horizon = int(
            horizon
        )

        self.local_radius = int(
            local_radius
        )

    # ==================================================
    # Belief helpers
    # ==================================================

    @staticmethod
    def _inside(
        belief,
        position,
    ):
        x, y = position

        return (
            0 <= x < belief.width
            and
            0 <= y < belief.height
        )

    @classmethod
    def _cell(
        cls,
        belief,
        position,
    ):
        """
        Outside-map cells are treated as walls.

        Map dimensions are assumed known to the agent.
        """

        if not cls._inside(
            belief,
            position,
        ):
            return "#"

        return belief.cell(
            position
        )

    @classmethod
    def _occupancy_one_hot(
        cls,
        belief,
        position,
    ):
        cell = cls._cell(
            belief,
            position,
        )

        if cell == ".":
            return [
                1.0,
                0.0,
                0.0,
            ]

        if cell == "#":
            return [
                0.0,
                1.0,
                0.0,
            ]

        if cell == "?":
            return [
                0.0,
                0.0,
                1.0,
            ]

        raise ValueError(
            f"Unsupported belief cell: {cell!r}"
        )

    # ==================================================
    # Path
    # ==================================================

    @staticmethod
    def optimistic_path(
        belief,
        goal,
    ):
        if (
            belief.estimated_position
            is None
        ):
            return []

        grid = belief.planning_grid(
            unknown_as_free=True
        )

        planner = AStarPlanner(
            grid
        )

        try:
            path = planner.plan(
                start=
                    belief.estimated_position,

                goal=
                    goal,
            )

        except (
            ValueError,
            RuntimeError,
        ):
            return []

        if path is None:
            return []

        return list(
            path
        )

    # ==================================================
    # Geometry
    # ==================================================

    @staticmethod
    def _direction(
        source,
        target,
    ):
        dx = (
            target[0]
            - source[0]
        )

        dy = (
            target[1]
            - source[1]
        )

        if (
            abs(dx)
            + abs(dy)
            != 1
        ):
            return (
                0,
                0,
            )

        return (
            dx,
            dy,
        )

    @staticmethod
    def _left_offset(
        direction,
    ):
        dx, dy = direction

        return (
            dy,
            -dx,
        )

    @staticmethod
    def _right_offset(
        direction,
    ):
        dx, dy = direction

        return (
            -dy,
            dx,
        )

    @staticmethod
    def _add(
        position,
        offset,
    ):
        return (
            position[0]
            + offset[0],

            position[1]
            + offset[1],
        )

    # ==================================================
    # Local structural information around a waypoint
    # ==================================================

    def _local_ratios(
        self,
        belief,
        center,
    ):
        unknown_count = 0
        wall_count = 0
        total = 0

        cx, cy = center

        for dy in range(
            -self.local_radius,
            self.local_radius + 1,
        ):
            for dx in range(
                -self.local_radius,
                self.local_radius + 1,
            ):
                position = (
                    cx + dx,
                    cy + dy,
                )

                cell = self._cell(
                    belief,
                    position,
                )

                total += 1

                if cell == "?":
                    unknown_count += 1

                elif cell == "#":
                    wall_count += 1

        if total == 0:
            return (
                0.0,
                0.0,
            )

        return (
            unknown_count / total,
            wall_count / total,
        )

    @classmethod
    def _branching_factor(
        cls,
        belief,
        position,
    ):
        """
        Optimistic branching factor.

        FREE and UNKNOWN are considered potentially
        traversable. Known WALL is not.

        Normalized to [0, 1].
        """

        x, y = position

        neighbors = [
            (x, y - 1),
            (x + 1, y),
            (x, y + 1),
            (x - 1, y),
        ]

        traversable = 0

        for neighbor in neighbors:
            cell = cls._cell(
                belief,
                neighbor,
            )

            if cell != "#":
                traversable += 1

        return (
            traversable
            / 4.0
        )

    # ==================================================
    # One route waypoint
    # ==================================================

    def _step_features(
        self,
        belief,
        path,
        path_index,
        goal,
    ):
        """
        path_index indexes the complete path.

        path[0] should be current agent position.
        Future route begins at path[1].
        """

        waypoint = path[
            path_index
        ]

        previous = path[
            path_index - 1
        ]

        direction = self._direction(
            previous,
            waypoint,
        )

        left_position = self._add(
            waypoint,
            self._left_offset(
                direction
            ),
        )

        right_position = self._add(
            waypoint,
            self._right_offset(
                direction
            ),
        )

        center_occupancy = (
            self._occupancy_one_hot(
                belief,
                waypoint,
            )
        )

        left_occupancy = (
            self._occupancy_one_hot(
                belief,
                left_position,
            )
        )

        right_occupancy = (
            self._occupancy_one_hot(
                belief,
                right_position,
            )
        )

        (
            local_unknown_ratio,
            local_wall_ratio,
        ) = self._local_ratios(
            belief,
            waypoint,
        )

        branching_factor = (
            self._branching_factor(
                belief,
                waypoint,
            )
        )

        # ----------------------------------------------
        # Does the route turn after this waypoint?
        # ----------------------------------------------

        turn_ahead = 0.0

        if (
            path_index + 1
            < len(path)
        ):
            next_direction = (
                self._direction(
                    waypoint,
                    path[
                        path_index + 1
                    ],
                )
            )

            if (
                next_direction
                != direction
            ):
                turn_ahead = 1.0

        goal_here = float(
            waypoint
            == goal
        )

        return np.asarray(
            [
                1.0,

                *center_occupancy,
                *left_occupancy,
                *right_occupancy,

                local_unknown_ratio,
                local_wall_ratio,

                branching_factor,

                turn_ahead,

                goal_here,
            ],
            dtype=np.float64,
        )

    # ==================================================
    # Public extraction
    # ==================================================

    def extract(
        self,
        belief,
        goal,
    ):
        """
        Returns flattened fixed-length vector:

            horizon
            *
            15 features per waypoint

        Default:

            16 * 15 = 240 features
        """

        path = self.optimistic_path(
            belief=belief,
            goal=goal,
        )

        per_step_feature_count = len(
            self.STEP_FEATURE_NAMES
        )

        representation = np.zeros(
            (
                self.horizon,
                per_step_feature_count,
            ),
            dtype=np.float64,
        )

        if len(path) <= 1:
            return representation.reshape(
                -1
            )

        future_count = min(
            self.horizon,
            len(path) - 1,
        )

        for future_index in range(
            future_count
        ):
            path_index = (
                future_index + 1
            )

            representation[
                future_index
            ] = self._step_features(
                belief=belief,
                path=path,
                path_index=path_index,
                goal=goal,
            )

        return representation.reshape(
            -1
        )

    # ==================================================
    # Feature names
    # ==================================================

    def feature_names(
        self,
    ):
        names = []

        for future_step in range(
            1,
            self.horizon + 1,
        ):
            for feature_name in (
                self.STEP_FEATURE_NAMES
            ):
                names.append(
                    (
                        f"path_t+{future_step:02d}_"
                        f"{feature_name}"
                    )
                )

        return names


# ======================================================
# Manual diagnostic
# ======================================================

def main():
    from src.env import GridWorld

    from src.observation import (
        LocalObserver,
    )

    from src.belief import (
        OccupancyBelief,
    )

    env = GridWorld(
        map_name="corridor"
    )

    observer = LocalObserver(
        normal_radius=1,
        query_radius=4,
    )

    belief = OccupancyBelief(
        width=env.width,
        height=env.height,
        goal_position=
            env.goal_position,
    )

    env.reset()

    belief.update(
        observer.observe(
            env
        )
    )

    encoder = (
        PathCentricRepresentation(
            horizon=16,
            local_radius=1,
        )
    )

    path = encoder.optimistic_path(
        belief=belief,
        goal=env.goal_position,
    )

    vector = encoder.extract(
        belief=belief,
        goal=env.goal_position,
    )

    matrix = vector.reshape(
        encoder.horizon,
        len(
            encoder.STEP_FEATURE_NAMES
        ),
    )

    print(
        "=== Path-Centric Level-3 Test ==="
    )

    print(
        f"Optimistic path length: "
        f"{len(path)}"
    )

    print(
        f"Horizon: "
        f"{encoder.horizon}"
    )

    print(
        f"Features per step: "
        f"{len(encoder.STEP_FEATURE_NAMES)}"
    )

    print(
        f"Flattened features: "
        f"{len(vector)}"
    )

    print()

    print(
        "step  valid  "
        "center?  left?  right?  "
        "unknown  wall  branch  "
        "turn  goal"
    )

    print(
        "-" * 78
    )

    for index in range(
        encoder.horizon
    ):
        row = matrix[
            index
        ]

        if row[0] == 0:
            continue

        center_unknown = row[3]
        left_unknown = row[6]
        right_unknown = row[9]

        print(
            f"{index + 1:>4} "
            f"{row[0]:>6.0f} "
            f"{center_unknown:>8.0f} "
            f"{left_unknown:>6.0f} "
            f"{right_unknown:>7.0f} "
            f"{row[10]:>8.3f} "
            f"{row[11]:>5.3f} "
            f"{row[12]:>7.3f} "
            f"{row[13]:>5.0f} "
            f"{row[14]:>5.0f}"
        )


if __name__ == "__main__":
    main()