import numpy as np

from src.planner import (
    AStarPlanner,
)


# ======================================================
# Spatial representation
# ======================================================

CHANNEL_NAMES = [
    "known_free",
    "known_wall",
    "unknown",
    "planned_path",
    "goal",
]


class SpatialBeliefPatch:
    """
    Agent-visible structured Level-3 representation.

    Instead of compressing the belief into scalar
    summary statistics, preserve WHERE information
    appears relative to the agent.

    Default representation:

        9 x 9 local patch

    Channels:

        0. known_free
        1. known_wall
        2. unknown
        3. planned_path
        4. goal

    IMPORTANT:

        Agent is always at the center of the patch.

        No true map is used.
        No true state is used.
        No oracle action is used.
        No regret is used.
        No adequacy label is used.

    Everything comes from the current agent belief.
    """

    def __init__(
        self,
        patch_size=9,
    ):

        if patch_size <= 0:

            raise ValueError(
                "patch_size must be positive."
            )

        if (
            patch_size
            % 2
            == 0
        ):

            raise ValueError(
                "patch_size must be odd."
            )

        self.patch_size = int(
            patch_size
        )

        self.radius = (
            self.patch_size
            // 2
        )

        self.channel_names = list(
            CHANNEL_NAMES
        )

    # ==================================================
    # Optimistic plan
    # ==================================================

    @staticmethod
    def _optimistic_path(
        belief,
        goal,
    ):
        """
        Agent-side optimistic plan.

        UNKNOWN cells are treated as free,
        exactly like the current Level-2/3 planner.
        """

        if (
            belief.estimated_position
            is None
        ):

            return []

        grid = (
            belief.planning_grid(
                unknown_as_free=True
            )
        )

        planner = AStarPlanner(
            grid
        )

        try:

            return planner.plan(
                start=
                    belief.estimated_position,

                goal=goal,
            )

        except (
            ValueError,
            RuntimeError,
        ):

            return []

    # ==================================================
    # Extract patch
    # ==================================================

    def extract(
        self,
        belief,
        goal,
    ):
        """
        Returns:

            np.ndarray

            shape:

                (
                    5,
                    patch_size,
                    patch_size
                )

        Coordinate frame:

            agent is always at patch center.

        Therefore spatial coordinates are relative
        to the current decision point.
        """

        current = (
            belief.estimated_position
        )

        if current is None:

            raise ValueError(
                "Belief has no estimated position."
            )

        patch = np.zeros(
            (
                len(
                    self.channel_names
                ),
                self.patch_size,
                self.patch_size,
            ),
            dtype=np.float64,
        )

        path = (
            self._optimistic_path(
                belief=belief,
                goal=goal,
            )
        )

        # Exclude current position.
        planned_positions = set(
            path[
                1:
            ]
        )

        center_x, center_y = (
            current
        )

        for patch_y in range(
            self.patch_size
        ):

            for patch_x in range(
                self.patch_size
            ):

                dx = (
                    patch_x
                    - self.radius
                )

                dy = (
                    patch_y
                    - self.radius
                )

                world_position = (
                    center_x + dx,
                    center_y + dy,
                )

                world_x, world_y = (
                    world_position
                )

                # ======================================
                # Known map boundary
                #
                # The agent already knows map dimensions.
                # Outside the map is non-navigable,
                # so encode it as wall.
                # ======================================

                if not (
                    0
                    <= world_x
                    < belief.width
                    and
                    0
                    <= world_y
                    < belief.height
                ):

                    patch[
                        1,
                        patch_y,
                        patch_x,
                    ] = 1.0

                    continue

                cell = (
                    belief.cell(
                        world_position
                    )
                )

                # ======================================
                # Occupancy channels
                # ======================================

                if cell == ".":

                    patch[
                        0,
                        patch_y,
                        patch_x,
                    ] = 1.0

                elif cell == "#":

                    patch[
                        1,
                        patch_y,
                        patch_x,
                    ] = 1.0

                elif cell == "?":

                    patch[
                        2,
                        patch_y,
                        patch_x,
                    ] = 1.0

                else:

                    raise ValueError(
                        f"Unsupported belief cell: "
                        f"{cell!r}"
                    )

                # ======================================
                # Current optimistic plan
                # ======================================

                if (
                    world_position
                    in planned_positions
                ):

                    patch[
                        3,
                        patch_y,
                        patch_x,
                    ] = 1.0

                # ======================================
                # Known task goal
                # ======================================

                if (
                    world_position
                    == goal
                ):

                    patch[
                        4,
                        patch_y,
                        patch_x,
                    ] = 1.0

        return patch

    # ==================================================
    # Flattened interface
    # ==================================================

    def flatten(
        self,
        patch,
    ):
        """
        Flatten while preserving fixed relative
        spatial positions.

        A linear model can therefore learn different
        weights for:

            unknown 1 cell ahead

        versus:

            unknown 4 cells behind

        even though both contribute equally to a
        scalar unknown ratio.
        """

        expected_shape = (
            len(
                self.channel_names
            ),
            self.patch_size,
            self.patch_size,
        )

        if (
            patch.shape
            != expected_shape
        ):

            raise ValueError(
                f"Expected patch shape "
                f"{expected_shape}, "
                f"got {patch.shape}."
            )

        return (
            patch.reshape(
                -1
            )
        )

    # ==================================================
    # Feature names
    # ==================================================

    def feature_names(
        self,
    ):
        """
        Stable names for flattened patch features.

        Example:

            spatial_unknown_dx+1_dy+0
        """

        names = []

        for channel_name in (
            self.channel_names
        ):

            for patch_y in range(
                self.patch_size
            ):

                for patch_x in range(
                    self.patch_size
                ):

                    dx = (
                        patch_x
                        - self.radius
                    )

                    dy = (
                        patch_y
                        - self.radius
                    )

                    names.append(
                        (
                            f"spatial_"
                            f"{channel_name}_"
                            f"dx{dx:+d}_"
                            f"dy{dy:+d}"
                        )
                    )

        return names


# ======================================================
# Manual test
# ======================================================

def main():

    from src.env import (
        GridWorld,
    )

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
        width=
            env.width,

        height=
            env.height,

        goal_position=
            env.goal_position,
    )

    belief.update(
        observer.observe(
            env
        )
    )

    encoder = (
        SpatialBeliefPatch(
            patch_size=9
        )
    )

    patch = (
        encoder.extract(
            belief=belief,
            goal=
                env.goal_position,
        )
    )

    print(
        "=== Spatial Level-3 Test ==="
    )

    print(
        f"Patch shape: "
        f"{patch.shape}"
    )

    print(
        f"Flattened features: "
        f"{len(encoder.flatten(patch))}"
    )

    print()

    for (
        index,
        channel_name,
    ) in enumerate(
        encoder.channel_names
    ):

        print(
            f"--- {channel_name} ---"
        )

        channel = (
            patch[
                index
            ]
        )

        for row in channel:

            print(
                "".join(
                    "1"
                    if value > 0.5
                    else "."
                    for value
                    in row
                )
            )

        print()


if __name__ == "__main__":
    main()