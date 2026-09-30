from collections import deque
from dataclasses import dataclass
import random


Position = tuple[int, int]


@dataclass(frozen=True)
class GeneratedMap:
    """
    Reproducible generated GridWorld map.
    """

    seed: int
    grid: list[str]

    shortest_path_length: int
    manhattan_distance: int

    obstacle_probability: float

    @property
    def detour_extra(self) -> int:
        return (
            self.shortest_path_length
            - self.manhattan_distance
        )


def _shortest_path_length(
    grid,
    start: Position,
    goal: Position,
):
    """
    Exact BFS shortest-path length.

    Used only during map generation to make sure
    the generated map is valid.

    This is NOT an agent-side planner.
    """

    height = len(grid)
    width = len(grid[0])

    queue = deque([
        (start, 0)
    ])

    visited = {
        start
    }

    deltas = [
        (0, -1),
        (1, 0),
        (0, 1),
        (-1, 0),
    ]

    while queue:

        position, distance = (
            queue.popleft()
        )

        if position == goal:
            return distance

        x, y = position

        for dx, dy in deltas:

            next_position = (
                x + dx,
                y + dy,
            )

            nx, ny = next_position

            if not (
                0 <= nx < width
                and
                0 <= ny < height
            ):
                continue

            if (
                grid[ny][nx]
                == "#"
            ):
                continue

            if (
                next_position
                in visited
            ):
                continue

            visited.add(
                next_position
            )

            queue.append(
                (
                    next_position,
                    distance + 1,
                )
            )

    return None


def generate_random_map(
    seed: int,
    width: int = 15,
    height: int = 9,
    obstacle_probability: float = 0.28,
    min_detour_extra: int = 2,
    max_attempts: int = 500,
) -> GeneratedMap:
    """
    Generate a deterministic random map from seed.

    Requirements:
        - outer boundary is blocked
        - start = (1, 1)
        - goal = (width - 2, height - 2)
        - S and G are connected

    Prefer maps whose true shortest path is at least
    min_detour_extra steps longer than Manhattan
    distance.

    If no such map is found, return the best reachable
    candidate found during generation.
    """

    if width < 7:
        raise ValueError(
            "width must be >= 7"
        )

    if height < 7:
        raise ValueError(
            "height must be >= 7"
        )

    if not (
        0.0
        <= obstacle_probability
        < 1.0
    ):
        raise ValueError(
            "obstacle_probability "
            "must be in [0, 1)."
        )

    start = (
        1,
        1,
    )

    goal = (
        width - 2,
        height - 2,
    )

    manhattan_distance = (
        abs(
            goal[0]
            - start[0]
        )
        +
        abs(
            goal[1]
            - start[1]
        )
    )

    best_candidate = None

    for attempt in range(
        max_attempts
    ):

        # Each attempt is deterministic for:
        #     seed + attempt
        #
        # Do not use global random state.
        rng_seed = (
            seed * 1_000_003
            + attempt * 97_409
            + 17
        )

        rng = random.Random(
            rng_seed
        )

        grid = []

        for y in range(height):

            row = []

            for x in range(width):

                # ------------------------------
                # Outer boundary
                # ------------------------------

                if (
                    x == 0
                    or y == 0
                    or x == width - 1
                    or y == height - 1
                ):
                    row.append("#")
                    continue

                # ------------------------------
                # Interior
                # ------------------------------

                if (
                    rng.random()
                    < obstacle_probability
                ):
                    row.append("#")

                else:
                    row.append(".")

            grid.append(row)

        # Force start / goal to be walkable.
        sx, sy = start
        gx, gy = goal

        grid[sy][sx] = "."
        grid[gy][gx] = "."

        shortest = (
            _shortest_path_length(
                grid=grid,
                start=start,
                goal=goal,
            )
        )

        if shortest is None:
            continue

        candidate = (
            grid,
            shortest,
        )

        # Keep the most interesting reachable map
        # seen so far as fallback.
        if (
            best_candidate is None
            or shortest
            > best_candidate[1]
        ):
            best_candidate = candidate

        detour_extra = (
            shortest
            - manhattan_distance
        )

        if (
            detour_extra
            < min_detour_extra
        ):
            continue

        grid[sy][sx] = "S"
        grid[gy][gx] = "G"

        return GeneratedMap(
            seed=seed,

            grid=[
                "".join(row)
                for row in grid
            ],

            shortest_path_length=
                shortest,

            manhattan_distance=
                manhattan_distance,

            obstacle_probability=
                obstacle_probability,
        )

    # ==================================================
    # Fallback:
    # return the best reachable map found
    # ==================================================

    if best_candidate is None:

        raise RuntimeError(
            f"Could not generate a connected "
            f"map for seed={seed}."
        )

    grid, shortest = (
        best_candidate
    )

    sx, sy = start
    gx, gy = goal

    grid[sy][sx] = "S"
    grid[gy][gx] = "G"

    return GeneratedMap(
        seed=seed,

        grid=[
            "".join(row)
            for row in grid
        ],

        shortest_path_length=
            shortest,

        manhattan_distance=
            manhattan_distance,

        obstacle_probability=
            obstacle_probability,
    )


def obstacle_probability_for_seed(
    seed: int,
) -> float:
    """
    Give generated maps several geometry densities.

    This creates context variation without
    manually designing each map.
    """

    choices = [
        0.18,
        0.22,
        0.26,
        0.30,
        0.34,
    ]

    return choices[
        seed
        % len(choices)
    ]


def main():

    generated = (
        generate_random_map(
            seed=42,

            obstacle_probability=
                obstacle_probability_for_seed(
                    42
                ),
        )
    )

    print(
        "=== Generated Map ==="
    )

    print(
        f"seed: {generated.seed}"
    )

    print(
        f"shortest path: "
        f"{generated.shortest_path_length}"
    )

    print(
        f"Manhattan: "
        f"{generated.manhattan_distance}"
    )

    print(
        f"detour extra: "
        f"{generated.detour_extra}"
    )

    print(
        f"obstacle probability: "
        f"{generated.obstacle_probability:.2f}"
    )

    print()

    for row in generated.grid:
        print(row)


if __name__ == "__main__":
    main()