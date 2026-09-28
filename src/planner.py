import heapq

from src.env import Action


class AStarPlanner:
    """
    Deterministic A* path planner for GridWorld.

    The planner only knows the map geometry.
    It does NOT modify the environment or move the agent.

    Coordinate convention:
        position = (x, y)

        x increases to the right
        y increases downward
    """

    ACTION_DELTAS = {
        Action.UP: (0, -1),
        Action.DOWN: (0, 1),
        Action.LEFT: (-1, 0),
        Action.RIGHT: (1, 0),
    }

    DELTA_TO_ACTION = {
        (0, -1): Action.UP,
        (0, 1): Action.DOWN,
        (-1, 0): Action.LEFT,
        (1, 0): Action.RIGHT,
    }

    def __init__(self, grid):
        """
        Args:
            grid:
                2D map represented as either:

                [
                    ['#', '#', '#'],
                    ['#', 'S', '#'],
                    ...
                ]

                or:

                [
                    "#####",
                    "#S..#",
                    ...
                ]
        """

        if not grid:
            raise ValueError("Grid cannot be empty.")

        self.grid = [list(row) for row in grid]

        self.height = len(self.grid)
        self.width = len(self.grid[0])

        self._validate_grid()

    def _validate_grid(self):
        """Check basic grid consistency."""

        expected_width = self.width

        for row in self.grid:
            if len(row) != expected_width:
                raise ValueError(
                    "All grid rows must have the same width."
                )

    def _is_inside(self, position):
        """Return True if position is inside map bounds."""

        x, y = position

        return (
            0 <= x < self.width
            and 0 <= y < self.height
        )

    def _is_walkable(self, position):
        """
        Return True if a position can be occupied.

        Any cell except '#' is considered walkable.
        """

        if not self._is_inside(position):
            return False

        x, y = position

        return self.grid[y][x] != "#"

    def _heuristic(self, position, goal):
        """
        Manhattan distance heuristic.

        Appropriate because GridWorld only allows
        four-direction movement.
        """

        x1, y1 = position
        x2, y2 = goal

        return abs(x1 - x2) + abs(y1 - y2)

    def _neighbors(self, position):
        """
        Return valid neighboring positions.

        Neighbor ordering is deterministic.
        """

        x, y = position

        neighbors = []

        # Fixed ordering is important for reproducible experiments.
        ordered_actions = [
            Action.UP,
            Action.RIGHT,
            Action.DOWN,
            Action.LEFT,
        ]

        for action in ordered_actions:
            dx, dy = self.ACTION_DELTAS[action]

            next_position = (
                x + dx,
                y + dy,
            )

            if self._is_walkable(next_position):
                neighbors.append(next_position)

        return neighbors

    def _reconstruct_path(self, came_from, current):
        """
        Reconstruct path from start to current node.

        Returns:
            list[tuple[int, int]]
        """

        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()

        return path

    def plan(self, start, goal):
        """
        Find the shortest path from start to goal using A*.

        Args:
            start:
                (x, y) starting position

            goal:
                (x, y) goal position

        Returns:
            List of positions including both start and goal.

            Example:
                [
                    (1, 1),
                    (2, 1),
                    (3, 1),
                ]

            Returns [] if no path exists.
        """

        if not self._is_walkable(start):
            raise ValueError(
                f"Start position is invalid or blocked: {start}"
            )

        if not self._is_walkable(goal):
            raise ValueError(
                f"Goal position is invalid or blocked: {goal}"
            )

        if start == goal:
            return [start]

        # Priority queue entries:
        # (f_score, g_score, position)
        open_heap = []

        start_g = 0
        start_f = self._heuristic(start, goal)

        heapq.heappush(
            open_heap,
            (start_f, start_g, start),
        )

        came_from = {}

        g_score = {
            start: 0
        }

        closed_set = set()

        while open_heap:
            _, current_g, current = heapq.heappop(
                open_heap
            )

            if current in closed_set:
                continue

            if current == goal:
                return self._reconstruct_path(
                    came_from,
                    current,
                )

            closed_set.add(current)

            for neighbor in self._neighbors(current):

                if neighbor in closed_set:
                    continue

                tentative_g = current_g + 1

                previous_g = g_score.get(
                    neighbor,
                    float("inf"),
                )

                if tentative_g < previous_g:

                    came_from[neighbor] = current

                    g_score[neighbor] = tentative_g

                    f_score = (
                        tentative_g
                        + self._heuristic(
                            neighbor,
                            goal,
                        )
                    )

                    heapq.heappush(
                        open_heap,
                        (
                            f_score,
                            tentative_g,
                            neighbor,
                        ),
                    )

        # No route exists.
        return []

    def next_action(self, start, goal):
        """
        Return the first action along the optimal path.

        Returns:
            Action
            or None if start == goal

        Raises:
            RuntimeError if no path exists.
        """

        path = self.plan(
            start=start,
            goal=goal,
        )

        if not path:
            raise RuntimeError(
                f"No path exists from {start} to {goal}."
            )

        if len(path) == 1:
            return None

        current = path[0]
        next_position = path[1]

        dx = next_position[0] - current[0]
        dy = next_position[1] - current[1]

        delta = (dx, dy)

        action = self.DELTA_TO_ACTION.get(delta)

        if action is None:
            raise RuntimeError(
                f"Invalid path transition: "
                f"{current} -> {next_position}"
            )

        return action


if __name__ == "__main__":
    from src.env import GridWorld

    env = GridWorld()

    planner = AStarPlanner(
        env.grid
    )

    start = env.agent_position
    goal = env.goal_position

    path = planner.plan(
        start=start,
        goal=goal,
    )

    print("=== A* Planner Test ===")
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"Path length: {len(path) - 1}")
    print(f"Path: {path}")

    action = planner.next_action(
        start=start,
        goal=goal,
    )

    if action is not None:
        print(
            f"First action: {action.value}"
        )
    else:
        print("Already at goal.")