from enum import Enum


class Action(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class GridWorld:
    """
    Deterministic 2D Grid World.

    Symbols:
        # : obstacle
        . : free space
        S : start
        G : goal

    Coordinate convention:
        position = (x, y)

        x increases to the right
        y increases downward
    """

    DEFAULT_MAP = [
        "##########",
        "#S.......#",
        "#..###...#",
        "#........#",
        "#....###.#",
        "#.......G#",
        "##########",
    ]

    ACTION_DELTAS = {
        Action.UP: (0, -1),
        Action.DOWN: (0, 1),
        Action.LEFT: (-1, 0),
        Action.RIGHT: (1, 0),
    }

    def __init__(self, grid_map=None):
        if grid_map is None:
            grid_map = self.DEFAULT_MAP

        self.grid = [list(row) for row in grid_map]

        self.height = len(self.grid)
        self.width = len(self.grid[0])

        self._validate_map()

        self.start_position = self._find_symbol("S")
        self.goal_position = self._find_symbol("G")

        self.agent_position = None
        self.steps = 0
        self.done = False

        self.reset()

    def _validate_map(self):
        """Basic map sanity checks."""

        if not self.grid:
            raise ValueError("Map cannot be empty.")

        expected_width = len(self.grid[0])

        for row in self.grid:
            if len(row) != expected_width:
                raise ValueError("All map rows must have the same width.")

        flat_map = [cell for row in self.grid for cell in row]

        if flat_map.count("S") != 1:
            raise ValueError("Map must contain exactly one start 'S'.")

        if flat_map.count("G") != 1:
            raise ValueError("Map must contain exactly one goal 'G'.")

    def _find_symbol(self, symbol):
        """Return the (x, y) position of a map symbol."""

        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == symbol:
                    return (x, y)

        raise ValueError(f"Symbol {symbol!r} not found.")

    def reset(self):
        """
        Reset the environment to the initial state.

        Returns:
            tuple[int, int]: starting position
        """

        self.agent_position = self.start_position
        self.steps = 0
        self.done = False

        return self.agent_position

    def is_obstacle(self, position):
        """Check whether a position is outside the map or blocked."""

        x, y = position

        if x < 0 or x >= self.width:
            return True

        if y < 0 or y >= self.height:
            return True

        return self.grid[y][x] == "#"

    def step(self, action):
        """
        Execute one action.

        Args:
            action (Action): movement action

        Returns:
            dict: information about the transition
        """

        if self.done:
            raise RuntimeError(
                "Episode is already finished. Call reset() before stepping again."
            )

        if action not in self.ACTION_DELTAS:
            raise ValueError(f"Invalid action: {action}")

        current_x, current_y = self.agent_position
        dx, dy = self.ACTION_DELTAS[action]

        candidate_position = (
            current_x + dx,
            current_y + dy,
        )

        collision = self.is_obstacle(candidate_position)

        if not collision:
            self.agent_position = candidate_position

        self.steps += 1

        reached_goal = self.agent_position == self.goal_position

        if reached_goal:
            self.done = True

        return {
            "position": self.agent_position,
            "collision": collision,
            "reached_goal": reached_goal,
            "done": self.done,
            "steps": self.steps,
        }

    def render_ascii(self):
        """Print the current world state to the terminal."""

        display = [row.copy() for row in self.grid]

        # Replace S with normal free space after the episode begins.
        start_x, start_y = self.start_position
        display[start_y][start_x] = "."

        agent_x, agent_y = self.agent_position

        if self.agent_position != self.goal_position:
            display[agent_y][agent_x] = "A"

        print()

        for row in display:
            print("".join(row))

        print(
            f"\nposition={self.agent_position} "
            f"goal={self.goal_position} "
            f"steps={self.steps}"
        )


if __name__ == "__main__":
    env = GridWorld()

    env.render_ascii()

    test_actions = [
        Action.RIGHT,
        Action.RIGHT,
        Action.DOWN,
    ]

    for action in test_actions:
        print(f"\nAction: {action.value}")

        result = env.step(action)

        print(result)
        env.render_ascii()
