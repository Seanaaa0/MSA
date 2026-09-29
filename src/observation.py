from dataclasses import dataclass
from typing import Dict, Tuple


Position = Tuple[int, int]


@dataclass(frozen=True)
class LocalObservation:
    """
    Agent-visible local observation.

    The observer knows the true environment,
    but the agent receives ONLY these revealed cells.
    """

    position: Position
    cells: Dict[Position, str]
    radius: int


class LocalObserver:
    """
    Partial observation model.

    Normal sensing:
        small radius

    QueryReality:
        larger radius

    Current V0:
        Manhattan-radius visibility.
        No image model / learned perception yet.
    """

    def __init__(
        self,
        normal_radius: int = 1,
        query_radius: int = 3,
    ):
        if normal_radius < 0:
            raise ValueError(
                "normal_radius must be non-negative."
            )

        if query_radius < normal_radius:
            raise ValueError(
                "query_radius must be >= normal_radius."
            )

        self.normal_radius = normal_radius
        self.query_radius = query_radius

    def _observe(
        self,
        env,
        radius: int,
    ) -> LocalObservation:

        cx, cy = env.agent_position

        cells = {}

        for y in range(env.height):
            for x in range(env.width):

                distance = (
                    abs(x - cx)
                    + abs(y - cy)
                )

                if distance > radius:
                    continue

                symbol = env.grid[y][x]

                # Agent only needs occupancy here.
                if symbol == "#":
                    observed = "#"
                else:
                    observed = "."

                cells[(x, y)] = observed

        return LocalObservation(
            position=env.agent_position,
            cells=cells,
            radius=radius,
        )

    def observe(self, env):
        """
        Cheap/default observation.
        """

        return self._observe(
            env=env,
            radius=self.normal_radius,
        )

    def query(self, env):
        """
        More expensive / wider observation.
        """

        return self._observe(
            env=env,
            radius=self.query_radius,
        )