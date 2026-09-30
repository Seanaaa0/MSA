from dataclasses import dataclass

from src.env import Action
from src.planner import AStarPlanner


@dataclass(frozen=True)
class Level3FeatureVector:
    """
    Agent-visible Level-3 feature vector.

    No experiment-side ground truth is allowed here.
    """

    values: dict[str, float]

    def as_dict(self):
        return dict(
            self.values
        )


class Level3FeatureExtractor:
    """
    Extract decision-context features visible
    to the agent.

    Feature groups:

        uncertainty:
            - action instability
            - path uncertainty
            - near-path uncertainty
            - local uncertainty
            - global uncertainty

        history:
            - previous path uncertainty
            - uncertainty change
            - EMA uncertainty

        context:
            - distance to goal
            - optimistic path length
            - known wall density
            - local branching
            - candidate action

    IMPORTANT:

        No true map.
        No true state.
        No oracle action.
        No actual regret.
        No adequacy label.
    """

    ACTION_DELTAS = {
        Action.UP: (0, -1),
        Action.DOWN: (0, 1),
        Action.LEFT: (-1, 0),
        Action.RIGHT: (1, 0),
    }

    def __init__(
        self,
        sensitivity_radius=4,
        near_horizon=4,
        local_radius=3,
        ema_alpha=0.4,
    ):

        self.sensitivity_radius = (
            sensitivity_radius
        )

        self.near_horizon = (
            near_horizon
        )

        self.local_radius = (
            local_radius
        )

        self.ema_alpha = (
            ema_alpha
        )

        self.reset()

    def reset(self):

        self._previous_path_unknown = (
            None
        )

        self._ema_path_unknown = (
            None
        )

    # ==================================================
    # Agent-side optimistic plan
    # ==================================================

    @staticmethod
    def _optimistic_path(
        belief,
        goal,
    ):

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
    # Path uncertainty
    # ==================================================

    def _path_features(
        self,
        belief,
        path,
        goal,
    ):

        if not path:

            return {
                "path_length": 0,
                "path_unknown_ratio": 1.0,
                "near_unknown_ratio": 1.0,
            }

        future_path = (
            path[1:]
        )

        relevant_path = [
            position
            for position in future_path
            if position != goal
        ]

        if relevant_path:

            path_unknown = sum(
                belief.is_unknown(
                    position
                )
                for position
                in relevant_path
            )

            path_unknown_ratio = (
                path_unknown
                / len(relevant_path)
            )

        else:

            path_unknown_ratio = 0.0

        near_path = (
            future_path[
                :self.near_horizon
            ]
        )

        relevant_near = [
            position
            for position in near_path
            if position != goal
        ]

        if relevant_near:

            near_unknown = sum(
                belief.is_unknown(
                    position
                )
                for position
                in relevant_near
            )

            near_unknown_ratio = (
                near_unknown
                / len(relevant_near)
            )

        else:

            near_unknown_ratio = 0.0

        return {
            "path_length":
                len(future_path),

            "path_unknown_ratio":
                path_unknown_ratio,

            "near_unknown_ratio":
                near_unknown_ratio,
        }

    # ==================================================
    # Local context
    # ==================================================

    def _local_features(
        self,
        belief,
    ):

        cx, cy = (
            belief.estimated_position
        )

        total = 0
        unknown = 0
        known_walls = 0

        for y in range(
            belief.height
        ):

            for x in range(
                belief.width
            ):

                distance = (
                    abs(x - cx)
                    + abs(y - cy)
                )

                if (
                    distance
                    > self.local_radius
                ):
                    continue

                total += 1

                value = (
                    belief.cell(
                        (x, y)
                    )
                )

                if value == "?":
                    unknown += 1

                elif value == "#":
                    known_walls += 1

        if total == 0:

            return {
                "local_unknown_ratio":
                    0.0,

                "local_wall_ratio":
                    0.0,
            }

        return {
            "local_unknown_ratio":
                unknown / total,

            "local_wall_ratio":
                known_walls / total,
        }

    # ==================================================
    # Action stability
    # ==================================================

    def _action_stability(
        self,
        belief,
        candidate_action,
        goal,
    ):

        current = (
            belief.estimated_position
        )

        unknowns = (
            belief.unknown_positions(
                center=current,
                radius=
                    self.sensitivity_radius,
            )
        )

        tested = 0
        changed = 0

        for position in unknowns:

            if (
                position == current
                or position == goal
            ):
                continue

            grid = (
                belief.planning_grid(
                    unknown_as_free=True
                )
            )

            x, y = position

            grid[y][x] = "#"

            planner = AStarPlanner(
                grid
            )

            tested += 1

            try:

                alternative_action = (
                    planner.next_action(
                        start=current,
                        goal=goal,
                    )
                )

            except (
                ValueError,
                RuntimeError,
            ):

                changed += 1
                continue

            if (
                alternative_action
                != candidate_action
            ):
                changed += 1

        if tested == 0:
            return 1.0

        return (
            (tested - changed)
            / tested
        )

    # ==================================================
    # Branching context
    # ==================================================

    def _branching_factor(
        self,
        belief,
    ):

        grid = (
            belief.planning_grid(
                unknown_as_free=True
            )
        )

        x, y = (
            belief.estimated_position
        )

        count = 0

        for dx, dy in (
            self.ACTION_DELTAS.values()
        ):

            nx = x + dx
            ny = y + dy

            if not (
                0 <= nx < belief.width
                and
                0 <= ny < belief.height
            ):
                continue

            if (
                grid[ny][nx]
                != "#"
            ):
                count += 1

        return (
            count / 4.0
        )

    # ==================================================
    # Main feature extraction
    # ==================================================

    def extract(
        self,
        belief,
        candidate_action,
        goal,
    ) -> Level3FeatureVector:

        current = (
            belief.estimated_position
        )

        if current is None:

            raise ValueError(
                "Belief has no estimated position."
            )

        path = (
            self._optimistic_path(
                belief=belief,
                goal=goal,
            )
        )

        path_features = (
            self._path_features(
                belief=belief,
                path=path,
                goal=goal,
            )
        )

        local_features = (
            self._local_features(
                belief
            )
        )

        stability = (
            self._action_stability(
                belief=belief,

                candidate_action=
                    candidate_action,

                goal=goal,
            )
        )

        # ----------------------------------------------
        # History
        # ----------------------------------------------

        current_unknown = (
            path_features[
                "path_unknown_ratio"
            ]
        )

        if (
            self._previous_path_unknown
            is None
        ):

            previous_unknown = (
                current_unknown
            )

            uncertainty_delta = 0.0

        else:

            previous_unknown = (
                self._previous_path_unknown
            )

            uncertainty_delta = (
                current_unknown
                - previous_unknown
            )

        if (
            self._ema_path_unknown
            is None
        ):

            ema_unknown = (
                current_unknown
            )

        else:

            ema_unknown = (
                self.ema_alpha
                * current_unknown
                +
                (1.0 - self.ema_alpha)
                * self._ema_path_unknown
            )

        self._previous_path_unknown = (
            current_unknown
        )

        self._ema_path_unknown = (
            ema_unknown
        )

        # ----------------------------------------------
        # Goal context
        # ----------------------------------------------

        distance_to_goal = (
            abs(
                current[0]
                - goal[0]
            )
            +
            abs(
                current[1]
                - goal[1]
            )
        )

        max_manhattan = max(
            (
                belief.width
                - 1
            )
            +
            (
                belief.height
                - 1
            ),
            1,
        )

        distance_to_goal_norm = (
            distance_to_goal
            / max_manhattan
        )

        path_length_norm = (
            path_features[
                "path_length"
            ]
            / max(
                belief.width
                * belief.height,
                1,
            )
        )

        # ----------------------------------------------
        # Candidate action
        # ----------------------------------------------

        action_up = float(
            candidate_action
            == Action.UP
        )

        action_down = float(
            candidate_action
            == Action.DOWN
        )

        action_left = float(
            candidate_action
            == Action.LEFT
        )

        action_right = float(
            candidate_action
            == Action.RIGHT
        )

        features = {
            # ==========================================
            # Uncertainty
            # ==========================================

            "action_instability":
                1.0 - stability,

            "path_unknown_ratio":
                current_unknown,

            "near_unknown_ratio":
                path_features[
                    "near_unknown_ratio"
                ],

            "global_unknown_fraction":
                belief.unknown_fraction(),

            "local_unknown_ratio":
                local_features[
                    "local_unknown_ratio"
                ],

            # ==========================================
            # History
            # ==========================================

            "previous_path_unknown_ratio":
                previous_unknown,

            "path_unknown_delta":
                uncertainty_delta,

            "ema_path_unknown_ratio":
                ema_unknown,

            # ==========================================
            # Decision context
            # ==========================================

            "distance_to_goal_norm":
                distance_to_goal_norm,

            "path_length_norm":
                path_length_norm,

            "local_wall_ratio":
                local_features[
                    "local_wall_ratio"
                ],

            "branching_factor_norm":
                self._branching_factor(
                    belief
                ),

            # ==========================================
            # Action identity
            # ==========================================

            "action_up":
                action_up,

            "action_down":
                action_down,

            "action_left":
                action_left,

            "action_right":
                action_right,
        }

        return Level3FeatureVector(
            values=features
        )


UNCERTAINTY_FEATURES = [
    "action_instability",
    "path_unknown_ratio",
    "near_unknown_ratio",
    "global_unknown_fraction",
    "local_unknown_ratio",
]


HISTORY_FEATURES = [
    "previous_path_unknown_ratio",
    "path_unknown_delta",
    "ema_path_unknown_ratio",
]


CONTEXT_FEATURES = [
    "distance_to_goal_norm",
    "path_length_norm",
    "local_wall_ratio",
    "branching_factor_norm",
    "action_up",
    "action_down",
    "action_left",
    "action_right",
]


FULL_FEATURES = (
    UNCERTAINTY_FEATURES
    + HISTORY_FEATURES
    + CONTEXT_FEATURES
)
