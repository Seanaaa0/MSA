from src.env import GridWorld
from src.planner import AStarPlanner


class EpisodeRunner:
    """
    Runs a complete deterministic episode.

    Current version:
        - Uses the true environment state directly.
        - Uses A* for planning.
        - No belief error.
        - No perturbation.
        - No adequacy monitor.

    This is the perfect-information baseline.
    """

    def __init__(self, env, planner, max_steps=100):
        self.env = env
        self.planner = planner
        self.max_steps = max_steps

    def run(self, verbose=True):
        """
        Run one complete episode.

        Returns:
            dict with episode-level metrics
        """

        self.env.reset()

        collisions = 0

        if verbose:
            print("=== Baseline Episode ===")
            self.env.render_ascii()

        while not self.env.done:
            if self.env.steps >= self.max_steps:
                break

            current_state = self.env.agent_position
            goal = self.env.goal_position

            action = self.planner.next_action(
                start=current_state,
                goal=goal,
            )

            # Already at goal
            if action is None:
                break

            result = self.env.step(action)

            if result["collision"]:
                collisions += 1

            if verbose:
                print(
                    f"\nStep {result['steps']} | "
                    f"Action: {action.value} | "
                    f"Position: {result['position']} | "
                    f"Collision: {result['collision']}"
                )

                self.env.render_ascii()

        success = self.env.agent_position == self.env.goal_position

        episode_result = {
            "success": success,
            "steps": self.env.steps,
            "collisions": collisions,
            "final_position": self.env.agent_position,
            "goal_position": self.env.goal_position,
            "timeout": not success and self.env.steps >= self.max_steps,
        }

        if verbose:
            print("\n=== Episode Result ===")
            print(f"Success: {episode_result['success']}")
            print(f"Steps: {episode_result['steps']}")
            print(f"Collisions: {episode_result['collisions']}")
            print(f"Timeout: {episode_result['timeout']}")
            print(
                f"Final position: "
                f"{episode_result['final_position']}"
            )

        return episode_result


def main():
    env = GridWorld()

    planner = AStarPlanner(
        env.grid
    )

    runner = EpisodeRunner(
        env=env,
        planner=planner,
        max_steps=100,
    )

    runner.run(
        verbose=True
    )


if __name__ == "__main__":
    main()