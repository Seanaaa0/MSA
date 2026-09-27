from src.env import GridWorld, Action


ACTION_KEYS = {
    "u": Action.UP,
    "d": Action.DOWN,
    "l": Action.LEFT,
    "r": Action.RIGHT,
}


def run_manual():
    """
    Simple terminal runner for testing the GridWorld manually.

    Controls:
        u = up
        d = down
        l = left
        r = right
        q = quit
        reset = reset
    """

    env = GridWorld()

    print("=== Model Shock Absorber - GridWorld V0 ===")
    print("Controls: W/A/S/D = move, R = reset, Q = quit")

    env.render_ascii()

    while True:
        command = input("\nAction > ").strip().lower()

        if command == "q":
            print("Exit.")
            break

        if command == "reset":
            env.reset()
            print("\nEnvironment reset.")
            env.render_ascii()
            continue

        if command not in ACTION_KEYS:
            print("Invalid command. Use u/d/l/r, reset, or q.")
            continue

        action = ACTION_KEYS[command]

        result = env.step(action)

        print(
            f"\nAction: {action.value} | "
            f"Position: {result['position']} | "
            f"Collision: {result['collision']} | "
            f"Steps: {result['steps']}"
        )

        env.render_ascii()

        if result["reached_goal"]:
            print("\n=== GOAL REACHED ===")
            print(f"Total steps: {result['steps']}")
            break


def main():
    run_manual()


if __name__ == "__main__":
    main()
