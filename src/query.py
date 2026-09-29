class QueryReality:
    """
    Active information acquisition.

    In simulation:
        reveal a larger local region.

    Future physical version:
        camera reacquisition
        localization refresh
        force sensing
        etc.
    """

    def __init__(
        self,
        observer,
        cost=1.0,
    ):
        self.observer = observer
        self.cost = cost

    def execute(
        self,
        env,
        belief,
    ):
        observation = (
            self.observer.query(env)
        )

        belief.update(
            observation
        )

        return {
            "cost": self.cost,
            "observation":
                observation,
        }