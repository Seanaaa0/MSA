DEFAULT_MAP_NAME = "simple"


SIMPLE_MAP = [
    "##########",
    "#S.......#",
    "#..###...#",
    "#........#",
    "#....###.#",
    "#.......G#",
    "##########",
]


# Multiple branch points.
# Designed to create locally plausible but suboptimal decisions.
BRANCH_MAP = [
    "###############",
    "#S...........G#",
    "#.#####.#####.#",
    "#.....#.#.....#",
    "###.#.#.#.###.#",
    "#...#.....#...#",
    "###############",
]


# Contains long and short route structures.
# Wrong decisions can remain legal but increase path cost.
DETOUR_MAP = [
    "###############",
    "#S......#....G#",
    "#.#####.#.###.#",
    "#.....#.#...#.#",
    "###.#.#.###.#.#",
    "#...#.#.....#.#",
    "#.###.#####.#.#",
    "#.............#",
    "###############",
]


# Contains branches that lead into dead-end-like structures.
DEAD_END_MAP = [
    "###############",
    "#S...........G#",
    "#.#####.#####.#",
    "#.....#.#.....#",
    "#.###.#.#.###.#",
    "#.#...#.#...#.#",
    "#.#.###.###.#.#",
    "#.............#",
    "###############",
]


# Narrow corridors and turns.
CORRIDOR_MAP = [
    "###############",
    "#S............#",
    "#####.#######.#",
    "#.....#.....#.#",
    "#.#####.###.#.#",
    "#.......#...#.#",
    "#.#######.###.#",
    "#............G#",
    "###############",
]


MAPS = {
    "simple": SIMPLE_MAP,
    "branch": BRANCH_MAP,
    "detour": DETOUR_MAP,
    "dead_end": DEAD_END_MAP,
    "corridor": CORRIDOR_MAP,
}


MAP_DESCRIPTIONS = {
    "simple":
        "Original baseline map.",

    "branch":
        "Multiple branch points with locally plausible "
        "but potentially suboptimal choices.",

    "detour":
        "Long and short route structure designed to "
        "produce non-collision regret.",

    "dead_end":
        "Branches and dead-end-like structures that "
        "punish locally plausible wrong turns.",

    "corridor":
        "Narrow corridors and turns that amplify "
        "decision-context dependence.",
}


def get_map(name: str):
    """
    Return a copy of a named map.
    """

    try:
        return list(MAPS[name])

    except KeyError as exc:

        available = ", ".join(MAPS)

        raise ValueError(
            f"Unknown map {name!r}. "
            f"Available maps: {available}"
        ) from exc


def available_maps():
    """
    Return all registered map names.
    """

    return tuple(MAPS.keys())