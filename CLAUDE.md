# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Install for development
```bash
git clone https://github.com/njustesen/botbowl
cd botbowl
python setup.py build   # compiles Cython pathfinding (requires C++ compiler)
pip install -e .
```

### Run tests
```bash
pytest                                      # all tests
pytest tests/game/test_block.py            # single test file
pytest tests/game/test_block.py::test_name # single test
```

### Lint
```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

### Run the web server
```bash
python -c "import botbowl.web.server as server; server.start_server(debug=True, use_reloader=False, port=1234)"
```

## Architecture

### Core engine (`botbowl/core/`)
- **`model.py`** — All game model classes: `Game`, `GameState`, `Team`, `Player`, `Square`, `Ball`, `Action`, `Agent`, `Configuration`, `RuleSet`, etc. Most model objects extend `Reversible` to support the forward model.
- **`procedure.py`** — All game procedures as a class hierarchy extending `Procedure` (itself extending `Reversible`). Procedures implement the game rules. They are pushed onto a stack; the top-most procedure runs until done. Examples: `Turn`, `Block`, `Move`, `Pass`, `Setup`, `CoinTossFlip`.
- **`game.py`** — The `Game` class, the primary API for interacting with a running game. Key methods: `game.init()`, `game.step(action)`, `game.get_available_actions()`, `game.get_procedure()`.
- **`forward_model.py`** — A change-tracking system enabling efficient undo/redo. `Reversible` objects log mutations as `Step` entries into a `Trajectory`. Call `game.enable_forward_model()`, then use `game.revert(step_id)` and `game.forward(steps)` to navigate game states without `deepcopy`.
- **`load.py`** — Functions to load configs, rulesets (XML), arenas (txt), teams (JSON), and formations from `botbowl/data/`.
- **`table.py`** — Enumerations and lookup tables (dice results, skills, action types, tile types, weather, etc.).
- **`pathfinding/`** — Pathfinding implemented in Python (`python_pathfinding.py`) and optionally compiled Cython (`cython_pathfinding.pyx`) for performance.

### AI layer (`botbowl/ai/`)
- **`env.py`** — `BotBowlEnv`: an OpenAI Gym-compatible environment wrapping the game. `EnvConf` configures size (1/3/5/7/11 players), feature layers, and formations. Observations are `(spatial_obs, non_spatial_obs, action_mask)`.
- **`layers.py`** — `FeatureLayer` classes that convert game state into spatial numpy arrays for RL observations.
- **`proc_bot.py`** — `ProcBot`: a base class for scripted bots that dispatches on the current `Procedure` type. Subclass this and override methods like `turn()`, `block()`, `move()`, etc.
- **`bots/`** — Reference bot implementations: `RandomBot`, `IdleBot`, `CrashBot`, and others used for testing.
- **`competition/`** — Infrastructure to run bot competitions (socket communication, result structures).
- **`registry.py`** — Global registry for bots by name.

### Web layer (`botbowl/web/`)
Flask-based web server (`server.py`) with a REST API (`api.py`) and game hosting (`host.py`). Serves a browser UI for human vs. human or human vs. bot play.

### Data (`botbowl/data/`)
- `config/` — JSON game configurations (e.g. `gym-11.json`, `bot-bowl.json`)
- `rules/` — XML rulesets (`BB2016.xml`, `LRB5-Experimental.xml`)
- `teams/` — JSON team definitions, organized by pitch size
- `arenas/` — Text-based pitch layouts
- `formations/` — Text-based formation definitions

## Key Patterns

### Procedure-based game loop
The game runs by repeatedly calling `game.step(action)`. Internally, the top procedure on the stack calls its `step(action)` method. If an action is required from an agent, the procedure returns without completing; otherwise it pushes sub-procedures and loops. To inspect what decision is pending, call `game.get_procedure()` and check its type.

### Dice fixing in tests
Tests use class-level `FixedRolls` lists on dice classes to control randomness. The `only_fixed_rolls` context manager in `tests/util.py` is the standard way to write deterministic tests:
```python
with only_fixed_rolls(game, d6=[6, 6], block_dice=[BBDieResult.DEFENDER_DOWN]):
    game.step(action)
```

### Test utilities
`tests/util.py` provides `get_game_turn()`, `get_custom_game_turn()`, and related helpers that set up a game at a specific point (e.g. at the start of a turn, with specific player positions and ball placement) for use in unit tests.

### Game sizes
Games are parameterized by team size: 1, 3, 5, 7, or 11 players per side. Each size has its own config, arena, and formation files. The gym environments are named `botbowl-1`, `botbowl-3`, etc.
