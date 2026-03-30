"""
Save / load games and replays to ~/.botbowl/.
"""
from __future__ import annotations
import os
import pickle
import glob
from typing import Optional

_SAVE_DIR = os.path.join(os.path.expanduser('~'), '.botbowl', 'saves')


def _ensure_save_dir():
    os.makedirs(_SAVE_DIR, exist_ok=True)


def save_game(game, name: str):
    """Pickle the game object to ~/.botbowl/saves/{name}.pkl"""
    _ensure_save_dir()
    path = os.path.join(_SAVE_DIR, f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(game, f)


def load_game(name: str):
    """Load a saved game by name. Returns Game or raises FileNotFoundError."""
    path = os.path.join(_SAVE_DIR, f'{name}.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def list_saves() -> list[str]:
    """Return list of save names (without .pkl extension)."""
    _ensure_save_dir()
    files = glob.glob(os.path.join(_SAVE_DIR, '*.pkl'))
    return sorted(os.path.splitext(os.path.basename(f))[0] for f in files)


def delete_save(name: str):
    path = os.path.join(_SAVE_DIR, f'{name}.pkl')
    if os.path.exists(path):
        os.remove(path)


def save_exists(name: str) -> bool:
    return os.path.exists(os.path.join(_SAVE_DIR, f'{name}.pkl'))


# ---------------------------------------------------------------------------
# Replays — use the engine's built-in Replay class
# ---------------------------------------------------------------------------

from botbowl.core.model import Replay
from botbowl.core.util import get_data_path


def list_replays() -> list[str]:
    """Return list of replay IDs found in the data/replays directory."""
    replay_dir = get_data_path('replays')
    if not os.path.exists(replay_dir):
        return []
    files = glob.glob(os.path.join(replay_dir, '*.rep'))
    # Filenames: {home}_VS_{away}_{uuid}.rep
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    return sorted(names)


def load_replay(name: str) -> Optional[Replay]:
    """Load a replay by its full name (e.g. 'Human_VS_RandomBot_<uuid>')."""
    replay_dir = get_data_path('replays')
    path = os.path.join(replay_dir, f'{name}.rep')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        replay = pickle.load(f)
    # Rebuild step reports
    for idx, step in replay.steps.items():
        if step.num_reports == 0:
            step.game['state']['reports'] = []
        else:
            step.game['state']['reports'] = [
                r.to_json() for r in replay.reports[:step.num_reports]
            ]
    return replay


def delete_replay(name: str):
    replay_dir = get_data_path('replays')
    path = os.path.join(replay_dir, f'{name}.rep')
    if os.path.exists(path):
        os.remove(path)
