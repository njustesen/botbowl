"""
Text utilities for the botbowl pygame GUI:
- Prettify snake_case strings
- Procedure status labels
- Game event log text generation
"""
from botbowl.core.table import OutcomeType
from botbowl.core.model import D6, D8, BBDie


def prettify(s: str) -> str:
    """Convert SNAKE_CASE or snake_case to Title Case."""
    if s is None:
        return ''
    return s.replace('_', ' ').title()


def get_procedure_label(game) -> str:
    """Return a human-readable label for the current game procedure/phase."""
    try:
        proc = game.get_procedure()
    except (IndexError, AttributeError):
        return ''
    if proc is None:
        return ''
    name = type(proc).__name__
    half_map = {1: '1st Half', 2: '2nd Half'}
    mapping = {
        'Pregame': 'Pre-Game',
        'WeatherTable': 'Pre-Game',
        'CoinTossFlip': 'Coin Toss',
        'CoinTossKickReceive': 'Coin Toss',
        'PostGame': 'Post-Game',
        'GameOver': 'Game Over',
        'QuickSnap': 'Quick Snap!',
        'Blitz_': 'Blitz!',
        'Setup': 'Setup',
        'KickOff': 'Kick Off',
        'LandKickOff': 'Kick Off',
        'EndGame': 'Game Over',
    }
    if name in mapping:
        return mapping[name]
    return half_map.get(game.state.half, '1st Half')


def get_turn_label(game, team) -> str:
    """Return turn display string for a team (e.g. '1/16')."""
    try:
        proc = game.get_procedure()
    except (IndexError, AttributeError):
        return ''
    if proc is None:
        return ''
    name = type(proc).__name__
    if name == 'QuickSnap' and team == game.state.current_team:
        return 'Quick Snap!'
    if name == 'Blitz_' and team == game.state.current_team:
        return 'Blitz!'
    config = game.config
    turn = team.state.turn
    rounds = config.rounds if hasattr(config, 'rounds') else 8
    return f'Turn {turn}/{rounds}'


def _player_label(player) -> str:
    if player is None:
        return '?'
    return f'{player.nr}. {player.name}'


def _team_label(team) -> str:
    if team is None:
        return '?'
    return team.name


def get_log_text(outcome) -> str:
    """Return a plain-text description of a game outcome/report."""
    t = outcome.outcome_type.name
    p = _player_label(outcome.player)
    op = _player_label(outcome.opp_player)
    tm = _team_label(outcome.team)
    n = str(outcome.n) if outcome.n else ''
    sk = prettify(outcome.skill.name) if outcome.skill else ''

    texts = {
        'GAME_STARTED': 'Game started.',
        'SPECTATORS': f'{n} spectators showed up.',
        'FAME': f'{tm} has +{n} FAME.',
        'HEADS_WON': f'Heads! {tm} won the coin toss.',
        'HEADS_LOSS': f'Heads! {tm} won the coin toss.',
        'TAILS_WON': f'Tails! {tm} won the coin toss.',
        'TAILS_LOSS': f'Tails! {tm} won the coin toss.',
        'HOME_RECEIVE': f'{tm} will receive the ball.',
        'AWAY_RECEIVE': f'{tm} will receive the ball.',
        'WEATHER_SWELTERING_HEAT': 'Sweltering Heat: Players may collapse.',
        'WEATHER_VERY_SUNNY': 'Very Sunny: -1 modifier on passing rolls.',
        'WEATHER_NICE': 'Nice weather: Perfect conditions.',
        'WEATHER_POURING_RAIN': 'Pouring Rain: -1 on catch/interception/pick-up.',
        'WEATHER_BLIZZARD': 'Blizzard: -1 on GFI, only quick/short passes.',
        'ILLEGAL_SETUP_NUM': 'Illegal Setup: Must field 3-11 players.',
        'ILLEGAL_SETUP_SCRIMMAGE': 'Illegal Setup: Min. 3 on scrimmage!',
        'ILLEGAL_SETUP_WINGS': 'Illegal Setup: Max. 3 per wing!',
        'BALL_PLACED': f'{tm} kicks the ball.',
        'TOUCHBACK_BALL_PLACED': f'{p} will start with the ball.',
        'KICKOFF_GET_THE_REF': 'Get the Ref!',
        'KICKOFF_RIOT': 'Riot!',
        'KICKOFF_PERFECT_DEFENSE': 'Perfect Defence! Kicking team may reorganize.',
        'KICKOFF_HIGH_KICK': 'High Kick! Receiving team may move a player under the ball.',
        'KICKOFF_CHEERING_FANS': 'Cheering Fans!',
        'KICKOFF_CHANGING_WHEATHER': 'Changing Weather!',
        'KICKOFF_BRILLIANT_COACHING': 'Brilliant Coaching!',
        'KICKOFF_QUICK_SNAP': 'Quick Snap! Receiving team can move one square.',
        'KICKOFF_BLITZ': 'Blitz! Kicking team gets a bonus action.',
        'KICKOFF_THROW_A_ROCK': 'Throw a Rock!',
        'KICKOFF_PITCH_INVASION': 'Pitch Invasion!',
        'THROW_A_ROCK_ROLL': f'{tm} rolls.',
        'EXTRA_BRIBE': f'{tm} gets a bribe.',
        'TURN_SKIPPED': 'Turn markers moved forward.',
        'TURN_ADDED': 'Turn markers moved backward.',
        'RIOT': f'{n} turn(s) added to this half.',
        'HIGH_KICK': 'High Kick!',
        'EXTRA_REROLL': f'{tm} receives an extra re-roll.',
        'PITCH_INVASION_ROLL': f'{p} is {n}.',
        'KICK_OUT_OF_BOUNDS': 'Ball will land out of bounds.',
        'SETUP_DONE': f'{tm} is done setting up.',
        'KNOCKED_DOWN': f'{p} goes to the ground.',
        'ARMOR_BROKEN': f"{p}'s armor was broken.",
        'ARMOR_NOT_BROKEN': f"{p}'s armor held.",
        'HIT_BY_ROCK': f'{p} was hit by a rock!',
        'STUNNED': f'{p} got stunned.',
        'KNOCKED_OUT': f"{p} got KO'd!",
        'BADLY_HURT': f'{p} got badly hurt!',
        'MISS_NEXT_GAME': f'{p} was injured: {n}',
        'DEAD': f'{p} was killed.',
        'INTERCEPTION': f'{p} intercepted the pass.',
        'BALL_CAUGHT': f'{p} caught the ball.',
        'BALL_DROPPED': f'{p} dropped the ball.',
        'FAILED_DODGE': f'{p} failed to dodge.',
        'SUCCESSFUL_DODGE': f'{p} dodged successfully.',
        'FAILED_GFI': f'{p} GFI failed.',
        'SUCCESSFUL_GFI': f'{p} GFI successful.',
        'FAILED_PICKUP': f'{p} failed to pick up the ball.',
        'SUCCESSFUL_PICKUP': f'{p} picked up the ball.',
        'HANDOFF': f'{p} handed the ball to {op}.',
        'END_PLAYER_TURN': f"{p}'s turn is over.",
        'MOVE_ACTION_STARTED': f'{p} starts a move action.',
        'BLOCK_ACTION_STARTED': f'{p} starts a block action.',
        'BLITZ_ACTION_STARTED': f'{p} starts a blitz action.',
        'PASS_ACTION_STARTED': f'{p} starts a pass action.',
        'FOUL_ACTION_STARTED': f'{p} starts a foul action.',
        'HANDOFF_ACTION_STARTED': f'{p} starts a handoff action.',
        'END_OF_GAME_WINNER': f'{tm} won {n}',
        'END_OF_GAME_DRAW': f'Draw {n}',
        'END_OF_TURN': f'{tm} ended their turn.',
        'END_OF_BLITZ': f'{tm} ended their blitz.',
        'END_OF_QUICK_SNAP': f'{tm} ended their quick snap.',
        'END_OF_FIRST_HALF': 'End of first half.',
        'END_OF_SECOND_HALF': 'End of second half.',
        'TOUCHDOWN': f'{p} scored a Touchdown!',
        'TOUCHBACK': f'Touchback! {tm} can give the ball to any player.',
        'TURNOVER': f'{tm} suffers a turnover.',
        'CASUALTY': f'{p} suffers a casualty!',
        'PUSHED_INTO_CROWD': f'{p} is pushed into the crowd.',
        'PUSHED': f'{p} was pushed.',
        'ACCURATE_PASS': f'{p} threw an accurate pass.',
        'INACCURATE_PASS': f'{p} threw an inaccurate pass.',
        'FUMBLE': f'{p} fumbles.',
        'FAILED_CATCH': f'{p} failed to catch the ball.',
        'CATCH': f'{p} caught the ball.',
        'BALL_SCATTER': 'The ball scattered.',
        'BOMB_SCATTER': 'The bomb scattered.',
        'PLAYER_SCATTER': f'{p} scattered.',
        'BALL_BOUNCED': 'The ball bounced.',
        'BALL_OUT_OF_BOUNDS': 'Out of bounds!',
        'TURN_START': f'{tm} Turn {n}.',
        'PLAYER_READY': f'{p} is ready to play.',
        'PLAYER_NOT_READY': f"{p} is still KO'd.",
        'FOLLOW_UP': f'{p} follows up.',
        'SKILL_USED': f'{p} uses the {sk} skill.',
        'PLAYER_EJECTED': f'{p} was ejected!',
        'BLOCK_ROLL': f'{p} blocks {op}.',
        'REROLL_USED': f'{tm} uses a team re-roll.',
        'FAILED_INTERCEPTION': f'{p} failed to intercept.',
        'THROW_IN': 'Ball thrown back by the fans.',
        'THROW_IN_OUT_OF_BOUNDS': 'Ball thrown out of bounds again.',
        'BLITZ_START': f'{tm} makes a blitz.',
        'QUICK_SNAP': f'{tm} makes a quick snap.',
        'TEAM_SPECTATORS': f'{tm} has {n} fans.',
        'END_OF_GAME_DISQUALIFICATION': f'{tm} was disqualified.',
        'FAILED_BONE_HEAD': f'{p} failed a bonehead roll.',
        'SUCCESSFUL_BONE_HEAD': f'{p} passed a bonehead roll.',
        'FAILED_REALLY_STUPID': f'{p} failed a really stupid roll.',
        'SUCCESSFUL_REALLY_STUPID': f'{p} passed a really stupid roll.',
        'FAILED_WILD_ANIMAL': f'{p} failed a wild animal roll.',
        'SUCCESSFUL_WILD_ANIMAL': f'{p} passed a wild animal roll.',
        'FAILED_LONER': f'{p} failed a loner roll.',
        'SUCCESSFUL_LONER': f'{p} passed a loner roll.',
        'FAILED_PRO': f'{p} failed a pro roll.',
        'SUCCESSFUL_PRO': f'{p} passed a pro roll.',
        'FAILED_REGENERATION': f'{p} failed a regeneration roll.',
        'SUCCESSFUL_REGENERATION': f'{p} passed a regeneration roll.',
        'FAILED_LEAP': f'{p} failed to leap.',
        'SUCCESSFUL_LEAP': f'{p} leaped successfully.',
        'SUCCESSFUL_TAKE_ROOT': f'{p} passed a take root roll.',
        'FAILED_TAKE_ROOT': f'{p} failed a take root roll.',
        'STAND_UP': f'{p} stood up.',
        'FAILED_STAND_UP': f'{p} failed to stand up.',
        'FAILED_JUMP_UP': f'{p} failed to jump up.',
        'ACTION_SELECT_DIE': f'{tm} selects block die.',
        'SUCCESSFUL_BRIBE': f'{tm} used a bribe successfully.',
        'FAILED_BRIBE': f'{tm} used a bribe unsuccessfully.',
        'SUCCESSFUL_HYPNOTIC_GAZE': f'{p} hypnotized {op}.',
        'FAILED_HYPNOTIC_GAZE': f'{p} failed to hypnotize {op}.',
        'FAILED_BLOOD_LUST': f'{p} has Blood Lust.',
        'SUCCESSFUL_BLOOD_LUST': f'{p} does not have Blood Lust.',
        'EJECTED_BY_BLOOD_LUST': f'{p} leaves the game to feed.',
        'EATEN_DURING_BLOOD_LUST': f'{p} was bitten by {op}.',
        'BOMB_HIT': f'Bomb hit {p}.',
        'BOMB_EXPLODED': 'Bomb exploded.',
        'SUCCESSFUL_LAND': f'{p} landed successfully.',
        'FAILED_LAND': f'{p} failed to land.',
        'PLAYER_BOUNCED': f'{p} bounced.',
        'BOMB_OUT_OF_BOUNDS': 'Bomb landed out of bounds.',
        'PLAYER_OUT_OF_BOUNDS': f'{p} landed out of bounds.',
        'SUCCESSFUL_CATCH': f'{p} caught the ball.',
        'SUCCESSFUL_CATCH_BOMB': f'{p} caught the bomb.',
        'WILL_CATCH_BOMB': f'{p} will attempt to catch the bomb.',
        'WONT_CATCH_BOMB': f'{p} will not attempt to catch the bomb.',
        'SUCCESSFUL_ESCAPE_BEING_EATEN': f'{p} escaped being eaten.',
        'FAILED_ESCAPE_BEING_EATEN': f'{p} failed to escape {op}.',
        'SUCCESSFUL_ALWAYS_HUNGRY': f'{p} is not hungry.',
        'FAILED_ALWAYS_HUNGRY': f'{p} is hungry.',
    }
    return texts.get(t, '')


def get_log_roll_outcomes(outcome) -> list:
    """Return list of (label_str, success_bool) per roll that has a target.
    E.g. [('3+', True), ('2+', False)]"""
    results = []
    for roll in outcome.rolls:
        target = roll.modified_target()
        if target is None:
            continue
        if roll.target_higher:
            label, ok = f'{target}+', roll.sum >= target
        elif roll.target_lower:
            label, ok = f'{target}-', roll.sum <= target
        else:
            label, ok = str(target), roll.sum == target
        results.append((label, ok))
    return results


def get_log_entry_color(outcome) -> tuple:
    """Return (R,G,B) text color for key events, or None for default."""
    t = outcome.outcome_type.name
    if t == 'TOUCHDOWN':
        return (255, 215, 80)
    if t in ('KNOCKED_OUT', 'BADLY_HURT', 'DEAD', 'CASUALTY'):
        return (220, 80, 80)
    if t in ('TURNOVER', 'FAILED_DODGE', 'FAILED_GFI', 'FUMBLE'):
        return (230, 150, 60)
    if t == 'ARMOR_BROKEN':
        return (200, 100, 60)
    return None


def get_log_dice_labels(outcome) -> list:
    """Return a list of (die_type, value) tuples for dice in a report.
    die_type is 'd6', 'd8', or 'block'. value is int for d6/d8, str name for block."""
    results = []
    for roll in outcome.rolls:
        for die in roll.dice:
            if isinstance(die, D6):
                results.append(('d6', die.get_value()))
            elif isinstance(die, D8):
                results.append(('d8', die.get_value()))
            elif isinstance(die, BBDie):
                results.append(('block', die.get_value().name))
    return results
