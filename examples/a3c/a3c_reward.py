from ffai.core.model import OutcomeType


rewards_own = {
    OutcomeType.TOUCHDOWN: 1,
    OutcomeType.SUCCESSFUL_CATCH: 0.1,
    OutcomeType.INTERCEPTION: 0.2,
    OutcomeType.SUCCESSFUL_PICKUP: 0.1,
    OutcomeType.FUMBLE: -0.1,
    OutcomeType.KNOCKED_DOWN: -0.1,
    OutcomeType.KNOCKED_OUT: -0.2,
    OutcomeType.CASUALTY: -0.5
}
rewards_opp = {
    OutcomeType.TOUCHDOWN: -1,
    OutcomeType.SUCCESSFUL_CATCH: -0.1,
    OutcomeType.INTERCEPTION: -0.2,
    OutcomeType.SUCCESSFUL_PICKUP: -0.1,
    OutcomeType.FUMBLE: 0.1,
    OutcomeType.KNOCKED_DOWN: 0.1,
    OutcomeType.KNOCKED_OUT: 0.2,
    OutcomeType.CASUALTY: 0.5
}
ball_progression_reward = 0.005


def reward_function(env, info, shaped=False):
    r = 0
    for outcome in env.get_outcomes():
        if not shaped and outcome.outcome_type != OutcomeType.TOUCHDOWN:
            continue
        team = None
        if outcome.player is not None:
            team = outcome.player.team
        elif outcome.team is not None:
            team = outcome.team
        if team == env.own_team and outcome.outcome_type in rewards_own:
            r += rewards_own[outcome.outcome_type]
        if team == env.opp_team and outcome.outcome_type in rewards_opp:
            r += rewards_opp[outcome.outcome_type]
    if info['ball_progression'] > 0:
        r += info['ball_progression'] * ball_progression_reward
    return r