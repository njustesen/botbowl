import botbowl
from botbowl.core import Action, Agent, ActionType


def print_available_action_types(game):
    for action_choice in game.get_available_actions():
        print(action_choice.action_type.name, end=', ')
    print("\n", "-"*5, sep="")


def main():

    # Setup a game
    config = botbowl.load_config("bot-bowl-iii")
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)
    agent_home = Agent("home agent", human=True)
    agent_away = Agent("home agent", human=True)
    game = botbowl.Game(1, home, away, agent_home, agent_away, config, arena=arena, ruleset=ruleset)
    game.init()

    # Enable forward model
    game.enable_forward_model()
    step_id = game.get_step()

    # Force determinism?
    # game.set_seed(1)  # Force determinism

    # Take some actions
    game.step(Action(ActionType.START_GAME))
    game.step(Action(ActionType.HEADS))
    game.step(Action(ActionType.RECEIVE))

    # Kicking team is random if you didn't set the seed
    print("Home is kicking: ", game.get_kicking_team() == game.state.home_team)
    print_available_action_types(game)

    # Revert state and save the steps
    steps = game.revert(step_id)

    # Print available actions: Should only contain START_GAME
    print_available_action_types(game)

    # step the game forward again
    game.forward(steps)
    print("Home is kicking: ", game.get_kicking_team() == game.state.home_team)
    print_available_action_types(game)


if __name__ == "__main__":
    main()