import botbowl
import time
import scripted_bot_example


def main():
    # Load configurations, rules, arena and teams
    config = botbowl.load_config("bot-bowl")
    config.competition_mode = False
    config.pathfinding_enabled = True
    # config = get_config("gym-7.json")
    # config = get_config("gym-5.json")
    # config = get_config("gym-3.json")
    ruleset = botbowl.load_rule_set(
        config.ruleset, all_rules=False
    )  # We don't need all the rules
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)

    num_games = 1000
    home_tds = 0
    away_tds = 0
    home_wins = 0
    away_wins = 0
    draws = 0
    # Play 10 games
    for i in range(num_games):
        home_agent = botbowl.make_bot("scripted")
        home_agent.name = "Scripted Bot (home)"
        away_agent = botbowl.make_bot("scripted")
        away_agent.name = "Scripted (away)"
        config.debug_mode = False
        game = botbowl.Game(
            i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset
        )
        game.config.fast_mode = True

        print("Starting game", (i + 1))
        start = time.time()
        game.init()
        end = time.time()
        print(end - start)

        home_wins += 1 if game.get_winning_team() is game.state.home_team else 0
        away_wins += 1 if game.get_winning_team() is game.state.away_team else 0
        home_tds += game.state.home_team.state.score
        away_tds += game.state.away_team.state.score

        print(
            f"(home, away): Wins: ({home_wins}, {away_wins}), TDs: ({home_tds}, {away_tds})"
        )


if __name__ == "__main__":
    main()
