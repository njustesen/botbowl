def main2():
    import ffai.web.api as api

    print(api.list_bots())


def main():
    import ffai as ffai
    import time as time
    import examples.grodbot as grodbot

    config = ffai.load_config("bot-bowl-ii")
    config.competition_mode = False
    ruleset = ffai.load_rule_set(config.ruleset, all_rules=False)  # We don't need all the rules
    arena = ffai.load_arena(config.arena)
    home = ffai.load_team_by_filename("human", ruleset)
    away = ffai.load_team_by_filename("human", ruleset)

    # Play 10 games
    for i in range(10):
        home_agent = ffai.make_bot('grodbot')
        home_agent.name = "Grod1"
        away_agent = ffai.make_bot('grodbot')
        away_agent.name = "Grod2"
        config.debug_mode = False
        game = ffai.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        print("Starting game", (i+1))
        start = time.time()
        game.init()
        end = time.time()
        print(end - start)


if __name__ == "__main__":
    # execute only if run as a script
    main()
