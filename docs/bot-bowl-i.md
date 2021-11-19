# Bot Bowl I

Bot Bowl I was the first AI competition in Blood Bowl! The competition uses the [Fantasy Football AI (botbowl) framework](www.github.com/njustesen/botbowl) and the results were be presented at the IEEE Conference on Games in London, 21st August, 2019.

Bot Bowl I featured one track which used the traditional board size of 26×15 squares with 11 players on each side. Participants are, however, limited to use a prefixed human team. In future competitions, we plan to allow all teams and the option to customized rosters.

Bot Bowl I had four bot submissions and we added the Random agent baseline to the competition. The first phase of the competition consisted of a round-robin tournament where all bots played every other bot 10 times. 

## Submitted bots:

- **A2C (Hwanhee Kim, Yihong Yang, Yoonshin Jeong):** a deep reinforcement learning bot based on A2C that uses reward shaping.
- **NO ONE (Wang Lingxiao, Dong Zongkuan):** a deep reinforcement learning bot based on A2C that is first trained against the random agent and then itself. NO ONE uses reward shaping to incentivize moving towards the ball (and then to the end zone) as well as knocking down opponent players.
- **GrodBot (Peter Moore):** a scripted bot using a great deal of domain knowledge.
- **EvoGrod (Hwanhee Kim, Yihong Yang, Yoonshin Jeong):** an extension to GrodBot where in the 16 parameters used in the heuristic function (used to select the next player action) are evolved using an evolutionary algorithm.
Results from the round-robin tournament:

|         | Random | NO ONE | GrodBot | EvoGrod  | A2C    | Wins   | Ties | Losses |
| ------- |:------:|:------:|:-------:|:--------:|:------:|:------:|:----:|:------:|
| Random  | –      | 0/5/5  | 0/0/10  | 0/0/10   | 0/10/0 | 0      | 15   | 25     |
| NO ONE  | 5/5/0  | –      | 0/0/10  | 0/0/10   | 3/7/0  | 8      | 12   | 20     |
| GrodBot | 10/0/0 | 10/0/0 | –       | 4/5/1    | 10/0/0 | 34     | 5    | 1      |
| EvoGrod | 10/0/0 | 10/0/0 | 1/5/4   | –        | 10/0/0 | 31     | 5    | 4      |
| A2C     | 0/10/0 | 0/7/3  | 0/0/10  | 0/0/10   | –      | 0      | 17   | 23     | 

GrodBot and EvoGrod were the best two bots in the round-robin tournament and thus played a final best-of-five series to find the winner. Here, GrodBot won with three wins and one tie in four games. Videos of the finals with commentaries by bzl11 can be seen below:

## Final Rankings

**1st place ($500):** GrodBot (Peter Moore)

**2nd place ($300):** EvoGrod (Hwanhee Kim, Yihong Yang, Yoonshin Jeong)

**3rd place ($200):** NO ONE (Wang Lingxiao, Dong Zongkuan)

**4th place:** A2C (Hwanhee Kim, Yihong Yang, Yoonshin Jeong)

Congratulations to all the winners!

## Analysis by bzl11

<iframe src="https://www.youtube.com/embed/6qv_pzeYoOU" 
    width="820" 
    height="480"
    frameborder="0" 
    allowfullscreen>
</iframe>

The two bots play very similarly, which is not at all surprising. Things they need to improve on:
- Setups at the start of a drive. Both bots use the exact same setup for offense and defense. As a defensive setup it is not very good but that’s easy to fix. Offensive setup should ideally take into consideration what the defense has done, but of course that’s a lot harder to do.
- Blocking. The bots are way too happy to take one dice blocks, even in cases where they have made the effort to set up 2ds. This led to a lot of early rerolls being wasted and easily avoidable trouble. This is just tactics, which I guess goes down as medium difficulty to fix.
- Long term planning. This is the really difficult stuff so it’s hard to be too critical, but you’d see things like a bot taking great risks to move a catcher into scoring range, and then next turn just moving him backwards. They just don’t feel like they’re playing with a consistent plan.
- No use of Thrower. Neither bot ever tried to use their thrower to do anything, which is notable. Maybe they just don’t value Sure Hands as a skill, maybe they ignore it completely, I don’t know. They were mostly using catchers to carry the ball, which has its logic, but I’d like to see them try to get more value out of the few players with skills that they do have.

We hope to see these things improved in the bots that will be submitted to [Bot Bowl II](docs/bot-bowl-ii.md). which will either be in the summer of 2020.

