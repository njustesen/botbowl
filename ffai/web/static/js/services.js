
appServices.factory('GameService', function($http) {
    return {
        get: function(id) {
            return $http.get(options.api.base_url + '/games/' + id);
        },
        
        findAll: function() {
            return $http.get(options.api.base_url + '/games/');
        },

        act: function(id, action) {
            return $http.post(options.api.base_url + '/games/' + id + '/act', {'action': action});
        },

        delete: function(id) {
            return $http.delete(options.api.base_url + '/games/' + id + "/delete");
        },

        create: function(game) {
            return $http.put(options.api.base_url + '/game/create', {'game': game});
        },

        save: function(game, name) {
            return $http.post(options.api.base_url + '/game/save', {'game_id': game, 'name': name});
        },

        load: function(name) {
            return $http.get(options.api.base_url + '/game/load/' + name);
        }

    };
});


appServices.factory('TeamService', function($http) {
    return {
        findAll: function() {
            return $http.get(options.api.base_url + '/teams/');
        }
    };
});

appServices.factory('GameLogService', function() {

    return {
        log_texts: {
            'GAME_STARTED': "<b>Game started.</b>",
            'SPECTATORS': "<n> spectators showed up to watch the game.",
            'FAME': "<team> has <b>+<n> FAME</b> (Fan Advantage ModifiEr).",
            'HEADS_WON': "<b>Heads!</b> <away_team> won the coin toss.",
            'HEADS_LOSS': "<b>Heads!</b> <home_team> won the coin toss.",
            'TAILS_WON': "<b>Tails!</b> <away_team> won the coin toss.",
            'TAILS_LOSS': "<b>Tails!</b> <home_team> won the coin toss.",
            'HOME_RECEIVE': "<team> will receive the ball this half.",
            'AWAY_RECEIVE': "<team> will receive the ball this half.",
            //'WEATHER_SWELTERING_HEAT': "<b>Sweltering Heat:</b> It’s so hot and humid that some players collapse from heat exhaustion. Roll a D6 for each player on the pitch at the end of a drive. On a roll of 1 the player collapses and may not be set up for the next kick-off.",
            //'WEATHER_VERY_SUNNY': "<b>Very Sunny:</b> A glorious day, but the blinding sunshine causes a -1 modifier on all passing rolls.",
            //'WEATHER_NICE': "<b>Nice weather:</b> Perfect Blood Bowl weather.",
            //'WEATHER_POURING_RAIN': "<b>Pouring Rain:</b> It’s raining, making the ball slippery and difficult to hold. A -1 modifier applies to all catch, intercept, or pick-up rolls.",
            //'WEATHER_BLIZZARD': "<b>Blizzard:</b> It’s cold and snowing! The ice on the pitch means that any player attempting to move an extra square (GFI) will slip and be Knocked Down on a roll of 1-2, while the snow means that only quick or short passes can be attempted.",
            'WEATHER_SWELTERING_HEAT': "<b>Sweltering Heat:</b> Players may collapse after this drive.",
            'WEATHER_VERY_SUNNY': "<b>Very Sunny:</b> A -1 modifier on passing rolls.",
            'WEATHER_NICE': "<b>Nice weather:</b> Perfect for a game of fantasy football",
            'WEATHER_POURING_RAIN': "<b>Pouring Rain:</b> A -1 modifier on all catch, interception, and pick-up attempts.",
            'WEATHER_BLIZZARD': "<b>Blizzard:</b> A -1 modifier on GFI attempts and only quick and short passes are possible.",
            'ILLEGAL_SETUP_NUM': '<b>Illegal Setup:</b> You must pitch between 3 and 11 players.',
            'ILLEGAL_SETUP_SCRIMMAGE': '<b>Illegal Setup:</b> Min. 3 players on the line of scrimmage!',
            'ILLEGAL_SETUP_WINGS': '<b>Illegal Setup:</b> Max. 2 players on each wing!',
            'BALL_PLACED': '<team> <b>Kicks</b> the ball',
            'TOUCHBACK_BALL_PLACED': "<player> will start with the ball.",
            //'KICKOFF_GET_THE_REF': "<b>Get the Ref:</b> The fans exact gruesome revenge on the referee for some of the dubious decisions he has made, either during this match or in the past. His replacement is so intimidated that he can be more easily persuaded to look the other way. Each team receives 1 additional Bribe to use during this game. A Bribe allows you to attempt to ignore one call by the referee for a player who has committed a foul to be sent off, or a player armed with a secret weapon to be banned from the match. Roll a D6: on a roll of 2-6 the bribe is effective (preventing a turnover if the player was ejected for fouling), but on a roll of 1 the bribe is wasted and the call still stands! Each bribe may be used once per match.",
            //'KICKOFF_RIOT': "<b>Riot:</b> The trash talk between two opposing players explodes and rapidly degenerates, involving the rest of the players. If the receiving team’s turn marker is on turn 7 for the half, both teams move their turn marker back one space as the referee resets the clock back to before the fight started. If the receiving team has not yet taken a turn this half the referee lets the clock run on during the fight and both teams’ turn markers are moved forward one space. Otherwise roll a D6. On a 1-3, both teams’ turn markers are moved forward one space. On a 4-6, both team’s turn markers are moved back one space.",
            //'KICKOFF_PERFECT_DEFENSE': "<b>Perfect Defence:</b> The kicking team’s coach may reorganize his players – in other words he can set them up again into another legal defence. The receiving team must remain in the set-up chosen by their coach.",
            //'KICKOFF_HIGH_KICK': "<b>High Kick:</b> The ball is kicked very high, allowing a player on the receiving team time to move into the perfect position to catch it. Any one player on the receiving team who is not in an opposing player’s tackle zone may be moved into the square where the ball will land no matter what their MA may be, as long as the square is unoccupied.",
            //'KICKOFF_CHEERING_FANS': "<b>Cheering Fans:</b> Each coach rolls a D3 and adds their team’s FAME (see page 18) and the number of cheerleaders on their team to the score. The team with the highest score is inspired by their fans' cheering and gets an extra re-roll this half. If both teams have the same score, then both teams get a re-roll.",
            //'KICKOFF_CHANGING_WHEATHER': "<b>Changing Weather:</b> Make a new roll on the Weather table (see page 20). Apply the new Weather roll. If the new Weather roll was a ‘Nice’ result, then a gentle gust of wind makes the ball scatter one extra square in a random direction before landing.",
            //'KICKOFF_BRILLIANT_COACHING': "<b>Brilliant Coaching:</b> Each coach rolls a D3 and adds their FAME (see page 18) and the number of assistant coaches on their team to the score. The team with the highest total gets an extra team re-roll this half thanks to the brilliant instruction provided by the coaching staff. In case of a tie both teams get an extra team re-roll.",
            //'KICKOFF_QUICK_SNAP': "<b>Quick Snap!</b> The offence start their drive a fraction before the defence is ready, catching the kicking team flat-footed. All of the players on the receiving team are allowed to move one square. This is a free move and may be made into any adjacent empty square, ignoring tackle zones. It may be used to enter the opposing half of the pitch.",
            //'KICKOFF_BLITZ': "<b>Blitz!</b> The defence start their drive a fraction before the offence is ready, catching the receiving team flat-footed. The kicking team receives a free ‘bonus’ turn: however, players that are in an enemy tackle zone at the beginning of this free turn may not perform an Action. The kicking team may use team re-rolls during a Blitz. If any player suffers a turnover then the bonus turn ends immediately.",
            //'KICKOFF_THROW_A_ROCK': "<b>Throw a Rock:</b> An enraged fan hurls a large rock at one of the players on the opposing team. Each coach rolls a D6 and adds their FAME (see page 18) to the roll. The fans of the team that rolls higher are the ones that threw the rock. In the case of a tie a rock is thrown at each team! Decide randomly which player in the other team was hit (only players on the pitch are eligible) and roll for the effects of the injury straight away. No Armour roll is required.",
            //'KICKOFF_PITCH_INVASION': "<B>Pitch Invasion:</b> Both coaches roll a D6 for each opposing player on the pitch and add their FAME (see page 18) to the roll. If a roll is 6 or more after modification then the player is Stunned (players with the Ball & Chain skill are KO'd). A roll of 1 before adding FAME will always have no effect.",
            'KICKOFF_GET_THE_REF': "<b>Get the Ref!</b>",
            'KICKOFF_RIOT': "<b>Riot!</b>",
            'KICKOFF_PERFECT_DEFENSE': "<b>Perfect Defence!</b> The kicking team may reorganize their defense.",
            'KICKOFF_HIGH_KICK': "<b>High Kick!</b> The receiving team may move a player under the ball.",
            'KICKOFF_CHEERING_FANS': "<b>Cheering Fans!</b>",
            'KICKOFF_CHANGING_WHEATHER': "<b>Changing Weather!</b>",
            'KICKOFF_BRILLIANT_COACHING': "<b>Brilliant Coaching!</b>",
            'KICKOFF_QUICK_SNAP': "<b>Quick Snap!</b> The receiving team can move one square ignoring tackle zones.",
            'KICKOFF_BLITZ': "<b>Blitz!</b> The kicking team may take an action for every player not in a tackle zone.",
            'KICKOFF_THROW_A_ROCK': "<b>Throw a Rock!</b>",
            'KICKOFF_PITCH_INVASION': "<B>Pitch Invasion!</b>",
            'EXTRA_BRIBE': "<team> gets a bribe",
            'TURN_SKIPPED': "Turn markers are moved one step forward.",
            'TURN_ADDED': "Turn markers are moved one step backward.",
            'RIOT': "<n> turn(s) added to this half.",
            'HIGH_KICK': "<b>High Kick!</b>",
            'EXTRA_REROLL': "<team> receives and additional team re-roll.",
            'PITCH_INVASION_ROLL': "<player> is <b><n></b>.",
            'KICK_OUT_OF_BOUNDS': "The ball will land out of bounds.",
            'SETUP_DONE': "<team> is done setting up.",
            'KNOCKED_DOWN': "<player> goes to the ground.",
            'ARMOR_BROKEN':  "<players> armor was <b>broken</b>.",
            'ARMOR_NOT_BROKEN': "<players> armor was not broken.",
            'HIT_BY_ROCK': "<player> was hit by a rock!",
            'STUNNED': "<player> got <b>stunned</b>.",
            'KNOCKED_OUT': "<player> got <b>KO'ed!</B>",
            'BADLY_HURT': "<player> got <b>badly hurt!</B>",
            'MISS_NEXT_GAME': "<player> was injured: <n>",
            'DEAD': "<player> was <b>killed.</b>",
            'INTERCEPTION':  "<player> intercepted the pass.",
            'BALL_CAUGhT':  "<player> catched the ball.",
            'BALL_DROPPED':  "<player> dropped the ball.",
            'FAILED_DODGE':  " <player> failed to dodge.",
            'SUCCESSFUL_DODGE': " <player> successfully dodged.",
            'FAILED_GFI': "<player> GFI failed.",
            'SUCCESSFUL_GFI': "<player> GFI was successful.",
            'FAILED_PICKUP': "<player> failed to pickup the ball.",
            'SUCCESSFUL_PICKUP': "<player> picked up the ball.",
            'HANDOFF': "<player> handed of the ball off to <opp_player>.",
            'END_PLAYER_TURN': "<players> turn is over.",
            'MOVE_ACTION_STARTED': "<player> starts a movement action.",
            'BLOCK_ACTION_STARTED': "<player> starts a block action.",
            'BLITZ_ACTION_STARTED': "<player> starts a blitz action.",
            'PASS_ACTION_STARTED': "<player> starts a pass action.",
            'FOUL_ACTION_STARTED': "<player> starts a foul action.",
            'HANDOFF_ACTION_STARTED': "<player> starts a handoff action.",
            'END_OF_GAME_WINNER': "<team> won <score>",
            'END_OF_GAME_DRAW': "Game ended in a draw <score>",
            'END_OF_PREGAME': "END_OF_PREGAME",
            'END_OF_TURN': "<team> ended their turn.",
            'END_OF_BLITZ': "<team> ended their blitz.",
            'END_OF_QUICK_SNAP': "<team> ended their quick snap.",
            'END_OF_HALF': "END_OF_HALF",
            'TOUCHDOWN': " <player> scored a <b>Touchdown</b>.",
            'TOUCHBACK': "<b>Touchback! </b> <team> can give the ball to any player on the pitch.",
            'TURNOVER': "<team> suffers a <b>turnover</b>",
            'CASUALTY': "<player> suffers a <b>casualty!</b>",
            'APOTHECARY_USED_KO': "APOTHECARY_USED_KO",
            'APOTHECARY_USED_CASUALTY': "APOTHECARY_USED_CASUALTY",
            'CASUALTY_APOTHECARY': "CASUALTY_APOTHECARY",
            'DAUNTLESS_USED': "DAUNTLESS_USED",
            'PUSHED_INTO_CROWD': "<player> is pushed into the crowd.",
            'PUSHED': "<player> was pushed.",
            'ACCURATE_PASS': "<player> threw an <b>accurate</b> pass.",
            'INACCURATE_PASS': "<player> threw an <b>inaccurate</b> pass.",
            'FUMBLE': "<player> <b>fumbled</b> the ball</b>",
            'CATCH_FAILED': "<player> failed to catch the ball.",
            "CATCH": " <player> <b>catched</b> the ball.",
            'BALL_SCATTER': "The ball scattered.",
            "BALL_BOUNCED": "The ball bounced",
            "GENTLE_GUST_OUT_OF_BOUNDS": "A gentle gust makes the ball scatter out of bounds.",
            "GENTLE_GUST_OPP_pitch": "A gentle gust makes the ball scatter into the opponents half.",
            "GENTLE_GUST_IN_BOUNDS": "A gentle gust makes the ball scatter an additional square.",
            "TURN_START": "<team> <b>Turn <n>.</b>",
            "PLAYER_READY": "<player> is ready to play.",
            "PLAYER_NOT_READY": "<player> is still KO'd.",
            "FOLLOW_UP": "<player> follows up",
            "BALL_OUT_OF_BOUNDS": "<b>Out of bounds!</b>.",
            "SKILL_USED": "<player> uses the <skill> skill.<b>",
            "PLAYER_EJECTED": "<player> was <b>ejected!</b>",
            "BLOCK_ROLL": "<player> blocks <opp_player>",
            "REROLL_USED": "<team> uses a <b>reroll</b>.",
            "INTERCEPTION_FAILED": "<player> failed to intercept the pass.",
            "THROW_IN_OUT_OF_BOUNDS": "The ball was thrown out of bounds again.",
            "THROW_IN": "The ball was thrown back in by the fans.",
            "BLITZ_START": "<team> makes a <b>blitz</b>.",
            "QUICK_SNAP": "<team> makes a <b>quick snap</b>.",
            "TEAM_SPECTATORS": "<team> has <b><n> fans</b> cheering for them."


        }
    };
});

appServices.factory('IconService', function() {

    return {
        playerIcons: {
            'Human': {
                'Lineman': 'hlineman',
                'Blitzer': 'hblitzer',
                'Thrower': 'hthrower',
                'Catcher': 'hcatcher',
                'Ogre': 'ogre'
            },
            'Orc': {
                'Lineman': 'olineman',
                'Blitzer': 'oblitzer',
                'Thrower': 'othrower',
                'Black Orc Blocker': 'oblackorc',
                'Troll': 'troll',
                'Goblin': 'goblin'
            }
        }
    };
});
