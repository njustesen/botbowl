
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

        save: function(game, name, team_id) {
            return $http.post(options.api.base_url + '/game/save', {'game_id': game, 'name': name, 'team_id': team_id});
        },

        load: function(name) {
            return $http.get(options.api.base_url + '/game/load/' + name);
        }

    };
});

appServices.factory('ReplayService', function($http) {
    return {
        get: function(id) {
            return $http.get(options.api.base_url + '/replays/' + id);
        },

        getSteps: function(id, from_idx, num_steps) {
            return $http.get(options.api.base_url + '/steps/' + id + "/" + from_idx + "/" + num_steps);
        },

        findAll: function() {
            return $http.get(options.api.base_url + '/replays/');
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

appServices.factory('BotService', function($http){
    return {
        listAll: function() {
            return $http.get(options.api.base_url + '/bots/');
        }
    }
});

appServices.factory('BigGuyService', function($http){
    return {
        bigGuys: [
            "Troll",
            "Minotaur",
            "Ogre",
            "Treeman",
            "Bull Centaur",
            "Chaos Troll",
            "Chaos Ogre",
            "Tomb Guardian", // Not sure
            "Kroxigor",
            "Yhetee",
            "Beast of Nurgle",
            "Rat Ogre",
            "Mummy", // Not sure
            "Warpstone Troll",

        ]
    }
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
            //'WEATHER_BLIZZARD': "<b>Blizzard:</b> It’s cold and snowing! The ice on the pitch means that any player attempting to move an extra square (GFI) will slip and be Knocked Down on a roll of 1-3, while the snow means that only quick or short passes can be attempted.",
            'WEATHER_SWELTERING_HEAT': "<b>Sweltering Heat:</b> Players may collapse after this drive.",
            'WEATHER_VERY_SUNNY': "<b>Very Sunny:</b> A -1 modifier on passing rolls.",
            'WEATHER_NICE': "<b>Nice weather:</b> Perfect for a game of fantasy football",
            'WEATHER_POURING_RAIN': "<b>Pouring Rain:</b> A -1 modifier on all catch, interception, and pick-up attempts.",
            'WEATHER_BLIZZARD': "<b>Blizzard:</b> A -1 modifier on GFI attempts and only quick and short passes are possible.",
            'ILLEGAL_SETUP_NUM': '<b>Illegal Setup:</b> You must pitch between 3 and 11 players.',
            'ILLEGAL_SETUP_SCRIMMAGE': '<b>Illegal Setup:</b> Min. 3 players on the line of scrimmage!',
            'ILLEGAL_SETUP_WINGS': '<b>Illegal Setup:</b> Max. 3 players on each wing!',
            'BALL_PLACED': '<team> <b>Kicks</b> the ball',
            'TOUCHBACK_BALL_PLACED': "<player> will start with the ball.",
            //'KICKOFF_GET_THE_REF': "<b>Get the Ref:</b> The fans exact gruesome revenge on the referee for some of the dubious decisions he has made, either during this match or in the past. His replacement is so intimidated that he can be more easily persuaded to look the other way. Each team receives 1 additional Bribe to use during this game. A Bribe allows you to attempt to ignore one call by the referee for a player who has committed a foul to be sent off, or a player armed with a secret weapon to be banned from the match. Roll a D6: on a roll of 3-6 the bribe is effective (preventing a turnover if the player was ejected for fouling), but on a roll of 1 the bribe is wasted and the call still stands! Each bribe may be used once per match.",
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
            'THROW_A_ROCK_ROLL': "<team> rolls.",
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
            'BALL_CAUGHT':  "<player> caught the ball.",
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
            'END_OF_FIRST_HALF': "<b>End of first half</b>.",
            'END_OF_SECOND_HALF': "<b>End of second half</b>.",
            'TOUCHDOWN': "<player> scored a <b>Touchdown</b>.",
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
            'FUMBLE': "<player> <b>fumbles</b>.",
            'FAILED_CATCH': "<player> failed to catch the ball.",
            "CATCH": "<player> <b>caught</b> the ball.",
            'BALL_SCATTER': "The ball scattered.",
            'BOMB_SCATTER': "The bomb scattered.",
            'PLAYER_SCATTER': "<player> scattered.",
            "BALL_BOUNCED": "The ball bounced.",
            "GENTLE_GUST_OUT_OF_BOUNDS": "A gentle gust makes the ball scatter out of bounds.",
            "GENTLE_GUST_OPP_pitch": "A gentle gust makes the ball scatter into the opponents half.",
            "GENTLE_GUST_IN_BOUNDS": "A gentle gust makes the ball scatter an additional square.",
            "TURN_START": "<team> <b>Turn <n>.</b>",
            "PLAYER_READY": "<player> is ready to play.",
            "PLAYER_NOT_READY": "<player> is still KO'd.",
            "FOLLOW_UP": "<player> follows up.",
            "BALL_OUT_OF_BOUNDS": "<b>Out of bounds!</b>",
            "SKILL_USED": "<player> uses the <skill> skill.<b>",
            "PLAYER_EJECTED": "<player> was <b>ejected!</b>",
            "BLOCK_ROLL": "<player> blocks <opp_player>.",
            "REROLL_USED": "<team> uses a team <b>re-roll</b>.",
            "FAILED_INTERCEPTION": "<player> failed to intercept the pass.",
            "THROW_IN_OUT_OF_BOUNDS": "The ball was thrown out of bounds again.",
            "THROW_IN": "The ball was thrown back in by the fans.",
            "BLITZ_START": "<team> makes a <b>blitz</b>.",
            "QUICK_SNAP": "<team> makes a <b>quick snap</b>.",
            "TEAM_SPECTATORS": "<team> has <b><n> fans</b> cheering for them.",
            "END_OF_GAME_DISQUALIFICATION": "<team> was <b>disqualified</b>.",
            "FAILED_BONE_HEAD": "<player> failed a <b>bonehead</b> roll.",
            "SUCCESSFUL_BONE_HEAD": "<player> passed a <b>bonehead</b> roll.",
            "FAILED_REALLY_STUPID": "<player> failed a <b>really stupid</b> roll.",
            "SUCCESSFUL_REALLY_STUPID": "<player> passed a <b>really stupid</b> roll.",
            "FAILED_WILD_ANIMAL": "<player> failed a <b>wild animal</b> roll.",
            "SUCCESSFUL_WILD_ANIMAL": "<player> passed a <b>wild animal</b> roll.",
            "FAILED_LONER": "<player> failed a <b>loner</b> roll.",
            "SUCCESSFUL_LONER": "<player> passed a <b>loner</b> roll.",
            "FAILED_PRO": "<player> failed a <b>pro</b> roll.",
            "SUCCESSFUL_PRO": "<player> passed a <b>pro</b> roll.",
            "FAILED_REGENERATION": "<player> failed a <b>Regeneration</b> roll.",
            "SUCCESSFUL_REGENERATION": "<player> passed a <b>Regeneration</b> roll.",
            "FAILED_LEAP": "<player> failed to <b>leap</b>.",
            "SUCCESSFUL_LEAP": "<player> successfully <b>leaped</b>.",
            "SUCCESSFUL_TAKE_ROOT": "<player> passed a <b>take root</b> roll.",
            "FAILED_TAKE_ROOT": "<player> failed a <b>take root</b> roll.",
            "STAND_UP": "<player> stood up.",
            "FAILED_STAND_UP": "<player> failed to stand up.",
            "FAILED_JUMP_UP": "<player> failed to jump up.",
            "FAILED_STAND_UP": "<player> failed to stand up.",
            "FAILED_JUMP_UP": "<player> failed to jump up.",
            "ACTION_SELECT_DIE": "<team> selects block die.",
            "SUCCESSFUL_BRIBE": "<team> used a bribe <b>successfully.",
            "FAILED_BRIBE": "<team> used a bribe <b>unsuccessfully.",
            "SUCCESSFUL_HYPNOTIC_GAZE": "<player> successfully <b>hypnotized</b> <opp_player>", 
            "FAILED_HYPNOTIC_GAZE": "<player> failed to <b>hypnotize</b> <opp_player>",
            "FAILED_BLOOD_LUST": "<player> has <b>Blood Lust</b>.",
            "SUCCESSFUL_BLOOD_LUST": "<player> does not have Blood Lust.",
            "EJECTED_BY_BLOOD_LUST": "<player> <b>leaves the game</b> to feed on a spectator.",
            "EATEN_DURING_BLOOD_LUST": "<player> was bit by <opp_player>.",
            "BOMB_HIT": "Bomb hit <player>.",
            "BOMB_EXPLODED": "Bomb exploded.",
            "SUCCESSFUL_LAND": "<player> landed successfully.",
            "FAILED_LAND": "<player> failed to land.",
            "PLAYER_BOUNCED": "<player> bounced.",
            "BOMB_OUT_OF_BOUNDS": "Bomb landed out of bounds.",
            "PLAYER_OUT_OF_BOUNDS": "<player> landed out of bounds.",
            "BALL_BOUNCE_PLAYER": "Ball bounced away from <player>.",
            "PLAYER_BOUNCE_PLAYER": "<player> bounced on <opp_player>.",
            "BOMB_ON_GROUND": "Bomb landed on the ground.",
            "BALL_BOUNCE_GROUND": "Ball bounced.",
            "PLAYER_BOUNCE_GROUND": "<player> bounced.",
            "FAILED_CATCH": "<player> failed to catch the bomb.",
            "SUCCESSFUL_CATCH": "<player> caught the ball.",
            "FAILED_CATCH": "<player> failed to catch the ball.",
            "SUCCESSFUL_CATCH_BOMB": "<player> caught the bomb.",
            "WILL_CATCH_BOMB": "<player> will attempt to catch the bomb.",
            "WONT_CATCH_BOMB": "<player> will <b>not</b> attempt to catch the bomb.",
            "SUCCESSFUL_ESCAPE_BEING_EATEN": "<player> escapes being eaten by <opp_player>.",
            "FAILED_ESCAPE_BEING_EATEN": "<player> failed escaping <opp_player>.",
            "SUCCESSFUL_ALWAYS_HUNGRY": "<player> is not hungry.",
            "FAILED_ALWAYS_HUNGRY": "<player> is hungry."
        },
        log_timouts: {
            'GAME_STARTED': 100,
            'SPECTATORS': 200,
            'FAME': 200,
            'HEADS_WON': 1000,
            'HEADS_LOSS': 1000,
            'TAILS_WON': 1000,
            'TAILS_LOSS': 1000,
            'HOME_RECEIVE': 1000,
            'AWAY_RECEIVE': 1000,
            'WEATHER_SWELTERING_HEAT': 1000,
            'WEATHER_VERY_SUNNY': 1000,
            'WEATHER_NICE': 1000,
            'WEATHER_POURING_RAIN': 1000,
            'WEATHER_BLIZZARD': 1000,
            'ILLEGAL_SETUP_NUM': 100,
            'ILLEGAL_SETUP_SCRIMMAGE': 100,
            'ILLEGAL_SETUP_WINGS': 100,
            'BALL_PLACED': 100,
            'TOUCHBACK_BALL_PLACED': 100,
            'KICKOFF_GET_THE_REF': 1000,
            'KICKOFF_RIOT': 1000,
            'KICKOFF_PERFECT_DEFENSE': 1000,
            'KICKOFF_HIGH_KICK': 1000,
            'KICKOFF_CHEERING_FANS':1000,
            'KICKOFF_CHANGING_WHEATHER': 1000,
            'KICKOFF_BRILLIANT_COACHING': 1000,
            'KICKOFF_QUICK_SNAP': 1000,
            'KICKOFF_BLITZ': 1000,
            'KICKOFF_THROW_A_ROCK': 1000,
            'KICKOFF_PITCH_INVASION': 1000,
            'THROW_A_ROCK_ROLL': 1000,
            'EXTRA_BRIBE': 1000,
            'TURN_SKIPPED': 1000,
            'TURN_ADDED': 1000,
            'RIOT': 1000,
            'HIGH_KICK': 1000,
            'EXTRA_REROLL': 1000,
            'PITCH_INVASION_ROLL': 1000,
            'KICK_OUT_OF_BOUNDS': 1000,
            'SETUP_DONE': 100,
            'KNOCKED_DOWN': 10,
            'ARMOR_BROKEN':  100,
            'ARMOR_NOT_BROKEN': 10,
            'HIT_BY_ROCK': 1000,
            'STUNNED': 100,
            'KNOCKED_OUT': 100,
            'BADLY_HURT': 100,
            'MISS_NEXT_GAME': 1000,
            'DEAD': 1000,
            'INTERCEPTION':  1000,
            'BALL_CAUGhT':  100,
            'BALL_DROPPED':  100,
            'FAILED_DODGE':  100,
            'SUCCESSFUL_DODGE': 10,
            'FAILED_GFI': 100,
            'SUCCESSFUL_GFI': 10,
            'FAILED_PICKUP': 100,
            'SUCCESSFUL_PICKUP': 10,
            'HANDOFF':10,
            'END_PLAYER_TURN': 100,
            'MOVE_ACTION_STARTED': 10,
            'BLOCK_ACTION_STARTED': 10,
            'BLITZ_ACTION_STARTED': 10,
            'PASS_ACTION_STARTED': 10,
            'FOUL_ACTION_STARTED': 10,
            'HANDOFF_ACTION_STARTED': 10,
            'END_OF_GAME_WINNER': null,
            'END_OF_GAME_DRAW': null,
            'END_OF_PREGAME': 100,
            'END_OF_TURN': 100,
            'END_OF_BLITZ': 100,
            'END_OF_QUICK_SNAP': 100,
            'END_OF_HALF': 100,
            'TOUCHDOWN': 1000,
            'TOUCHBACK': 100,
            'TURNOVER': 1000,
            'CASUALTY': 100,
            'APOTHECARY_USED_KO': 100,
            'APOTHECARY_USED_CASUALTY': 100,
            'CASUALTY_APOTHECARY': 100,
            'DAUNTLESS_USED': 10,
            'PUSHED_INTO_CROWD': 10,
            'PUSHED': 10,
            'ACCURATE_PASS': 10,
            'INACCURATE_PASS': 100,
            'FUMBLE': 100,
            'FAILED_CATCH': 100,
            "CATCH": 10,
            'BALL_SCATTER': 10,
            "BALL_BOUNCED": 10,
            "GENTLE_GUST_OUT_OF_BOUNDS": 100,
            "GENTLE_GUST_OPP_pitch": 100,
            "GENTLE_GUST_IN_BOUNDS": 10,
            "TURN_START": 10,
            "PLAYER_READY": 1000,
            "PLAYER_NOT_READY": 1000,
            "FOLLOW_UP": 10,
            "BALL_OUT_OF_BOUNDS": 100,
            "SKILL_USED": 10,
            "PLAYER_EJECTED": 1000,
            "BLOCK_ROLL": 10,
            "REROLL_USED": 100,
            "FAILED_INTERCEPTION": 100,
            "THROW_IN_OUT_OF_BOUNDS": 100,
            "THROW_IN": 10,
            "BLITZ_START": 1000,
            "QUICK_SNAP": 1000,
            "TEAM_SPECTATORS": 10,
            "END_OF_GAME_DISQUALIFICATION": null,
            "FAILED_BONE_HEAD": 1000,
            "SUCCESSFUL_BONE_HEAD": 100,
            "FAILED_REALLY_STUPID": 1000,
            "SUCCESSFUL_REALLY_STUPID": 100,
            "FAILED_WILD_ANIMAL": 1000,
            "SUCCESSFUL_WILD_ANIMAL": 100,
            "FAILED_LONER": 1000,
            "SUCCESSFUL_LONER": 100,
            "FAILED_PRO": 1000,
            "SUCCESSFUL_PRO": 100,
            "FAILED_REGENERATION": 1000,
            "SUCCESSFUL_REGENERATION": 1000,
            "FAILED_LEAP": 1000,
            "SUCCESSFUL_LEAP": 100,
            "SUCCESSFUL_TAKE_ROOT": 100,
            "FAILED_TAKE_ROOT": 1000,
            "STAND_UP": 10,
            "FAILED_STAND_UP": 1000,
            "FAILED_JUMP_UP": 1000, 
            "SUCCESSFUL_HYPNOTIC_GAZE": 100, 
            "FAILED_HYPNOTIC_GAZE": 1000,
            "FAILED_BLOOD_LUST": 100,
            "BOMB_HIT": 500,
            "BOMB_EXPLODED": 1000,
            "SUCCESSFUL_LAND": 500,
            "FAILED_LAND": 1000,
            "PLAYER_BOUNCED": 100,
            "BOMB_OUT_OF_BOUNDS": 500,
            "PLAYER_OUT_OF_BOUNDS": 500,
            "BALL_BOUNCE_PLAYER": 100,
            "PLAYER_BOUNCE_PLAYER": 100,
            "BOMB_ON_GROUND": 500,
            "BALL_BOUNCE_GROUND": 100,
            "PLAYER_BOUNCE_GROUND": 100,
            "FAILED_CATCH_BOMB": 1000,
            "SUCCESSFUL_CATCH_BOMB": 500,
            "WILL_CATCH_BOMB": 100,
            "WONT_CATCH_BOMB": 1000,
            "SUCCESSFUL_ESCAPE_BEING_EATEN": 500,
            "FAILED_ESCAPE_BEING_EATEN": 1000,
            "SUCCESSFUL_ALWAYS_HUNGRY": 500,
            "FAILED_ALWAYS_HUNGRY": 1000
        }
    };
});


appServices.factory('IconService', function() {

    return {
        playerIcons: {
            'Chaos': {
                'Beastman': 'cbeastman',
                'Chaos Warrior': 'cwarrior',
                'Minotaur': 'minotaur'
            },
            'Chaos Dwarf': {
                'Hobgoblin': 'cdhobgoblin',
                'Chaos Dwarf Blocker': 'cddwarf',
                'Bull Centaur': 'centaur',
                'Minotaur': 'minotaur'
            },
            'Dark Elf':{
                'Lineman': 'delineman',
                'Blitzer': 'deblitzer',
                'Witch Elf': 'dewitchelf',
                'Runner': 'dethrower',
                'Assassin': 'dehorkon'
            },
            'High Elf':{
                'Lineman': 'helineman',
                'Blitzer': 'heblitzer',
                'Thrower': 'hethrower',
                'Catcher': 'hecatcher'
            },
            'Wood Elf':{
                'Lineman': 'welineman',
                'Wardancer': 'weblitzer',
                'Thrower': 'wethrower',
                'Catcher': 'wecatcher',
                'Treeman': 'treeman'
            },
            'Human': {
                'Lineman': 'hlineman',
                'Blitzer': 'hblitzer',
                'Thrower': 'hthrower',
                'Catcher': 'hcatcher',
                'Ogre': 'ogre'
            },
            'Lizardman': {
                'Kroxigor': 'kroxigor',
                'Saurus': 'lmsaurus',
                'Skink': 'lmskink'
            },
            'Orc': {
                'Lineman': 'olineman',
                'Blitzer': 'oblitzer',
                'Thrower': 'othrower',
                'Black Orc Blocker': 'oblackorc',
                'Troll': 'troll',
                'Goblin': 'goblin'
            },
            'Elven Union': {
                'Lineman': 'eplineman',
                'Blitzer': 'epblitzer',
                'Thrower': 'epthrower',
                'Catcher': 'epcatcher'
            },
            'Skaven': {
                'Lineman': 'sklineman',
                'Blitzer': 'skstorm',
                'Thrower': 'skthrower',
                'Gutter Runner': 'skrunner',
                'Rat Ogre': 'ratogre'
            },
            'Amazon': {
                'Linewoman': 'amlineman',
                'Blitzer': 'amblitzer',
                'Thrower': 'amthrower',
                'Catcher': 'amcatcher'
            },
            'Undead': {
                'Zombie': 'uzombie',
                'Skeleton': 'uskeleton',
                'Ghoul': 'ughoul',
                'Wight': 'uwight',
                'Mummy': 'umummy'
            },
            'Vampire': {
                'Vampire': 'vampire',
                'Thrall': 'vthrall'
            }
        },

        getPlayerIcon: function (race, role, isHome, isActive){
            let icon_base = this.playerIcons[race][role];
            let icon_num = "1";
            let team_letter = isHome ? "b" : "";
            let angle = isActive ? "an" : "";
            return icon_base + icon_num + team_letter + angle + ".gif";
        }
    };
});
