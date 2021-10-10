'use strict';

var app = angular.module('app', ['ngRoute', 'ngSanitize', 'appControllers', 'appServices', 'appDirectives', 'appFilters']);

var appServices = angular.module('appServices', []);
var appControllers = angular.module('appControllers', []);
var appDirectives = angular.module('appDirectives', []);
var appFilters = angular.module('appFilters', []);

var options = {};
options.api = {};
//options.api.base_url = "http://127.0.0.1:5000";
options.api.base_url = window.location.protocol + "//" + window.location.host;

app.config(['$locationProvider', '$routeProvider', 
  function($location, $routeProvider) {
    $routeProvider.
        when('/', {
            templateUrl: 'static/partials/game.list.html',
            controller: 'GameListCtrl'
        }).
        when('/game/create', {
            templateUrl: 'static/partials/game.create.html',
            controller: 'GameCreateCtrl',
            access: { requiredAuthentication: true }
        }).
        when('/game/play/:id/:team_id', {
            templateUrl: 'static/partials/game.play.html',
            controller: 'GamePlayCtrl',
            access: { requiredAuthentication: true }
        }).
        when('/game/hotseat/:id', {
            templateUrl: 'static/partials/game.play.html',
            controller: 'GamePlayCtrl',
            access: { requiredAuthentication: true }
        }).
        when('/game/spectate/:id', {
            templateUrl: 'static/partials/game.play.html',
            controller: 'GamePlayCtrl',
            access: { requiredAuthentication: true }
        }).
        when('/game/replay/:id/', {
            templateUrl: 'static/partials/game.play.html',
            controller: 'GamePlayCtrl',
            access: { requiredAuthentication: true }
        }).
        otherwise({
            redirectTo: '/'
        });
}]);



appControllers.controller('GameListCtrl', ['$scope', '$window', 'GameService', 'ReplayService',
    function GameListCtrl($scope, $window, GameService, ReplayService) {
        $scope.games = [];
        $scope.replays = [];
        $scope.savedGames = [];

        GameService.findAll().success(function(data) {
            $scope.games = data.games;
            $scope.savedGames = data.saved_games;
        });

        ReplayService.findAll().success(function(data) {
            $scope.replays = data.replays;
        });

        $scope.loadGame = function loadGame(name){
            GameService.load(name).success(function(data) {
                 $window.location.href = '/#/game/play/' + data.game.game_id + '/' + data.team_id + '/'
            }).error(function(status, data) {
                console.log(status);
                console.log(data);
            });
        };

        $scope.loadReplay = function loadReplay(replay_id){
            ReplayService.get(replay_id).success(function(data) {
                 $window.location.href = '/#/game/replay/' + data.replay_id
            }).error(function(status, data) {
                console.log(status);
                console.log(data);
            });
        };

        $scope.deleteGame = function deletegame(id) {
            if (id != undefined) {

                GameService.delete(id).success(function(data) {
                    var games = $scope.games;
                    for (var gameKey in games) {
                        if (games[gameKey]._id == id) {
                            $scope.games.splice(gameKey, 1);
                            break;
                        }
                    }
                }).error(function(status, data) {
                    console.log(status);
                    console.log(data);
                });
            }
        };
    }
]);

appControllers.controller('GameCreateCtrl', ['$scope', '$location', 'GameService', 'TeamService', 'IconService', 'BotService',
    function GameCreateCtrl($scope, $location, GameService, TeamService, IconService, BotService) {

        $scope.teams = [];
        $scope.bots = [];
        $scope.home_team_id = null;
        $scope.away_team_id = null;

        TeamService.findAll().success(function(data) {
            $scope.teams = data;
        });

        BotService.listAll().success(function(data) {
            $scope.bots = data;
        });
        
        $scope.getTeam = function getTeam(team_name){
            for (let i in $scope.teams){
                if ($scope.teams[i].name === team_name){
                    return $scope.teams[i];
                }
            }
            return null;
        };

        $scope.playerIcon = function playerIcon(player, isHome, race){
            return IconService.getPlayerIcon(race, player.role, isHome, false);
        };

        $scope.prettify = function prettify(text){
            let pretty = text.replace("SETUP_FORMATION_", "").toLowerCase().split("_").join(" ");
            return pretty.charAt(0).toUpperCase() + pretty.slice(1);
        };

        $scope.home_player = "human";
        $scope.away_player = "human";

        $scope.createGame = function createGame(game) {
            //var content = $('#textareaContent').val();
            game = {};
            game.home_team_name = $scope.home_team_name;
            game.away_team_name = $scope.away_team_name;
            game.home_player = $scope.home_player;
            game.away_player = $scope.away_player;

            GameService.create(game).success(function(data) {
                $location.path("/");
            }).error(function(status, data) {
                console.log(status);
                console.log(data);
            });
        };
    }
]);

appControllers.controller('GamePlayCtrl', ['$scope', '$routeParams', '$location', '$sce', 'GameService', 'IconService', 'GameLogService', 'ReplayService', 'BigGuyService',
    function GamePlayCtrl($scope, $routeParams, $location, $sce, GameService, IconService, GameLogService, ReplayService, BigGuyService) {
        $scope.RELOAD_TIME_SLOW = 1000;
        $scope.RELOAD_TIME_FAST = 200;
        $scope.game = {};
        $scope.reportsLimit = 20;
        $scope.saved = false;
        $scope.loading = true;
        $scope.refreshing = false;
        $scope.hover_player = null;
        $scope.selected_square = null;
        $scope.selected_player = null;
        $scope.main_action = null;
        $scope.available_positions = [];
        $scope.modalVisible = false;
        $scope.modelError = false;
        $scope.special_actions = [];
        $scope.special_action_selected = null;
        $scope.gridClass = 'none';
        $scope.opp_turn = false;
        $scope.clock = "";
        $scope.local_state = {
            balls: [],
            board: [],
            home_dugout: [],
            away_dugout: [],
            player_positions: {}
        };

        $scope.game_id = $routeParams.id;
        $scope.team_id = $routeParams.team_id;
        $scope.spectating = window.location.href.indexOf('/spectate/') >= 0;
        $scope.replaying = window.location.href.indexOf('/replay/') >= 0;
        $scope.replaySpeed = 200;
        $scope.replayDoneLoading = false;
        $scope.loadingSteps = false

        if ($scope.replaying){
            $scope.replay_id = $scope.game_id;
            $scope.spectating = true;
            $scope.replayIsPlaying = false;
            $scope.replayStep = 0;
        }

        $scope.getAvailable = function getAvailable(square){
            if ($scope.special_action_selected !== null && square.special_actions.length > 0) {
                return square.special_actions.indexOf($scope.special_action_selected.action_type) > -1;
            } else {
                return square.available;
            }
        };

        $scope.getD6Rolls = function getD6Rolls(square, scaled){
            if ($scope.special_action_selected !== null && square.special_rolls[$scope.special_action_selected.action_type] !== undefined) {
                let rolls = square.special_rolls[$scope.special_action_selected.action_type];
                if (scaled){
                    if ($scope.special_action_selected.action_type === "STAB"){
                        let scaled_rolls = [];
                        for (let idx in rolls) {
                            let roll = rolls[idx];
                            scaled_rolls.push(Math.ceil(roll / 2));
                        }
                        return scaled_rolls;
                    }
                }
                return rolls;
            } else {
                return square.rolls;
            }
        };

        $scope.getBlockDice = function getBlockRoll(square){
            return square.block_dice;
            if ($scope.special_action_selected !== null && square.special_actions.indexOf($scope.special_action_selected.action_type) > -1) {
                return 0;
            } else {
                return square.block_dice;
            }
        };

        $scope.getActionType = function getActionType(square){
            if ($scope.special_action_selected !== null && square.special_actions.indexOf($scope.special_action_selected.action_type) >= 0) {
                return $scope.special_action_selected.action_type;
            } else {
                return square.action_type;
            }
        };

        document.addEventListener('keydown', function(event) {
            if (event.ctrlKey){

                $scope.$apply();
            }
            if (event.shiftKey){

                $scope.$apply();
            }
        });

        $scope.saveGame = function saveGame(name){
            $scope.modelError = false;
            // Get state
            GameService.save($scope.game.game_id, name, $scope.team_id).success(function(data) {
                $scope.saved = true;
                $scope.modalVisible = false;
            }).error(function(status, data) {
                $scope.modelError = true;
                //$location.path("/#/");
            });
        };

        $scope.showReport = function showReport(report){
            return report.outcome_type in GameLogService.log_texts;
        };

        $scope.getPlayer = function getPlayer(player_id){
            if (player_id in $scope.game.state.home_team.players_by_id){
                return $scope.game.state.home_team.players_by_id[player_id];
            }
            if (player_id in $scope.game.state.away_team.players_by_id){
                return $scope.game.state.away_team.players_by_id[player_id];
            }
            return null;
        };

        $scope.title = function(str){
            return str.replace(
                /\w\S*/g,
                function(txt) {
                    return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase().replace("_", " ");
                }
            );
        };

        $scope.reportBlock = function reportBlock(report) {
            if ($scope.showReport(report)){
                let line = GameLogService.log_texts[report.outcome_type] + "\n";
                line = line.replace("<home_team>", "<span class='label label-primary'>" + $scope.game.state.home_team.name + "</span> ");
                line = line.replace("<away_team>", "<span class='label label-danger'>" + $scope.game.state.away_team.name + "</span> ");
                line = line.replace("<team>", "<span class='label label-" + (report.team_id === $scope.game.state.home_team.team_id  ? ("primary'>" + $scope.game.state.home_team.name) : ("danger'>" + $scope.game.state.away_team.name)) + "</span> " );
                line = line.replace("<score-sorted>", Math.max($scope.game.state.home_team.state.score, $scope.game.state.away_team.state.score) + " - " + Math.min($scope.game.state.home_team.state.score, $scope.game.state.away_team.state.score) );

                if (report.skill !== null){
                    line = line.replace("<skill>", '<span class="label label-success skill">' + $scope.title(report.skill.replace("_", " ")) + '</span>');
                }
                let n = report.n;
                if (typeof(n) === "string"){
                    n = report.n.replace("NONE", "badly hurt").toLowerCase();
                }
                line = line.replace("<n>", n);
                if (report.player_id != null){
                    let player = $scope.getPlayer(report.player_id);
                    let team = $scope.teamOfPlayer(player);
                    line = line.replace("<player>", "<span class='label label-" + (team.team_id === $scope.game.state.home_team.team_id ? ("primary'>" + player.nr + ". " + player.name) : ("danger'>" + player.nr + ". " + player.name)) + "</span> " );
                    line = line.replace("<players>", "<span class='label label-" + (team.team_id === $scope.game.state.home_team.team_id ? ("primary'>" + player.nr + ". " + player.name) : ("danger'>" + player.nr + ". " + player.name + "'s")) + "</span> " );
                }
                if (report.opp_player_id != null){
                    let player = $scope.getPlayer(report.opp_player_id);
                    let team = $scope.teamOfPlayer(player);
                    line = line.replace("<opp_player>", "<span class='label label-" + (team.team_id === $scope.game.state.home_team.team_id ? ("primary'>" + player.nr + ". " + player.name) : ("danger'>" + player.nr + ". " + player.name)) + "</span> " );
                }
                return line;
            }
            return null;
        };

        $scope.teamOfPlayer = function teamOfPlayer(player){
            if (player.team_id === $scope.game.state.home_team.team_id){
                return $scope.game.state.home_team;
            }
            if (player.team_id === $scope.game.state.away_team.team_id) {
                return $scope.game.state.away_team;
            }
            return null;
        };

        $scope.isPlayerActive = function isPlayerActive(player){
            return player.player_id === $scope.game.state.active_player_id || player.player_id === $scope.game.active_other_player_id;
        };

        $scope.playerIcon = function playerIcon(player){
            let team = $scope.teamOfPlayer(player);
            let isHome = player.team_id === $scope.game.state.home_team.team_id;
            return IconService.getPlayerIcon(team.race, player.role, isHome, $scope.isPlayerActive(player));
        };

        $scope.getCursor = function getCursor(square){
            if ($scope.special_action_selected !== null && square.special_actions.indexOf($scope.special_action_selected.action_type) > -1 && $scope.special_action_selected.action_type === "STAB"){
                return "cursor: url(static/img/icons/actions/stab.gif), auto";
            } else if (square.available && (square.action_type === "HANDOFF" || square.handoff_roll !== null)){
                return "cursor: url(static/img/icons/actions/handover.gif), auto";
            } else if (square.available && square.block_dice !== 0 && square.block_dice !== null){
                return "cursor: url(static/img/icons/actions/block.gif), auto";
            } else if (square.available && (square.action_type === "FOUL" || square.foul_roll !== null)){
                return "cursor: url(static/img/icons/actions/foul.gif), auto";
            } else if (square.available && square.action_type === "PASS"){
                return "cursor: url(static/img/icons/actions/pass.gif), auto";
            } else if (square.available && square.action_type === "PICKUP_TEAM_MATE"){
                return "cursor: url(static/img/icons/actions/pickup-team-mate.gif), auto";
            } else if (square.available && square.action_type === "THROW_TEAM_MATE"){
                return "cursor: url(static/img/icons/actions/throw-team-mate.gif), auto";
            } else if (square.available && square.action_type === "HYPNOTIC_GAZE"){
                return "cursor: url(static/img/icons/actions/gaze.gif), auto";
            } else if (square.available && square.action_type === "THROW_BOMB"){
                return "cursor: url(static/img/icons/actions/bomb.gif), auto";
            }
            return "";
        };

        $scope.clickSpecialAction = function clickSpecialAction(action){
            if ($scope.special_action_selected !== action){
                $scope.special_action_selected = action;
            } else {
                $scope.special_action_selected = null;
            }
        };

        $scope.clickSkillAction = function clickSkillAction(event, skillAction){
            event.stopPropagation();
            for (let idx in $scope.game.state.available_actions) {
                let a = $scope.game.state.available_actions[idx];
                if (a.disabled){
                    continue;
                }
                if (a.action_type === "USE_SKILL" && skillAction === 'use'){
                    $scope.pickActionType(a);
                } else if (a.action_type === "DONT_USE_SKILL" && skillAction === 'dont-use'){
                    $scope.pickActionType(a);
                }
            }
        };

        $scope.clickAction = function clickAction(event, action){
            event.stopPropagation();
            for (let idx in $scope.game.state.available_actions) {
                let a = $scope.game.state.available_actions[idx];
                if (a.disabled){
                    continue;
                }
                if (a.action_type === "START_MOVE" && action === "move"){
                    $scope.pickActionType(a);
                } else if (a.action_type === "START_BLOCK" && action === "block"){
                    $scope.pickActionType(a);
                } else if (a.action_type === "START_PASS" && action === "pass"){
                    $scope.pickActionType(a);
                } else if (a.action_type === "START_HANDOFF" && action === "handoff"){
                    $scope.pickActionType(a);
                } else if (a.action_type === "START_BLITZ" && action === "blitz"){
                    $scope.pickActionType(a);
                } else if (a.action_type === "START_FOUL" && action === "foul"){
                    $scope.pickActionType(a);
                } else if (a.action_type === "START_THROW_BOMB" && action === "throw_bomb"){
                    $scope.pickActionType(a);
                }
            }
        };

        $scope.playerStartAction = function playerStartAction(x, y, typeName){
            if ($scope.selected_square == null || $scope.selectedPlayer().player_id == null || $scope.selected_square.x !== x || $scope.selected_square.y !== y){
                return false;
            }
            for (let idx in $scope.game.state.available_actions){
                let action = $scope.game.state.available_actions[idx];
                if ($scope.agent_id != null && $scope.agent_id !== action.agent_id){
                    action.disabled = true;
                }
                if (action.player_ids.indexOf($scope.selectedPlayer().player_id) === -1) {
                    continue;
                }
                if (action.action_type.indexOf("START_") >= 0 && (typeName === '' || action.action_type.split("START_")[1].toLowerCase() === typeName)){
                    return true;
                }
            }
            return false;
        };

        $scope.playerSkillAction = function playerSkillAction(x, y){
            for (let idx in $scope.game.state.available_actions){
                let action = $scope.game.state.available_actions[idx];
                if ($scope.agent_id != null && $scope.agent_id !== action.agent_id){
                    action.disabled = true;
                }
                let player =  $scope.local_state.board[y][x].player;
                if (player === null || action.player_ids.indexOf(player.player_id) === -1) {
                    continue;
                }
                if (action.action_type.indexOf("USE_SKILL") >= 0){
                    return $scope.title(action.skill).toLowerCase();
                }
            }
            return null;
        };

        $scope.newSquare = function newSquare(player_id, x, y, area, sub_area, number){
            let player = null;
            let player_icon = null;
            if (player_id != null){
                player = $scope.playersById[player_id];
                player_icon = player != null ? $scope.playerIcon(player) : null;
            }
            let ball = null;
            for (let i in $scope.game.state.pitch.balls){
                let ball_obj = $scope.game.state.pitch.balls[i];
                if (ball_obj.position != null && ball_obj.position.x === x && ball_obj.position.y === y){
                    ball = ball_obj;
                }
            }
            let big_guy = true;
            if (player !== null){
                big_guy = (BigGuyService.bigGuys.indexOf(player["role"]) > -1);
            }
            return {
                x: x,
                y: y,
                player: player,
                player_icon: player_icon,
                selected: false,
                available: false,
                path: null,
                action_type: undefined,
                on_path: false,
                rolls: [],
                special_actions: [],
                special_rolls: [],
                roll: false,
                block_dice: 0,
                area: area,
                sub_area: sub_area,
                ball: ball,
                num: number,
                big_guy: big_guy,
                foul_roll: null,
                handoff_roll: null
            };
        };

        $scope.clearSquareAction = function clearSquareAction(square) {
            square.available = false;
            square.d6_roll = 0;
            square.block_dice = 0;
            square.action_type = undefined;
        };

        $scope.actionButtonName = function actionButtonName(action_type){
            if (action_type === "USE_REROLL"){
                return 'Re-roll';
            } else if (action_type === "DONT_USE_REROLL"){
                return "Don't re-roll";
            }
            return $scope.prettify(action_type);
        };

        $scope.setAvailablePositions = function setAvailablePositions(){
            $scope.available_select_positions = [];
            $scope.available_move_positions = [];
            $scope.available_leap_positions = [];
            $scope.available_block_positions = [];
            $scope.available_handoff_positions = [];
            $scope.available_pass_positions = [];
            $scope.available_foul_positions = [];
            $scope.available_dodge_rolls = [];
            $scope.available_leap_rolls = [];
            $scope.available_block_dice = [];
            $scope.available_block_rolls = [];
            $scope.available_handoff_rolls = [];
            $scope.available_pass_rolls = [];
            $scope.available_players = [];
            $scope.available_interception_players = [];
            $scope.available_interception_rolls = [];
            $scope.special_actions = [];
            $scope.special_rolls = {};
            $scope.special_positions = {};
            $scope.available_select_rolls = [];
            $scope.main_action = null;
            $scope.blocked = false;
            $scope.available_paths = {};
            for (let idx in $scope.game.state.available_actions){
                let action = $scope.game.state.available_actions[idx];
                if (action.disabled){
                    continue;
                }
                if (action.positions.length > 0){
                    $scope.main_action = action;
                    // If an available player is selected
                    if (action.player_ids.length === 0 || ($scope.selectedPlayer() != null && action.player_ids.indexOf($scope.selectedPlayer().player_id) >= 0) || action.player_ids.length === 1){
                        if (action.hasOwnProperty('paths')){
                             $scope.available_paths[action.action_type] = action.paths;
                        }
                        if (action.action_type === "BLOCK") {
                            $scope.available_block_positions = action.positions;
                            $scope.available_block_dice = action.block_dice;
                            $scope.available_block_rolls = action.rolls;
                        } else if (action.action_type === "PASS"){
                            $scope.available_pass_positions = action.positions;
                            $scope.available_pass_rolls = action.rolls;
                            $scope.special_actions.push("PASS");
                            $scope.special_rolls["PASS"] = action.rolls;
                            $scope.special_positions["PASS"] = action.positions;
                        } else if (action.action_type === "HANDOFF"){
                            $scope.available_handoff_positions = action.positions;
                        } else if (action.action_type === "FOUL"){
                            $scope.available_foul_positions = action.positions;
                        } else if (action.action_type === "MOVE"){
                            $scope.available_move_positions = action.positions;
                            $scope.available_dodge_rolls = action.rolls;
                        } else if (action.action_type === "LEAP") {
                            $scope.special_actions.push("LEAP");
                            $scope.special_rolls["LEAP"] = action.rolls;
                            $scope.special_positions["LEAP"] = action.positions;
                        } else if (action.action_type === "STAB"){
                            $scope.special_actions.push("STAB");
                            $scope.special_rolls["STAB"] = action.rolls;
                            $scope.special_positions["STAB"] = action.positions;
                        } else if (action.action_type === "HYPNOTIC_GAZE"){
                            $scope.special_actions.push("HYPNOTIC_GAZE");
                            $scope.special_rolls["HYPNOTIC_GAZE"] = action.rolls;
                            $scope.special_positions["HYPNOTIC_GAZE"] = action.positions;
                        } else if (action.action_type === "PICKUP_TEAM_MATE"){
                            $scope.special_actions.push("PICKUP_TEAM_MATE");
                            $scope.special_rolls["PICKUP_TEAM_MATE"] = action.rolls;
                            $scope.special_positions["PICKUP_TEAM_MATE"] = action.positions;
                        } else if (action.action_type === "THROW_TEAM_MATE"){
                            $scope.special_actions.push("THROW_TEAM_MATE");
                            $scope.special_rolls["THROW_TEAM_MATE"] = action.rolls;
                            $scope.special_positions["THROW_TEAM_MATE"] = action.positions;
                            $scope.special_action_selected = action;
                        } else {
                            $scope.available_select_positions = action.positions;
                        }
                    }
                } else if (action.action_type === "STAND_UP"){
                    $scope.main_action = action;
                    let active_player_id = $scope.game.state.active_player_id;
                    let active_player = $scope.getPlayer(active_player_id);
                    let stand_up_position = active_player.position;
                    $scope.available_select_positions = [stand_up_position];
                    $scope.available_select_rolls = action.rolls;
                }
                if (action.action_type === "SELECT_PLAYER" && action.rolls.length > 0) {
                    $scope.available_interception_players = action.player_ids;
                    $scope.available_interception_rolls = action.rolls;
                    $scope.main_action = action;
                } else {
                    // Add available players
                    if (action.player_ids.length > 0){
                        if (!(action.action_type.startsWith("END_") || action.action_type.startsWith("START_"))) {
                            $scope.main_action = action;
                        }
                        if ($scope.available_players.length !== 1){
                            for (let p in action.player_ids){
                                $scope.available_players.push(action.player_ids[p]);
                            }
                        }
                    }
                    // Add player positions to available positions if none is selected
                    if ($scope.available_players.length > 1 && $scope.selectedPlayer() == null){
                        for (let p in action.player_ids){
                            $scope.available_select_positions.push($scope.local_state.player_positions[action.player_ids[p]]);
                        }
                    }
                }
            }
            // Select squares
            for (let i in $scope.available_select_positions){
                let position = $scope.available_select_positions[i];
                let roll = null;
                if ($scope.available_select_rolls.length > i){
                    roll = $scope.available_select_rolls[i];
                }
                // Reserves positions
                if (position == null && $scope.selected_square != null && $scope.selected_square.area === 'pitch'){
                    if ($scope.main_action.team_id === $scope.game.state.home_team.team_id){
                        for (let y = 0; y < $scope.local_state.home_dugout.length; y++){
                            for (let x = 0; x < $scope.local_state.home_dugout[y].length; x++){
                                if (y <= 7 && $scope.local_state.home_dugout[y][x].player == null){
                                    $scope.local_state.home_dugout[y][x].available = true;
                                    $scope.local_state.home_dugout[y][x].action_type = $scope.main_action.action_type;
                                }
                            }
                        }
                    } else if ($scope.main_action.team_id === $scope.game.state.away_team.team_id){
                        for (let y = 0; y < $scope.local_state.away_dugout.length; y++){
                            for (let x = 0; x < $scope.local_state.away_dugout[y].length; x++){
                                if (y <= 7 && $scope.local_state.away_dugout[y][x].player == null){
                                    $scope.local_state.away_dugout[y][x].available = true;
                                    $scope.local_state.away_dugout[y][x].action_type = $scope.main_action.action_type;
                                }
                            }
                        }
                    }
                    // Pitch positions
                } else if (position != null) {
                    $scope.local_state.board[position.y][position.x].available = true;
                    $scope.local_state.board[position.y][position.x].rolls = roll;
                    if ($scope.main_action !== null) {
                        $scope.local_state.board[position.y][position.x].action_type = $scope.main_action.action_type;
                    }
                }
                // Crowd in dugouts - available during pushes
                if (position != null){
                    if (position.x === 0 && position.y > 0 && position.y < $scope.local_state.board.length - 1){
                        $scope.local_state.away_dugout[position.y-1][1].available = true;
                        $scope.local_state.away_dugout[position.y-1][1].action_type = $scope.main_action.action_type;
                    }
                    if (position.x === $scope.local_state.board[0].length - 1 && position.y > 0 && position.y < $scope.local_state.board.length - 1){
                        $scope.local_state.home_dugout[position.y-1][0].available = true;
                        $scope.local_state.home_dugout[position.y-1][0].action_type = $scope.main_action.action_type;
                    }
                }
            }

            // Select player if only one available
            if ($scope.available_players.length === 1){
                $scope.select($scope.local_state.player_positions[$scope.available_players[0]])
            }

            // Pass squares
            if ($scope.available_pass_positions.length > 0) {
                let passer = $scope.getPlayer($scope.game.state.active_player_id);
                        let passerTeam = $scope.teamOfPlayer(passer);
                for (let i in $scope.available_pass_positions) {
                    let position = $scope.available_pass_positions[i];
                    let player = $scope.local_state.board[position.y][position.x].player;
                    // Only show square if player of interest of if throw team-mate
                    if ($scope.game.active_other_player_id !== null || (player !== null && $scope.teamOfPlayer(player).team_id === passerTeam.team_id && player.state.up)) {
                    if ($scope.game.active_other_player_id !== null || (player !== null && $scope.teamOfPlayer(player).team_id === passerTeam.team_id && player.state.up)) {
                            $scope.local_state.board[position.y][position.x].available = true;
                            $scope.local_state.board[position.y][position.x].action_type = "PASS";
                            if ($scope.available_pass_rolls.length > i) {
                                $scope.local_state.board[position.y][position.x].rolls = $scope.available_pass_rolls[i];
                            }
                        }
                    }
                }
            }

            // Interception squares
            for (let i in $scope.available_interception_players){
                let player_id = $scope.available_interception_players[i];
                let position = $scope.local_state.player_positions[player_id];
                $scope.local_state.board[position.y][position.x].available = true;
                $scope.local_state.board[position.y][position.x].action_type = "SELECT_PLAYER";
                if ($scope.available_interception_rolls.length > i){
                    $scope.local_state.board[position.y][position.x].rolls = $scope.available_interception_rolls[i];
                }
            }

            // Move squares
            for (let i in $scope.available_move_positions) {
                let path = null;
                if ($scope.available_paths["MOVE"].length > i){
                    path = $scope.available_paths["MOVE"][i];
                }
                let pos = $scope.available_move_positions[i];
                $scope.local_state.board[pos.y][pos.x].path = path;
                $scope.local_state.board[pos.y][pos.x].available = true;
                $scope.local_state.board[pos.y][pos.x].action_type = "MOVE";
                if ($scope.available_dodge_rolls.length > i){
                    $scope.local_state.board[pos.y][pos.x].rolls = $scope.available_dodge_rolls[i];
                }
            }

            // Block squares
            for (let i in $scope.available_block_positions) {
                let path = null;
                if ($scope.available_paths["BLOCK"].length > i){
                    path = $scope.available_paths["BLOCK"][i];
                }
                let position = $scope.available_block_positions[i];
                $scope.local_state.board[position.y][position.x].path = path;
                $scope.local_state.board[position.y][position.x].available = true;
                $scope.local_state.board[position.y][position.x].action_type = "BLOCK";
                if ($scope.available_block_dice.length > i){
                    $scope.local_state.board[position.y][position.x].block_dice = $scope.available_block_dice[i];
                }
                if ($scope.available_block_rolls.length > i){
                    $scope.local_state.board[position.y][position.x].rolls = $scope.available_block_rolls[i];
                }
            }

            // Foul squares
            for (let i in $scope.available_foul_positions) {
                let path = null;
                if ($scope.available_paths["FOUL"].length > i){
                    path = $scope.available_paths["FOUL"][i];
                }
                let position = $scope.available_foul_positions[i];
                $scope.local_state.board[position.y][position.x].path = path;
                $scope.local_state.board[position.y][position.x].available = true;
                $scope.local_state.board[position.y][position.x].action_type = "FOUL";
                $scope.local_state.board[position.y][position.x].available_foul_position = true;
            }

            // Hand-off squares
            for (let i in $scope.available_handoff_positions) {
                let path = null;
                if ($scope.available_paths["HANDOFF"].length > i){
                    path = $scope.available_paths["HANDOFF"][i];
                }
                let position = $scope.available_handoff_positions[i];
                $scope.local_state.board[position.y][position.x].path = path;
                $scope.local_state.board[position.y][position.x].available = true;
                $scope.local_state.board[position.y][position.x].action_type = "HANDOFF";
                if ($scope.available_dodge_rolls.length > i){
                    $scope.local_state.board[position.y][position.x].rolls = $scope.available_handoff_rolls[i];
                }
            }
            // Special actions: TODO: do like this for all actions
            for (let i in $scope.special_actions) {
                let action = $scope.special_actions[i];
                for (let j in $scope.special_positions[action]) {
                    let pos = $scope.special_positions[action][j];
                    $scope.local_state.board[pos.y][pos.x].special_actions.push(action);
                    if ($scope.special_rolls[action].length > j){  // TODO: Is this check necessary?
                        $scope.local_state.board[pos.y][pos.x].special_rolls[action] = $scope.special_rolls[action][j];
                    }
                }
            }
            // Reset special action to null if not available
            if ($scope.game.state.available_actions.length > 0 && ($scope.special_action_selected == null || $scope.special_actions.indexOf($scope.special_action_selected.action_type) < 0)){
                $scope.special_action_selected = null;
            }
        };

        $scope.getTile = function(x, y){
            let tile = $scope.game.arena.board[y][x];
            if (tile === "CROWD"){
                return "crowd";
            }
            return "pitch";
        };

        $scope.setLocalState = function setLocalState(){
            $scope.local_state.player_positions = {};
            $scope.local_state.balls = $scope.game.state.pitch.balls;
            $scope.local_state.current_team_id = $scope.game.state.current_team_id;
            for (let y = 0; y < $scope.game.state.pitch.board.length; y++){
                if ($scope.local_state.board.length <= y){
                    $scope.local_state.board.push([]);
                }
                for (let x = 0; x < $scope.game.state.pitch.board[y].length; x++){
                    let player_id = $scope.game.state.pitch.board[y][x];
                    let square = $scope.newSquare(player_id, x, y, $scope.getTile(x, y), '', undefined);
                    if (player_id != null){
                        $scope.local_state.player_positions[player_id] = square;
                    }
                    if ($scope.selected_square != null && $scope.selected_square.player != null && $scope.selected_square.player.player_id === player_id){
                        $scope.selected_square = square;
                    }
                    if ($scope.local_state.board[y].length <= x){
                        $scope.local_state.board[y].push(square);
                    } else {
                        $scope.local_state.board[y][x] = square;
                    }
                }
            }
            for (let y = 0; y < 15; y++){
                if ($scope.local_state.home_dugout.length <= y){
                    $scope.local_state.home_dugout.push([]);
                    $scope.local_state.away_dugout.push([]);
                }
                for (let x = 0; x < 2; x++){
                    let sub_area = '';
                    let home_player_id = null;
                    let home_square = null;
                    let away_player_id = null;
                    let away_square = null;
                    let idx = y*2+x;
                    if (y >= 12){
                        sub_area = 'casualties';
                        home_player_id = $scope.game.state.home_dugout.casualties[idx-12*2];
                        away_player_id = $scope.game.state.away_dugout.casualties[idx-12*2];
                    } else if (y >= 8){
                        sub_area = 'kod';
                        home_player_id = $scope.game.state.home_dugout.kod[idx-8*2];
                        away_player_id = $scope.game.state.away_dugout.kod[idx-8*2];
                    } else {
                        sub_area = 'reserves';
                        home_player_id = $scope.game.state.home_dugout.reserves[idx];
                        away_player_id = $scope.game.state.away_dugout.reserves[idx];
                    }
                    away_square = $scope.newSquare(away_player_id, x, y, 'dugout-away', sub_area);
                    home_square = $scope.newSquare(home_player_id, x, y, 'dugout-home', sub_area);
                    if ($scope.local_state.home_dugout[y].length <= x){
                        $scope.local_state.home_dugout[y].push(home_square);
                        $scope.local_state.away_dugout[y].push(away_square);
                    } else {
                        $scope.local_state.home_dugout[y][x] = home_square;
                        $scope.local_state.away_dugout[y][x] = away_square;
                    }
                }
            }
            for (let i = 0; i < $scope.game.squares_moved.length; i++){
                $scope.local_state.board[$scope.game.squares_moved[i].y][$scope.game.squares_moved[i].x].number = i
            }
        };

        $scope.newAction = function newAction(action_type){
            return {
                'player_id': null,
                'position': null,
                //'position_to': null,
                //'team_home': null,
                'team_id': $scope.team_id,
                'idx': -1,
                'action_type': action_type
            };
        };

        $scope.padWithZeroes = function padWithZeroes(number, length) {
            var my_string = '' + number;
            while (my_string.length < length) {
                my_string = '0' + my_string;
            }
            return my_string;
        };

        $scope.teamAgent = function teamAgent(team){
            if (team.team_id == $scope.game.state.home_team.team_id){
                return $scope.game.home_agent;
            } else if (team.team_id == $scope.game.state.away_team.team_id){
                return $scope.game.away_agent;
            }
            return null;
        };

        $scope.agentTeam = function agentTeam(agent){
            if (agent.agent_id == $scope.game.home_agent.agent_id){
                return $scope.game,state.home_team;
            } else if (agent.agent_id == $scope.game.away_agent.agent_id){
                return $scope.game,state.agent_team;
            }
            return null;
        };

        $scope.agentIdTeam = function agentIdTeam(agent_id){
            if (agent_id == $scope.game.home_agent.agent_id){
                return $scope.game,state.home_team;
            } else if (agent_id == $scope.game.away_agent.agent_id){
                return $scope.game,state.agent_team;
            }
            return null;
        };

        $scope.getTeamClock = function getTeamClock(team){
            for (var i=0; i < $scope.game.state.clocks.length; i++){
                let clock = $scope.game.state.clocks[i];
                if (team.team_id == clock.team_id){
                    return clock;
                }
            }
            return null;
        };

        $scope.getActiveClock = function getActiveClock(){
            let out = null;
            for (var i=0; i < $scope.game.state.clocks.length; i++){
                let c = $scope.game.state.clocks[i];
                if (!c.is_primary){
                     return c;
                } else {
                    out = c;
                }
            }
            return out;
        };

        $scope.getSecondsLeft = function getSecondsLeft(clock, ratio){
            let now = new Date() / 1000;
            let runningTime = clock.running_time;
            if (clock.is_running){
                // Calculate running time
                runningTime = now - clock.started_at - clock.paused_seconds;
            }
            let secondsLeft = clock.seconds - runningTime;
            if (ratio){
                return runningTime / clock.seconds;
            }
            return secondsLeft;
        };

        $scope.setClock = function setClock(){

            // Set width of turnmarkers
            let clock = $scope.getTeamClock($scope.game.state.away_team);
            if (clock != null){
                let awayTimeRatio = 1 - $scope.getSecondsLeft(clock, true);
                if (awayTimeRatio != null){
                    $('#turnmarker-away').width((awayTimeRatio * 100) + '%');
                }
            } else {
                $('#turnmarker-away').width('0%');
            }
            clock = $scope.getTeamClock($scope.game.state.home_team);
            if (clock != null){
                let homeTimeRatio = 1 - $scope.getSecondsLeft(clock, true);
                if (homeTimeRatio != null){
                    $('#turnmarker-home').width((homeTimeRatio * 100) + '%');
                }
            } else {
                $('#turnmarker-home').width('0%');
            }

            // Set clock to actor's time left
            let activeClock = $scope.getActiveClock();

            // Update clock
            if (activeClock != null){
                let secondsLeft = Math.max(0, $scope.getSecondsLeft(activeClock, false));
                var m = Math.floor(secondsLeft / 60);
                var s = Math.floor(secondsLeft % 60);
                $scope.$apply(function(){
                    $scope.clock = $scope.padWithZeroes(m, 2) + ":" + $scope.padWithZeroes(s, 2);
                });
            } else {
                $scope.$apply(function(){
                    $scope.clock = "";
                });
            }
        };

        $scope.checkForReload = function checkForReload(time){
            if ($scope.available_positions.length === 0 && !$scope.game.state.game_over){
                setTimeout(function(){
                    if (!$scope.loading && !$scope.refreshing){
                        if ($scope.opp_turn){
                            // It's opponent's turn
                            $scope.act($scope.newAction('CONTINUE'));
                        }
                    }
                }, time);
            }
        };

        $scope.select = function select(square){
            $scope.selected_square = square;
            if (square !== undefined) {
                if (square.area === 'pitch') {
                    $scope.local_state.board[square.y][square.x].selected = true;
                } else if (square.area === 'dugout-home') {
                    $scope.local_state.home_dugout[square.y][square.x].selected = true;
                } else if (square.area === 'dugout-away') {
                    $scope.local_state.away_dugout[square.y][square.x].selected = true;
                }
            };
        };

        $scope.resetSquares = function resetSquares(deselect){
            if (deselect){
                $scope.selected_square = null;
            }
            $scope.available_positions = [];
            for (let y = 0; y < $scope.local_state.board.length; y++){
                for (let x = 0; x < $scope.local_state.board[y].length; x++){
                    $scope.local_state.board[y][x].selected = false;
                    $scope.local_state.board[y][x].available = false;
                    $scope.local_state.board[y][x].roll = false;
                    $scope.local_state.board[y][x].block_dice = 0;
                    $scope.local_state.board[y][x].rolls = [];
                }
            }
            for (let y = 0; y < $scope.local_state.home_dugout.length; y++){
                for (let x = 0; x < $scope.local_state.home_dugout[y].length; x++){
                    $scope.local_state.home_dugout[y][x].selected = false;
                    $scope.local_state.home_dugout[y][x].available = false;
                }
            }
            for (let y = 0; y < $scope.local_state.away_dugout.length; y++){
                for (let x = 0; x < $scope.local_state.away_dugout[y].length; x++){
                    $scope.local_state.away_dugout[y][x].selected = false;
                    $scope.local_state.away_dugout[y][x].available = false;
                }
            }
        };

        $scope.selectedPlayer = function selectedPlayer(){
            if ($scope.selected_square != null){
                return $scope.selected_square.player;
            }
            return null;
        };

        $scope.playerReadyStateClass = function playerStateClass(player){
            if (player.state.heated || player.state.bone_headed || player.state.hypnotized || player.state.used){
                return "secondary";
            } else if (player.state.up){
                return "success";
            } if (player.state.stunned){
                return "warning";
            }
            return "default";
        };

        $scope.playerReadyStateName = function playerReadyStateName(player){
            if (player.state.heated){
                return "Heated";
            } else if (player.state.bone_headed){
                return "Bone Headed";
            } else if (player.state.hypnotized){
                return "Hypnotized";
            } else if (player.state.used){
                return "Used";
            } else if (player.state.up){
                return "";
            } else if (player.state.stunned){
                return "Stunned";
            }
            return "Down";
        };

        $scope.placeBall = function placeBall(square){
            // Remove ball first
            for (let y = 0; y < $scope.local_state.board.length; y++){
                for (let x = 0; x < $scope.local_state.board[y].length; x++){
                    $scope.local_state.board[y][x].ball = false;
                }
            }
            // Place ball
            ball = {position: square, on_ground: false, is_carried: false};
            $scope.game.state.balls[0] = ball;
            $scope.local_state.board[square.y][square.x].ball = ball;
        };

        $scope.create_action = function create_action(square){
            let action_type = $scope.getActionType(square);
            let player_id = $scope.selectedPlayer() == null ? null : $scope.selectedPlayer().player_id;
            //if (action_type === "PLACE_PLAYER"){
            //    player_id = square.player == null ? null : square.player.player_id;
            //}
            return {
                'player_id': player_id,
                // 'pos_from': $scope.selected_square != null && $scope.selected_square.area === 'pitch' ? {'x': $scope.selected_square.x, 'y': $scope.selected_square.y} : null,
                'position': square.area === 'pitch' || square.area === 'crowd' ? {'x': square.x, 'y': square.y} : null,
                'idx': -1,
                'action_type': action_type
            };
        };

        $scope.square = function square(square) {
            console.log("Click on: " + square);

            if (square === undefined){
                return;
            }

            // If position is available
            if ($scope.main_action != null && $scope.getAvailable(square)){

                // Select player
                if ($scope.main_action.action_type === "SELECT_PLAYER"){
                    $scope.selected_square = square;
                }

                // If action does not require a selected player or a player is selected
                if ($scope.main_action.player_ids.length === 0 || ($scope.selectedPlayer() != null)){

                    // If player is selected or only one player available
                    if ($scope.available_players.length <= 1 || $scope.selectedPlayer() != null){

                        // Convert dugout squares to pitch (crowd) squares if push procedure
                        let crowd = $scope.game.stack[$scope.game.stack.length-1] === "Push" && square.area.startsWith("dugout");
                        let crowd_square = {
                            y: square.y+1,
                            area: 'pitch',
                            action_type: square.action_type
                        };
                        if (crowd){
                            if (square.area === "dugout-away"){
                                crowd_square.x = 0;
                            } else if (square.area === "dugout-home"){
                                crowd_square.x = $scope.local_state.board[0].length-1;
                            }
                        }
                        // Otherwise - send action
                        let action = null;
                        if (crowd){
                            action = $scope.create_action(crowd_square);
                        } else {
                            action = $scope.create_action(square);
                        }
                        $scope.act(action);
                    } else if (square.player != null && $scope.selectedPlayer() != null && $scope.selectedPlayer().player_id === square.player.player_id){
                        // Only deselect if other players are available
                        if ($scope.available_players.length !== 1) {
                            $scope.resetSquares(true);
                            $scope.setAvailablePositions();
                        }
                        return;
                    } else if ($scope.main_action.action_type === "PLACE_BALL"){
                        $scope.placeBall(square);
                        $scope.resetSquares(true);
                        $scope.setAvailablePositions();
                    }
                }
            }
            if (square.player == null){
                // Clicked on an empty square with no selected player
                if ($scope.available_players.length !== 1) {
                    $scope.resetSquares(true);
                }
                $scope.setAvailablePositions();
            } else {
                // Clicked on a player - select it - unless only non-player actions
                if ($scope.main_action == null || $scope.main_action.action_type !== "PLACE_BALL"){
                    if ($scope.available_players.length !== 1){
                        $scope.resetSquares(true);
                        $scope.select(square);
                    }
                    $scope.setAvailablePositions();
                }
            }

        };

        $scope.squareHover = function squareHover(square) {
            if (square.player != null){
                $scope.hover_player = square.player;
            } else {
                $scope.hover_player = null;
            }
            $scope.resetPaths();
            if (square.path !== null){
                $scope.refreshPaths(square.path);
            }
        };

        $scope.resetPaths = function resetPaths(){
            for (let y = 0; y < $scope.game.state.pitch.board.length; y++) {
                for (let x = 0; x < $scope.game.state.pitch.board[y].length; x++) {
                    if ($scope.local_state.board[y][x].path !== null) {
                        $scope.local_state.board[y][x].rolls = [];
                        $scope.local_state.board[y][x].on_path = false;
                        //$scope.local_state.board[y][x].block_dice = 0;
                        $scope.local_state.board[y][x].foul_roll = null;
                        $scope.local_state.board[y][x].handoff_roll = null;
                    }
                }
            }
        };

        $scope.refreshPaths = function refreshPaths(path) {
            for (let i=0; i < path.steps.length; i++){
                let x = path.steps[i].x;
                let y = path.steps[i].y;
                $scope.local_state.board[y][x].rolls = path.rolls[i];
                $scope.local_state.board[y][x].on_path = true;
                if (i === path.steps.length-1){
                    $scope.local_state.board[y][x].block_dice = path.block_dice;
                    if (path.foul_roll !== null){
                        $scope.local_state.board[y][x].rolls = [path.foul_roll];
                        $scope.local_state.board[y][x].foul_roll = path.foul_roll;
                    }
                    if (path.handoff_roll !== null){
                        $scope.local_state.board[y][x].rolls = [path.handoff_roll];
                        $scope.local_state.board[y][x].handoff_roll = path.handoff_roll;
                    }
                }
            }
        };

        $scope.currentProc = function currentProc(team){
            if ($scope.loading){
                return "";
            } else if ($scope.game.stack[$scope.game.stack.length-1] === "Pregame" && team == null){
                return "Pre-Game";
            } else if ($scope.game.stack[$scope.game.stack.length-1] === "WeatherTable" && team == null){
                return "Pre-Game";
            } else if ($scope.game.stack[$scope.game.stack.length-1] === "CoinToss" && team == null){
                return "Coin Toss";
            } else if ($scope.game.stack[$scope.game.stack.length-1] === "PostGame" && team == null){
                return "Post-Game";
            } else if ($scope.game.stack[$scope.game.stack.length-1] === "QuickSnap"){
                if (team != null && team.team_id === $scope.game.state.current_team_id){
                    return "Quick Snap!";
                } else if (team == null){
                    if ($scope.game.state.half === 1 && team == null){
                        return "1st half";
                    } else if ($scope.game.state.half === 2 && team == null){
                        return "2nd half";
                    }
                }
            } else if ($scope.game.stack[$scope.game.stack.length-1] === "Blitz"){
                if (team != null && team.team_id === $scope.game.state.current_team_id){
                    return "Blitz!";
                } else if (team == null){
                    if ($scope.game.state.half === 1 && team == null){
                        return "1st half";
                    } else if ($scope.game.state.half === 2 && team == null){
                        return "2nd half";
                    }
                }
            } else if ($scope.game.state.game_over && team == null){
                return "Game over";
            } else if ($scope.game.state.half === 1 && team == null){
                return "1st half";
            } else if ($scope.game.state.half === 2 && team == null){
                return "2nd half";
            }
            if (team != null && team === $scope.game.state.home_team){
                return $scope.game.state.home_team.state.turn + " / " + $scope.game.rounds;
            } else if (team != null && team !== $scope.game.state.home_team){
                return $scope.game.state.away_team.state.turn + " / " + $scope.game.rounds;
            }
            return "";
        };

        $scope.platerStateText = function platerStateText(state) {
            if (state.stunned){
                return "Stunned";
            } else if (state.bone_headed){
                return "Bone Headed";
            } else if (state.really_stupid){
                return "Really Stupid";
            } else if (state.heated){
                return "Heated";
            }
            return "";
        };

        $scope.playerInFocus = function playerInFocus(team) {
            if ($scope.hover_player != null){
                let player = $scope.hover_player;
                if (player != null && team.team_id === $scope.teamOfPlayer(player).team_id){
                    return player;
                }
            }
            if ($scope.selected_square != null && $scope.selected_square.player != null){
                if (team.team_id === $scope.teamOfPlayer($scope.selected_square.player).team_id){
                    return $scope.selected_square.player;
                }
            }
            return null;
        };

        $scope.prettify = function prettify(text){
            let pretty = text.replace("SETUP_FORMATION_", "").toLowerCase().split("_").join(" ");
            return pretty.charAt(0).toUpperCase() + pretty.slice(1);
        };

        $scope.showAllReports = function showAllReports(){
            $scope.reportsLimit=10000;
        };

        $scope.disableOppActions = function disableOppActions(){
            $scope.opp_turn = true;
            for (let idx in $scope.game.state.available_actions) {
                let action = $scope.game.state.available_actions[idx];
                if ($scope.replaying || action.disabled || $scope.spectating || ($scope.team_id !== undefined && action.team_id !== $scope.team_id)) {
                    action.disabled = true;
                    continue;
                }
                $scope.opp_turn = false;
            }
        };

        $scope.lastReportIdx = 0;

        $scope.act = function act(action){
            if ($scope.loading || $scope.refreshing){
                return;
            }
            console.log(action);
            if (action.action_type === "END_TURN" || action.action_type === "PASS"){
                $scope.resetSquares(true);
            }
            $scope.refreshing = true;
            $scope.reportsLimit = 20;
            GameService.act($scope.game.game_id, action).success(function(data) {
                $scope.game = data;
                $scope.disableOppActions();
                console.log(data);
                $scope.playersById = Object.assign({}, $scope.game.state.home_team.players_by_id, $scope.game.state.away_team.players_by_id);
                $scope.setLocalState();
                $scope.setAvailablePositions();
                //$scope.updateClock();
                //$scope.updateMoveLines();
                $scope.refreshing = false;
                document.getElementById('gamelog').scrollTop = 0;
                let time = 10;
                if ($scope.game.state.reports.length > 0){
                    let newestReport = $scope.game.state.reports[$scope.game.state.reports.length-1];
                    if (newestReport.outcome_type in GameLogService.log_timouts){
                        if ($scope.game.state.reports.length === $scope.lastReportIdx){
                            time = 100;
                        } else if ($scope.game.state.reports[$scope.game.state.reports.length-1].outcome_type in GameLogService.log_timouts){
                            time = GameLogService.log_timouts[$scope.game.state.reports[$scope.game.state.reports.length-1]];
                        }
                        $scope.lastReportIdx = $scope.game.state.reports.length;
                    }
                }
                if (time !== null){
                    $scope.checkForReload(time);
                }
                $scope.saved = false;
                $scope.blocked = false;
            }).error(function(status, data) {
                $location.path("/#/");
            });
        };

        $scope.pickActionType = function pickActionType(action){
            if (action.action_type === "PLACE_BALL" && $scope.local_state.balls.length > 0){
                let a = $scope.newAction(action.action_type);
                a.position = $scope.local_state.balls[0].position;
                $scope.act(a);
            } else if (action.player_ids.length > 0 && $scope.selectedPlayer != null && action.player_ids.indexOf($scope.selectedPlayer().player_id) >= 0){
                let a = $scope.newAction(action.action_type);
                a.player_id = $scope.selectedPlayer().player_id;
                $scope.act(a);
            } else if (action.positions.length === 0){
                let a = $scope.newAction(action.action_type);
                $scope.act(a);
            }
        };

        $scope.showActionAsDice = function showActionAsDice(action) {
            if (action.action_type === "SELECT_ATTACKER_DOWN") {
                return true;
            }
            if (action.action_type === "SELECT_PUSH") {
                return true;
            }
            if (action.action_type === "SELECT_BOTH_DOWN") {
                return true;
            }
            if (action.action_type === "SELECT_DEFENDER_STUMBLES") {
                return true;
            }
            if (action.action_type === "SELECT_DEFENDER_DOWN") {
                return true;
            }
            return false;
        };

        $scope.showActionAsSpecialButton = function showActionAsSpecialButton(action) {
            return $scope.special_actions.indexOf(action.action_type) > -1;
        };

        $scope.showActionAsButton = function showActionAsButton(action) {
            if (action.action_type === "SELECT_ATTACKER_DOWN"){
                return false;
            }
            if (action.action_type === "SELECT_PUSH"){
                return false;
            }
            if (action.action_type === "SELECT_BOTH_DOWN"){
                return false;
            }
            if (action.action_type === "SELECT_DEFENDER_STUMBLES"){
                return false;
            }
            if (action.action_type === "SELECT_DEFENDER_DOWN"){
                return false;
            }
            if (action.action_type === "USE_SKILL"){
                return false;
            }
            if (action.action_type === "DONT_USE_SKILL"){
                for (let idx in $scope.game.state.available_actions) {
                    let a = $scope.game.state.available_actions[idx];
                    if (a.action_type === "USE_SKILL"){
                        return false;  // Dump off exception
                    }
                }
                return true;
            }
            if (action.action_type !== "START_GAME" && action.action_type.indexOf("START_") > -1){
                return false;
            }
            // If no args -> show
            if (action.player_ids.length === 0 && action.positions.length === 0){
                return true;
            }
            if (action.player_ids.length > 0 && $scope.selected_square != null && $scope.selectedPlayer() != null && action.player_ids.indexOf($scope.selectedPlayer().player_id) >= 0 && action.positions.length === 0){
                return true;
            }
            if (action.player_ids.length === 0 && action.positions.length > 0){
                if (action.action_type === "PLACE_BALL" && $scope.local_state.balls.length > 0){
                    return true;
                }
            }
            return false;
        };

        $scope.updateMoveLines = function updateMoveLines() {
            let lastX = null;
            let lastY = null;
            $( ".moveline" ).sort(function (a, b) {
                return parseInt(a.id.replace('moveline-', '')) > parseInt(b.id.replace('moveline-', ''));
            }).each(function() {
                let x = parseInt($( this ).attr('X'));
                let y = parseInt($( this ).attr('Y'));
                if (lastX !== null){
                    let yDir = y - lastY;
                    let xDir = x - lastX;
                    $( this ).attr('x1', 30 - xDir*30);
                    $( this ).attr('y1', 30 - yDir*30);
                    console.log(x)
                }
                lastX = x;
                lastY = y;
            });
        };
        
        $scope.runTimeLoop = function runTimeLoop(time, game_id){
            if ($scope.game.state.game_over){
                $scope.setClock();
                return;
            }
            setTimeout(function(){
                if (!window.location.href.includes(game_id)){
                    return;
                }
                $scope.setClock();
                var clock = $scope.getActiveClock();
                if (clock != null){
                    if ($scope.getSecondsLeft(clock, false) < 0 && !$scope.refreshing){
                        $scope.reload();
                    } else {
                        $scope.runTimeLoop(time, game_id);
                    }
                } else {
                    $scope.runTimeLoop(time, game_id);
                }
            }, time);
        };

        $scope.loadReplaySteps = function loadReplaySteps(){
            $scope.loadingSteps = true;
            ReplayService.getSteps($scope.replay_id, $scope.numOfSteps(), 10).success(function (data) {
                for (let key in data){
                    $scope.replay.steps[key] = data[key];
                }
                if (Object.keys(data).length == 0){
                    $scope.replayDoneLoading = true;
                }
                $scope.loadingSteps = false;
            }).error(function (status, data) {
                $location.path("/#/");
                $scope.loadingSteps = false;
            });
        };

        $scope.runPlayLoop = function runPlayLoop(){
            setTimeout(function(){
                if (!$scope.replayDoneLoading && !$scope.loadingSteps){
                    $scope.loadReplaySteps();
                }
                if ($scope.replayIsPlaying) {
                    $scope.nextStep();
                    $scope.$apply();
                }
                $scope.runPlayLoop();
            }, $scope.replaySpeed);
        };

        $scope.numOfSteps = function (){
            return Object.keys($scope.replay.steps).length;
        };

        $scope.nextStep = function (){
            if ($scope.replayStep + 1 < $scope.numOfSteps()){
                $scope.replayStep += 1;
                let stepId = Object.keys($scope.replay.steps)[$scope.replayStep];
                $scope.game = $scope.replay.steps[stepId];
                $scope.disableOppActions();
                $scope.playersById = Object.assign({}, $scope.game.state.home_team.players_by_id, $scope.game.state.away_team.players_by_id);
                $scope.setLocalState();
                $scope.setAvailablePositions();
            } else {
                $scope.replayIsPlaying = false;
            }
        };

        $scope.playOrPause = function(){
            $scope.replayIsPlaying = !$scope.replayIsPlaying;
        };

        $scope.prevStep = function (){
            if ($scope.replayStep > 0){
                $scope.replayStep -= 1;
                let stepId = Object.keys($scope.replay.steps)[$scope.replayStep];
                $scope.game = $scope.replay.steps[stepId];
                $scope.disableOppActions();
                $scope.playersById = Object.assign({}, $scope.game.state.home_team.players_by_id, $scope.game.state.away_team.players_by_id);
                $scope.setLocalState();
                $scope.setAvailablePositions();
            } else {
                $scope.replayIsPlaying = false;
            }
        };

        $scope.reload = function reload(){
            $scope.refreshing = true;
            if ($scope.replaying){

                ReplayService.get($scope.replay_id).success(function (data) {
                    $scope.replay = data;
                    $scope.game = data.steps[0];
                    $scope.disableOppActions();
                    $scope.playersById = Object.assign({}, $scope.game.state.home_team.players_by_id, $scope.game.state.away_team.players_by_id);
                    $scope.setLocalState();
                    $scope.setAvailablePositions();
                    //$scope.updateMoveLines();
                    $scope.loading = false;
                    $scope.refreshing = false;
                    console.log(data);
                    $scope.saved = false;
                    $scope.blocked = false;
                    $scope.runPlayLoop();
                }).error(function (status, data) {
                    $location.path("/#/");
                });

            } else {

                GameService.get($scope.game_id).success(function (data) {
                    $scope.game = data;
                    $scope.disableOppActions();
                    $scope.playersById = Object.assign({}, $scope.game.state.home_team.players_by_id, $scope.game.state.away_team.players_by_id);
                    $scope.setLocalState();
                    $scope.setAvailablePositions();
                    //$scope.updateMoveLines();
                    $scope.loading = false;
                    $scope.refreshing = false;
                    console.log(data);
                    $scope.checkForReload(2500);
                    $scope.saved = false;
                    $scope.blocked = false;
                    $scope.runTimeLoop(20, data.game_id);
                }).error(function (status, data) {
                    $location.path("/#/");
                });

            }

        };

        // Get game-state when document is ready
        $( document ).ready(function() {
            $scope.reload();
        });

    }
]);

appDirectives.directive('displayMessage', function() {
	return {
		restrict: 'E',
		scope: {
        	messageType: '=type',
        	message: '=data'
      	},
		template: '<div class="alert {{messageType}}">{{message}}</div>',
		link: function (scope, element, attributes) {
            scope.$watch(attributes, function (value) {
            	console.log(attributes);
            	console.log(value);
            	console.log(element[0]);
                element[0].children.hide(); 
            });
        }
	}
});
appFilters.filter('range', function() {
  return function(input, total) {
    total = parseInt(total);
    for (var i=0; i<total; i++)
      input.push(i);
    return input;
  };
});

appFilters.filter('board', function() {
  return function(input) {
    total = parseInt(total);
    for (var i=0; i<total; i++)
      input.push(i);
    return input;
  };
});

app.filter('reverse', function() {
  return function(items) {
    return items.slice().reverse();
  };
});

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
