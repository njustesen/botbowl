'''
A number of static methods for interpretting the state of the fantasy football pitch that aren't required directly by
the client
'''
import ffai.core.game as g
import ffai.core.table as t
import ffai.core.model as m
from typing import Optional, List
import math


class ActionSequence():
    ''' Class containing a single possible move of a single player.
    '''

    def __init__(self, action_steps: List[m.Action], score: float = 0, description: str = ''):
        ''' Creates a new ActionSequence - an ordered list of sequential m.Actions to attempt to undertake.
        :param action_steps: Sequence of action steps that form this action.
        :param score: A score representing the attractiveness of the move (default: 0)
        :param description: A debug string (defaul: '')
        '''

        # Note the intention of this object is that when the object is acting, as steps are completed,
        # they are removed from the move_sequence so the next move is always the top of the move_sequence
        # list.

        self.action_steps = action_steps
        self.score = score
        self.description = description

    def is_valid(self, game: g.Game) -> bool:
        ''' Check if move can be executed given current game state.
        Checks if the object represents a valid, executable sequence given current game state.  For example, steps must
        be adjacent, and begin adjacent to players present position.  Player must be available to move (another player
        is not already moving, player has not already moved) etc.
        :param game:
        :return: True if controlling bot program *should* be able to execute the set of steps represented, else False
        '''
        pass

    def popleft(self):
        val = self.action_steps[0]
        del self.action_steps[0]
        return val


def blitz_used(game: g.Game) -> bool:
    for action in game.state.available_actions:
        if action.action_type == t.ActionType.START_BLITZ:
            return False
    return True


def handoff_used(game: g.Game) -> bool:
    for action in game.state.available_actions:
        if action.action_type == t.ActionType.START_HANDOFF:
            return False
    return True


def foul_used(game: g.Game) -> bool:
    for action in game.state.available_actions:
        if action.action_type == t.ActionType.START_FOUL:
            return False
    return True


def pass_used(game: g.Game) -> bool:
    for action in game.state.available_actions:
        if action.action_type == t.ActionType.START_PASS:
            return False
    return True


def get_players(game: g.Game, team: m.Team, include_own: bool = True, include_opp: bool = True, include_stunned: bool = True, include_used: bool = True, include_off_pitch: bool = False, only_blockable: bool = False, only_used: bool = False) -> List[m.Player]:
    players: List[m.Player] = []
    selected_players: List[m.Player] = []
    for iteam in game.state.teams:
        if iteam == team and include_own:
            players.extend(iteam.players)
        if iteam != team and include_opp:
            players.extend(iteam.players)
    for player in players:
        if only_blockable and not player.state.up:
            continue
        if only_used and not player.state.used:
            continue

        if include_stunned or not player.state.stunned:
            if include_used or not player.state.used:
                if include_off_pitch or (player.position is not None and not game.state.pitch.is_out_of_bounds(player.position)):
                    selected_players.append(player)

    return selected_players


def caging_squares_north_east(game: g.Game, protect_square: m.Square) -> List[m.Square]:
    '''
     * WARNING:  The following 4 methods have yet to be tested properly.
     *
     * At it's simplest, a cage requires 4 platers in the North-East, South-East, South-West and North-West
     * positions, relative to the ball carrier, such that there is no more than 2 squares between the players in
     * each of those adjacent compass directions.
     *
     *   1     2
     *    xx-xx
     *    xx-xx
     *    --o--
     *    xx-xx
     *    xx-xx
     *   3     4
     *
     * pitch is 26 long
     *
     *
     * Basically we need one player in each of the corners: 1-4, but spaced such that there is no gap of 3 squares.
     * If the caging player is in 1-4, but next to ball carrier, he ensures this will automatically be met.
     *
     * The only exception to this is when the ball carrier is on, or near, the sideline.  Then return the squares
     * that can otherwise form the cage.
     *

    '''
    caging_squares: List[m.Square] = []
    x = protect_square.x
    y = protect_square.y

    if x <= game.state.pitch.width - 3:
        if y == game.state.pitch.height-2:
            caging_squares.append(game.state.pitch.get_square(x + 1, y + 1))
            caging_squares.append(game.state.pitch.get_square(x + 2, y + 1))
            caging_squares.append(game.state.pitch.get_square(x + 1, y))
            caging_squares.append(game.state.pitch.get_square(x + 2, y))
        elif y == game.state.pitch.height-1:
            caging_squares.append(game.state.pitch.get_square(x + 1, y))
            caging_squares.append(game.state.pitch.get_square(x + 2, y))
        else:
            caging_squares.append(game.state.pitch.get_square(x + 1, y + 1))
            caging_squares.append(game.state.pitch.get_square(x + 1, y + 2))
            caging_squares.append(game.state.pitch.get_square(x + 2, y + 1))
            caging_squares.append(game.state.pitch.get_square(x + 2, y + 2))

    return caging_squares


def caging_squares_north_west(game: g.Game, protect_square: m.Square) -> List[m.Square]:

    caging_squares: List[m.Square] = []
    x = protect_square.x
    y = protect_square.y

    if x >= 3:
        if y == game.state.pitch.height-2:
            caging_squares.append(game.state.pitch.get_square(x - 1, y + 1))
            caging_squares.append(game.state.pitch.get_square(x - 2, y + 1))
            caging_squares.append(game.state.pitch.get_square(x - 1, y))
            caging_squares.append(game.state.pitch.get_square(x - 2, y))
        elif y == game.state.pitch.height-1:
            caging_squares.append(game.state.pitch.get_square(x - 1, y))
            caging_squares.append(game.state.pitch.get_square(x - 2, y))
        else:
            caging_squares.append(game.state.pitch.get_square(x - 1, y + 1))
            caging_squares.append(game.state.pitch.get_square(x - 1, y + 2))
            caging_squares.append(game.state.pitch.get_square(x - 2, y + 1))
            caging_squares.append(game.state.pitch.get_square(x - 2, y + 2))

    return caging_squares


def caging_squares_south_west(game: g.Game, protect_square: m.Square) -> List[m.Square]:

    caging_squares: List[m.Square] = []
    x = protect_square.x
    y = protect_square.y

    if x >= 3:
        if y == 2:
            caging_squares.append(game.state.pitch.get_square(x - 1, y - 1))
            caging_squares.append(game.state.pitch.get_square(x - 2, y - 1))
            caging_squares.append(game.state.pitch.get_square(x - 1, y))
            caging_squares.append(game.state.pitch.get_square(x - 2, y))
        elif y == 1:
            caging_squares.append(game.state.pitch.get_square(x - 1, y))
            caging_squares.append(game.state.pitch.get_square(x - 2, y))
        else:
            caging_squares.append(game.state.pitch.get_square(x - 1, y - 1))
            caging_squares.append(game.state.pitch.get_square(x - 1, y - 2))
            caging_squares.append(game.state.pitch.get_square(x - 2, y - 1))
            caging_squares.append(game.state.pitch.get_square(x - 2, y - 2))

    return caging_squares


def caging_squares_south_east(game: g.Game, protect_square: m.Square) -> List[m.Square]:

    caging_squares: List[m.Square] = []
    x = protect_square.x
    y = protect_square.y

    if x <= game.state.pitch.width-3:
        if y == 2:
            caging_squares.append(game.state.pitch.get_square(x + 1, y - 1))
            caging_squares.append(game.state.pitch.get_square(x + 2, y - 1))
            caging_squares.append(game.state.pitch.get_square(x + 1, y))
            caging_squares.append(game.state.pitch.get_square(x + 2, y))
        elif y == 1:
            caging_squares.append(game.state.pitch.get_square(x + 1, y))
            caging_squares.append(game.state.pitch.get_square(x + 2, y))
        else:
            caging_squares.append(game.state.pitch.get_square(x + 1, y - 1))
            caging_squares.append(game.state.pitch.get_square(x + 1, y - 2))
            caging_squares.append(game.state.pitch.get_square(x + 2, y - 1))
            caging_squares.append(game.state.pitch.get_square(x + 2, y - 2))

    return caging_squares


def is_caging_position(game: g.Game, player: m.Player, protect_player: m.Player) -> bool:
    return player.position.distance(protect_player.position) <= 2 and not is_castle_position_of(player, protect_player)


def has_player_within_n_squares(game: g.Game, units: List[m.Player], square: m.Square, num_squares: int) -> bool:
    for cur in units:
        if cur.position.distance(square) <= num_squares:
            return True
    return False


def has_adjacent_player(game: g.Game, square: m.Square) -> bool:
    return not game.state.pitch.adjacent_players(square)


def is_castle_position_of(game: g.Game, player1: m.Player, player2: m.Player) -> bool:
    return player1.position.x == player2.position.x or player1.position.y == player2.position.y


def is_bishop_position_of(game: g.Game, player1: m.Player, player2: m.Player) -> bool:
    return abs(player1.position.x - player2.position.x) == abs(player1.position.y - player2.position.y)


def attacker_would_surf(game: g.Game, attacker: m.Player, defender: m.Player) -> bool:
    if (defender.has_skill(t.Skill.SIDE_STEP) and not attacker.has_skill(t.Skill.GRAB)) or defender.has_skill(t.Skill.STAND_FIRM):
        return False

    if not attacker.position.is_adjacent(defender.position):
        return False

    return direct_surf_squares(game, attacker.position, defender.position)


def direct_surf_squares(game: g.Game, attackFrom: m.Square, defendFrom: m.Square) -> bool:
    defenderOnSideline = on_sideline(game, defendFrom)
    defenderInEndzone = on_endzone(game, defendFrom)

    if defenderOnSideline and defendFrom.x == attackFrom.x:
        return True

    if defenderInEndzone and defendFrom.y == attackFrom.y:
        return True

    if defenderInEndzone and defenderOnSideline:
        return True

    return False


def reverse_x_for_right(game: g.Game, team: m.Team, x: int) -> int:
    if not game.is_team_side(m.Square(13, 3), team):
        x = game.state.pitch.width - 1 - x
    return x


def reverse_x_for_left(game: g.Game, team: m.Team, x: int) -> int:
    if game.is_team_side(m.Square(13, 3), team):
        x = game.state.pitch.width - 1 - x
    return x


def on_sideline(game: g.Game, square: m.Square) -> bool:
    return square.y == 1 or square.y == game.state.pitch.height - 1


def on_endzone(game: g.Game, square: m.Square) -> bool:
    return square.x == 1 or square.x == game.state.pitch.width - 1


def on_los(game: g.Game, team: g.Team, square: m.Square) -> bool:
    return (reverse_x_for_right(game, team, square.x) == 13) and square.y > 4 and square.y < 21


def los_squares(game: g.Game, team: g.Team) -> List[m.Square]:
    squares: List[m.Square] = []
    squares.append(game.state.pitch.get_square(reverse_x_for_right(13), 5))
    squares.append(game.state.pitch.get_square(reverse_x_for_right(13), 6))
    squares.append(game.state.pitch.get_square(reverse_x_for_right(13), 7))
    squares.append(game.state.pitch.get_square(reverse_x_for_right(13), 8))
    squares.append(game.state.pitch.get_square(reverse_x_for_right(13), 9))
    squares.append(game.state.pitch.get_square(reverse_x_for_right(13), 10))
    squares.append(game.state.pitch.get_square(reverse_x_for_right(13), 11))
    return squares


def distance_to_sideline(game: g.Game, square: m.Square) -> int:
    '''
    distance_to_sideline.  0 indicates on sideline.
    :param game:
    :param square:
    :return:
    '''
    return min(square.y - 1, game.state.pitch.height - square.y - 1)


def is_endzone(game, square: m.Square) -> bool:
    return square.x == 1 or square.x == game.state.pitch.width - 1


def is_adjacent_ball(game: g.Game, square: m.Square) -> bool:
    ball_square = game.state.pitch.get_ball_position()
    return ball_square is not None and ball_square.is_adjacent(square)


def squares_within(game: g.Game, square: m.Square, distance: int) -> List[m.Square]:
    squares: List[m.Square] = []
    x = square.x
    y = square.y
    for i in range(-distance-1, distance+1):
        for j in range(-distance-1, distance+1):
            cur_square = game.state.pitch.get_square(square.x+i, square.y+i)
            if cur_square != square and not game.state.pitch.is_out_of_bounds(cur_square):
               squares.append(cur_square)
    return squares


def distance_to_defending_endzone(game: g.Game, team: m.Team, position: m.Square) -> int:
    return reverse_x_for_right(game, team, position.x) - 1


def distance_to_scoring_endzone(game: g.Game, team: m.Team, position: m.Square) -> int:
    return game.state.pitch.width - 1 - reverse_x_for_right(game, team, position.x)


def players_in_scoring_endzone(game: g.Game, team: m.Team, include_own: bool = True, include_opp: bool = False) -> List[m.Player]:
    players: List[m.Player] = []
    for cur_team in game.state.teams:
        if cur_team == team and not include_own:
            break
        if cur_team != team and not include_opp:
            break
        for cur_player in team.players:
            if in_scoring_endzone(game, cur_player.team, cur_player.position):
                players.append(cur_player)
    return players


def in_scoring_endzone(game: g.Game, team: m.Team, square: m.Square) -> bool:
    return reverse_x_for_right(game, team, square.x) == game.state.pitch.width - 1


def players_in_scoring_distance(game: g.Game, team: m.Team, include_own: bool = True, include_opp: bool = True, include_stunned: bool = False) -> List[m.Player]:
    players: List[m.Player] = get_players(game, team, include_own=include_own, include_opp=include_opp, include_stunned=include_stunned)
    selected_players: List[m.Player] = []
    for player in players:
        if distance_to_scoring_endzone(game, team, player.position) <= player.move_allowed(): selected_players.append(player)
    return selected_players


def distance_to_nearest_player(game: g.Game, team: m.Team, square: m.Square, include_own: bool = True, include_opp: bool = True, only_used: bool = False, include_used: bool = True, include_stunned: bool = True, only_blockable: bool = False) -> int:
    opps: List[m.Player] = get_players(game, team, include_own=include_own, include_opp=include_opp, only_used=only_used, include_used=include_used, include_stunned=include_stunned, only_blockable=only_blockable)
    curMax = 100
    for opp in opps:
        dist = opp.position.distance(square)
        curMax = min(curMax, dist)
    return curMax


def screening_distance(game: g.Game, from_square: m.Square, to_square : m.Square) -> float:
    # Return the "screening distance" between 2 squares.  (To complete)
    #float dist =math.sqrt(math.pow(square.x - cur.position.x, 2) + math.pow(square.y - cur.position.y, 2))
    return 0


def num_opponents_can_reach(game: g.Game, team: m.Team, square: m.Square) -> int:
    opps: List[m.Player] = get_players(game, team, include_own=False, include_opp=True)
    numOppsReach: int = 0
    for cur in opps:
        dist = max(square.x - cur.position.x, square.y - cur.position.y)
        if cur.state.stunned: continue
        moveAllowed = cur.get_ma() + 2
        if not cur.state.up: moveAllowed -= 3
        if dist < moveAllowed: numOppsReach+=1
    return numOppsReach


def num_opponents_on_field(game: g.Game, team: m.Team) -> int:
    opps: List[m.Player] = get_players(game, team, include_own=False, include_opp=True)
    numOpps = 0
    for cur in opps:
        if cur.position is not None: numOpps+=1
    return numOpps


def number_opponents_closer_than_to_endzone (game: g.Game, team: g.Team, square: m.Square) -> int:
    opponents: List[m.Player] = get_players(game, team, include_own=False, include_opp=True)
    num_opps = 0
    distance_square_endzone = distance_to_defending_endzone(game, team, square)

    for opponent in opponents:
        distance_opponenent_endzone = distance_to_defending_endzone(game, team, opponent.position)
        if distance_opponenent_endzone < distance_square_endzone: num_opps += 1
    return num_opps


def in_scoring_range(game: g.Game, player: m.Player) -> bool:

    dist_to_endzone = distance_to_scoring_endzone(game, player.team, player.position)
    ma = player.move_allowed()
    return player.move_allowed() >= dist_to_endzone


def players_in_scoring_range(game: g.Game, team: m.Team, include_own=True, include_opp=True, include_used=True, include_stunned=True) -> List[m.Player]:
    players: List[m.Player] = get_players(game, team, include_own=include_own, include_opp=include_opp, include_stunned=include_stunned, include_used=include_used)
    in_scoring_range: List[m.Player] = []
    for player in players:
        if distance_to_defending_endzone(game, player.team, player.position) <= player.move_allowed(): in_scoring_range.append(player)
    return in_scoring_range


def contains_a_player(game: g.Game, team: m.Team, squares: List[m.Square], include_own=True, include_opp=True, include_used=True, include_stunned=True, only_blockable=False) -> bool:

    allowed_players: List[m.Player] = get_players(game, team, include_own=include_own, include_opp=include_opp, include_used=include_used, include_stunned=include_stunned, only_blockable=only_blockable)

    for square in squares:
        player: Optional[None, m.Player] = game.state.pitch.get_player_at(square)
        if player is None:
            continue
        if player in allowed_players:
            return True
    return False

'''

def ArrayList<Unit> GetListOpponentsStandingToMoveAdjacent(Unit unit)

    ArrayList<Unit> units = GetListOpponentsAdjacent(unit)
    ArrayList<Unit> unitsRet = new ArrayList<Unit>()
    for curUnit in units:
        if (curUnit.GetState() == UnitState.Standing and !curUnit.GetHasMoved())
            unitsRet.add(curUnit)


    return unitsRet


def ArrayList<Unit> GetListOpponentsNotStunnedAdjacent(square: m.Square)
    List[m.Square] opponentSquaresAdjacent = GetListOpponentSquaresAdjacent(square)
    units: List[m.Player]Adjacent = new ArrayList<Unit>()
    for (BoardSquare curSquare : opponentSquaresAdjacent)
        if (curSquare.GetHasUnit() and curSquare.GetUnit().GetState() != UnitState.Stunned)
            unitsAdjacent.add(curSquare.GetUnit())

    return unitsAdjacent


def ArrayList<Unit> GetListProneOpponentsAdjacent(square: m.Square)
    List[m.Square] opponentSquaresAdjacent = GetListOpponentSquaresAdjacent(square)
    unitsAdjacent: List[m.Player] = new ArrayList<Unit>()
    for (BoardSquare curSquare : opponentSquaresAdjacent)
        if (curSquare.GetHasUnit() and curSquare.GetUnit().GetState() != UnitState.Standing)
            unitsAdjacent.add(curSquare.GetUnit())


    return unitsAdjacent


def ArrayList<Unit> GetListOpponentsBlockable(Unit unit)
    units: List[m.Player] = GetListOpponentsAdjacent(unit)
    for (Iterator<Unit> iter = units.iterator() iter.hasNext())
        Unit cur = iter.next()
        if (cur.GetState() != UnitState.Standing)
            iter.remove()


    return units


def ArrayList<Unit> GetListFriendlyWithTackleZoneAdjacent(square: m.Square)
    List[m.Square] friendlySquaresAdjacent = GetListFriendlySquaresAdjacent(square)
    unitsAdjacent: List[m.Player] = new ArrayList<Unit>()
    for (BoardSquare curSquare : friendlySquaresAdjacent)
        if (curSquare.GetHasUnit())
            Unit checkUnit = curSquare.GetUnit()
            if (checkUnit.GetHasTackleZone())
                unitsAdjacent.add(checkUnit)



    return unitsAdjacent




def ArrayList<Unit> GetListOpponentsWithTackleAndTackleZoneAdjacent(Unit unit)
    units: List[m.Player] = GetListOpponentsAdjacent(unit)
    for (Iterator<Unit> iter = units.iterator() iter.hasNext())
        Unit cur = iter.next()
        if (not (cur.GetHasTackleZone()) and cur.HasSkill(Skills.Tackle))
            iter.remove()


    return units


def ArrayList<Unit> GetListOpponentsWithTackleAndTackleZoneAdjacent(square: m.Square)
    units: List[m.Player] = GetListOpponentsAdjacent(square)
    for (Iterator<Unit> iter = units.iterator() iter.hasNext())
        Unit cur = iter.next()
        if (not (cur.GetHasTackleZone()) and cur.HasSkill(Skills.Tackle))
            iter.remove()


    return units


def List[m.Square] GetListEmptySquaresAdjacent(square: m.Square)
    List[m.Square] squares = game.state.pitch.get_adjacent_squares(square)
    for (Iterator<BoardSquare> iter = squares.iterator() iter.hasNext())
        BoardSquare cur = iter.next()
        if (cur.GetHasUnit())
            iter.remove()


    return squares



def ArrayList<Unit> GetListFriendlyPlayersToMove()
    ArrayList<Unit> allUnits = team.GetUnits()
    ArrayList<Unit> listPlayersToMove = new ArrayList<Unit>()
    for (Unit cur : allUnits)
        if (gameState.GetBoard().IsOnPitch(cur) and !cur.GetHasMoved())
            listPlayersToMove.add(cur)


    return listPlayersToMove


def ArrayList<Unit> GetListFriendlyPlayersMoved()
    ArrayList<Unit> allUnits = team.GetUnits()
    ArrayList<Unit> listPlayersToMove = new ArrayList<Unit>()
    for (Unit cur : allUnits)
        if (gameState.GetBoard().IsOnPitch(cur) and cur.GetHasMoved())
            listPlayersToMove.add(cur)


    return listPlayersToMove



def ArrayList<Unit> GetListFriendlyPlayersNotStunned()
    ArrayList<Unit> allUnits = team.GetUnits()
    ArrayList<Unit> listPlayersToMove = new ArrayList<Unit>()
    for (Unit cur : allUnits)
        if (gameState.GetBoard().IsOnPitch(cur) and cur.GetState() != UnitState.Stunned)
            listPlayersToMove.add(cur)


    return listPlayersToMove


def ArrayList<Unit> GetListStandingFriendlyPlayersToMove()
    ArrayList<Unit> allUnits = team.GetUnits()
    ArrayList<Unit> listPlayersToMove = new ArrayList<Unit>()
    for (Unit cur : allUnits)
        if (gameState.GetBoard().IsOnPitch(cur) and !cur.GetHasMoved() and cur.GetState() == UnitState.Standing)
            listPlayersToMove.add(cur)


    return listPlayersToMove


def ArrayList<Unit> GetListStandingFriendlyPlayers()
    ArrayList<Unit> allUnits = team.GetUnits()
    ArrayList<Unit> listPlayersToMove = new ArrayList<Unit>()
    for (Unit cur : allUnits)
        if (gameState.GetBoard().IsOnPitch(cur) and cur.GetState() == UnitState.Standing)
            listPlayersToMove.add(cur)


    return listPlayersToMove


def ArrayList<Unit> GetListFriendlyPlayersStanding()
    ArrayList<Unit> allUnits = team.GetUnits()
    ArrayList<Unit> listPlayersStanding = new ArrayList<Unit>()
    for (Unit cur : allUnits)
        if (gameState.GetBoard().IsOnPitch(cur) and cur.GetState() == UnitState.Standing)
            listPlayersStanding.add(cur)


    return listPlayersStanding


def ArrayList<Unit> GetListFriendlyPlayersWithHandsStanding()
    ArrayList<Unit> allUnits = team.GetUnits()
    ArrayList<Unit> listPlayersStanding = new ArrayList<Unit>()
    for (Unit cur : allUnits)
        if (gameState.GetBoard().IsOnPitch(cur) and cur.GetState() == UnitState.Standing and !cur.HasSkill(Skills.NoHands))
            listPlayersStanding.add(cur)


    return listPlayersStanding


 def ArrayList<Unit> GetListOpponentPlayersInReserves()
    ArrayList<Unit> allOppUnits = opponentTeam.GetUnits()
    ArrayList<Unit> opponentsInReserves = new ArrayList<Unit>()

    for (Unit cur : allOppUnits)
        if (cur.position instanceof ReserveSquare)
            opponentsInReserves.add(cur)


    return opponentsInReserves


def ArrayList<Unit> GetListNotStunnedOpponentPlayers()
    ArrayList<Unit> allOppUnits = opponentTeam.GetUnits()
    ArrayList<Unit> opponentsOnPitch = new ArrayList<Unit>()

    for (Unit cur : allOppUnits)
        if (gameState.GetBoard().IsOnPitch(cur) and cur.GetState() != UnitState.Stunned)
            opponentsOnPitch.add(cur)


    return opponentsOnPitch


def ArrayList<Unit> GetListOpponentPlayersStanding()
    ArrayList<Unit> allOppUnits = opponentTeam.GetUnits()
    ArrayList<Unit> opponentsOnPitch = new ArrayList<Unit>()

    for (Unit cur : allOppUnits)
        if (gameState.GetBoard().IsOnPitch(cur) and cur.GetState() == UnitState.Standing)
            opponentsOnPitch.add(cur)


    return opponentsOnPitch


def ArrayList<Unit> GetListOpponentPlayersNotStunned()
    ArrayList<Unit> allOppUnits = opponentTeam.GetUnits()
    ArrayList<Unit> opponentsOnPitch = new ArrayList<Unit>()

    for (Unit cur : allOppUnits)
        if (gameState.GetBoard().IsOnPitch(cur) and cur.GetState() != UnitState.Stunned)
            opponentsOnPitch.add(cur)


    return opponentsOnPitch


def ArrayList<Unit> GetListOpponentPlayersOnGround()
    ArrayList<Unit> allOppUnits = opponentTeam.GetUnits()
    ArrayList<Unit> opponentsOnPitch = new ArrayList<Unit>()

    for (Unit cur : allOppUnits)
        if (gameState.GetBoard().IsOnPitch(cur) and (cur.GetState() == UnitState.Prone or cur.GetState() == UnitState.Stunned))
            opponentsOnPitch.add(cur)


    return opponentsOnPitch
'''