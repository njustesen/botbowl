"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains all the procedures. Procedures contain the core part of the rules in FFAI. Each procedure are
responsible for an isolated part of the game. Procedures are added to a stack, and the top-most procedure must finish
before other procedures are run. Procedures can add other procedures to the stack simply by instantiating procedures.
"""
from abc import abstractmethod, ABCMeta

from ffai.core.model import *
from ffai.core.table import *
import time


class Procedure:

    def __init__(self, game):
        self.game = game
        self.game.state.stack.push(self)
        self.done = False
        self.initialized = False

    def setup(self):
        """
        Is called just before step() is called the first time. The Game clas will set initialized to True.
        """
        pass

    def step(self, action):
        """
        Is called when this procedure is on the top of the stack until the procedure is done.
        :param action: an action performed by an agent - will be None if there are no available actions.
        :return: a bool indicating whether the procedure is done.
        """
        pass

    def available_actions(self):
        """
        :return: a list of available actions. This can vary based on the state of the procedure.
        """
        pass


class Apothecary(Procedure):

    def __init__(self, game, player, roll, outcome, inflictor, casualty=None, effect=None):
        super().__init__(game)
        self.game = game
        self.player = player
        self.inflictor = inflictor
        self.waiting_apothecary = False
        self.roll_first = roll
        self.roll_second = roll
        self.outcome = outcome
        self.casualty_first = casualty
        self.effect_first = effect
        self.casualty_second = None
        self.effect_second = None

    def setup(self):
        if self.game.state.current_team != self.player.team:
            self.game.add_opp_clock(self.player.team)

    def step(self, action):

        if self.outcome == OutcomeType.KNOCKED_OUT:

            if action.action_type == ActionType.USE_APOTHECARY:

                # Player is moved to reserves
                self.player.team.state.apothecary_available = False
                self.player.state.stunned = True
                self.player.state.up = False
                self.game.report(Outcome(OutcomeType.APOTHECARY_USED_KO, player=self.player, team=self.player.team))

            else:

                # Player is KO
                self.game.pitch_to_kod(self.player)
                self.game.report(Outcome(OutcomeType.APOTHECARY_USED_KO, player=self.player, team=self.player.team))

            return True

        elif self.outcome == OutcomeType.CASUALTY:

            if action.action_type == ActionType.USE_APOTHECARY:

                self.roll_second = DiceRoll([D6(self.game.rnd), D8(self.game.rnd)], roll_type=RollType.CASUALTY_ROLL)
                result = self.roll_second.get_sum()
                n = min(61, max(38, result))
                self.casualty_second = CasualtyType(n)
                self.effect_second = Rules.casualty_effect[self.casualty_second]
                self.player.team.state.apothecary_available = False
                self.game.report(Outcome(OutcomeType.CASUALTY_APOTHECARY, player=self.player, team=self.player.team,
                                         rolls=[self.roll_first, self.roll_second]))
                self.waiting_apothecary = True

                return False

            if action.action_type == ActionType.SELECT_FIRST_ROLL or ActionType.SELECT_SECOND_ROLL:

                effect = self.effect_first if action.action_type == ActionType.SELECT_FIRST_ROLL else self.effect_second
                casualty = self.casualty_first if action.action_type == ActionType.SELECT_FIRST_ROLL else self.casualty_second
                roll = self.roll_first if action.action_type == ActionType.SELECT_FIRST_ROLL else self.roll_second

                # Apply casualty
                self.game.pitch_to_casualties(self.player, casualty, effect, apothecary=True)
                if effect == CasualtyEffect.NONE:
                    self.game.report(Outcome(OutcomeType.BADLY_HURT, player=self.player, team=self.player.team, rolls=[roll]))
                elif effect in Casualty.miss_next_game:
                    self.game.report(Outcome(OutcomeType.MISS_NEXT_GAME, player=self.player, team=self.player.team,
                                             rolls=[roll]))
                elif effect == CasualtyEffect.DEAD:
                    self.game.report(Outcome(OutcomeType.DEAD, player=self.player, team=self.player.team, rolls=[roll]))

        return True

    def available_actions(self):
        if self.outcome == OutcomeType.KNOCKED_OUT or self.waiting_apothecary:
            return [ActionChoice(ActionType.USE_APOTHECARY, team=self.player.team),
                    ActionChoice(ActionType.DONT_USE_APOTHECARY, team=self.player.team)]
        else:
            # TODO: Does not really work
            return [ActionChoice(ActionType.SELECT_FIRST_ROLL, team=self.player.team),
                    ActionChoice(ActionType.SELECT_SECOND_ROLL, team=self.player.team)]


class Armor(Procedure):

    def __init__(self, game, player, modifiers=0, inflictor=None, foul=False):
        super().__init__(game)
        self.player = player
        self.modifiers = modifiers
        self.inflictor = inflictor
        self.skip_armor = False
        self.armor_rolled = False
        self.foul = foul
        self.inflictor = inflictor

    def step(self, action):

        # Roll
        roll = DiceRoll([D6(self.game.rnd), D6(self.game.rnd)], roll_type=RollType.ARMOR_ROLL)
        roll.modifiers = self.modifiers
        roll.target = self.player.get_av() + 1
        result = roll.get_sum() + self.modifiers
        self.armor_rolled = True

        armor_broken = False
        mighty_blow_used = False
        dirty_player_used = False

        if not self.foul:
            # Armor broken - Claws
            if roll.sum >= 8 and self.inflictor is not None and self.inflictor.has_skill(Skill.CLAWS):
                armor_broken = True

            # Armor broken
            if result >= roll.target:
                armor_broken = True
            elif result == roll.target -1 and self.inflictor is not None and self.inflictor.has_skill(Skill.MIGHTY_BLOW):
                # only use mighty_blow if it makes the armour break
                roll.modifiers += 1
                armor_broken = True
                mighty_blow_used = True
        else:

            # Armor broken - Dirty player
            if self.inflictor is not None and self.inflictor.has_skill(Skill.DIRTY_PLAYER) \
                    and result + 1 > self.player.get_av():
                roll.modifiers += 1
                armor_broken = True
                dirty_player_used = True

            # Armor broken
            if result >= roll.target:
                armor_broken = True

        # EJECTION?
        ejected = False
        if self.foul:
            if roll.same():
                if not self.inflictor.has_skill(Skill.SNEAKY_GIT) or armor_broken:
                    self.game.report(Outcome(OutcomeType.PLAYER_EJECTED, player=self.inflictor))
                    Turnover(self.game)
                    Ejection(self.game, self.inflictor)
                    ejected = True

        # Break armor - roll injury
        if armor_broken:
            Injury(self.game, self.player, self.inflictor, foul=self.foul,
                   mighty_blow_used=mighty_blow_used, dirty_player_used=dirty_player_used, ejected=ejected)
            self.game.report(Outcome(OutcomeType.ARMOR_BROKEN, player=self.player, opp_player=self.inflictor,
                                     rolls=[roll]))
        else:
            self.game.report(Outcome(OutcomeType.ARMOR_NOT_BROKEN, player=self.player, opp_player=self.inflictor,
                                     rolls=[roll]))

        return True

    def available_actions(self):
        return []


class Block(Procedure):

    def __init__(self, game, attacker, defender, blitz=False, frenzy_block=False, gfi=False):
        super().__init__(game)
        self.attacker = attacker
        self.defender = defender
        self.reroll_decision_made = False
        self.roll = None
        self.blitz = blitz
        self.gfi = gfi
        self.frenzy_block = frenzy_block
        self.waiting_wrestle_attacker = False
        self.waiting_wrestle_defender = False
        self.selected_die = None
        self.wrestle = False
        self.favor = None
        self.dauntless_roll = None
        self.dauntless_success = False
        self.frenzy_checked = False
        self.waiting_gfi = False

    def step(self, action):

        # GfI
        if self.gfi:
            self.gfi = False
            GFI(self.game, self.attacker, self.attacker.position)
            return False

        # Frenzy check
        if self.frenzy_block and not self.frenzy_checked:
            # End frenzy block procedure if plays is not on the field or not standing
            if self.defender.position is None or not self.defender.state.up:
                return True
            # If not adjacent to defender - e.g. in case of fend
            if self.defender not in self.game.get_adjacent_opponents(self.attacker):
                return True
            self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.attacker, skill=Skill.FRENZY))
            self.attacker.state.used_skills.append(Skill.FRENZY)
            self.frenzy_checked = True

        # Roll
        if self.roll is None:

            # Assists
            if self.defender.get_st() > self.attacker.get_st() and self.attacker.has_skill(Skill.DAUNTLESS) \
                    and self.dauntless_roll is None:
                self.dauntless_roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.STRENGTH_ROLL)
                self.dauntless_success = self.dauntless_roll.get_sum() + self.attacker.get_st() > self.defender.get_st()
                self.game.report(Outcome(OutcomeType.DAUNTLESS_USED, team=self.attacker.team, player=self.attacker,
                                         rolls=[self.dauntless_roll], n=True))
                return False

            # Report Horns
            if self.blitz and self.attacker.has_skill(Skill.HORNS):
                self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.attacker, skill=Skill.HORNS))

            dice = self.game.num_block_dice(self.attacker, self.defender, blitz=self.blitz,
                                            dauntless_success=self.dauntless_success)
            self.favor = self.attacker.team if dice > 0 else self.defender.team

            # Roll
            self.roll = DiceRoll([], roll_type=RollType.BLOCK_ROLL)
            for i in range(abs(dice)):
                self.roll.dice.append(BBDie(self.game.rnd))

            self.game.report(Outcome(OutcomeType.BLOCK_ROLL, player=self.attacker, opp_player=self.defender,
                                     rolls=[self.roll]))

            # Add secondary clock if opponent has the favor
            if not self.game.can_use_reroll(self.attacker.team):
                if self.favor == self.defender.team:
                    self.game.add_secondary_clock(self.favor)

            return False

        elif self.waiting_wrestle_defender:

            #self.wrestle = action.action_type == ActionType.USE_WRESTLE
            self.waiting_wrestle_defender = False
            self.selected_die = BBDieResult.BOTH_DOWN

        elif action.action_type == ActionType.USE_REROLL:

            # Roll again
            self.reroll_decision_made = True
            self.attacker.team.state.use_reroll()
            self.game.report(Outcome(OutcomeType.REROLL_USED, team=self.attacker.team))
            self.roll = None
            return self.step(None)

        elif action.action_type == ActionType.DONT_USE_REROLL:
            # Roll again
            self.reroll_decision_made = True

            # Add secondary clock if opponent has the favor
            if self.favor != self.game.state.current_team:
                if self.game.state.current_team != self.favor:
                    self.game.add_secondary_clock(self.favor)

        #elif action.action_type == ActionType.USE_JUGGERNAUT:
        #    self.selected_die = BBDieResult.PUSH

        # Select die
        elif action.action_type == ActionType.SELECT_ATTACKER_DOWN:
            self.selected_die = BBDieResult.ATTACKER_DOWN
        elif action.action_type == ActionType.SELECT_BOTH_DOWN:
            self.selected_die = BBDieResult.BOTH_DOWN
        elif action.action_type == ActionType.SELECT_PUSH:
            self.selected_die = BBDieResult.PUSH
        elif action.action_type == ActionType.SELECT_DEFENDER_STUMBLES:
            self.selected_die = BBDieResult.DEFENDER_STUMBLES
        elif action.action_type == ActionType.SELECT_DEFENDER_DOWN:
            self.selected_die = BBDieResult.DEFENDER_DOWN

        # Remove secondary clocks
        if self.selected_die is not None:
            self.game.remove_secondary_clocks()

        # Dice result
        if self.selected_die == BBDieResult.ATTACKER_DOWN:
            Turnover(self.game)
            KnockDown(self.game, self.attacker, inflictor=self.defender)
            return True

        elif self.selected_die == BBDieResult.BOTH_DOWN:
            if not self.attacker.has_skill(Skill.BLOCK):
                Turnover(self.game)
                KnockDown(self.game, self.attacker, inflictor=self.defender)
            else:
                self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.attacker, skill=Skill.BLOCK))
            if not self.defender.has_skill(Skill.BLOCK):
                KnockDown(self.game, self.defender, inflictor=self.attacker)
            else:
                self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.defender, skill=Skill.BLOCK))
            return True

        elif self.selected_die == BBDieResult.DEFENDER_DOWN:
            Push(self.game, self.attacker, self.defender, knock_down=True, blitz=self.blitz)
            return True

        elif self.selected_die == BBDieResult.DEFENDER_STUMBLES:
            Push(self.game, self.attacker, self.defender, knock_down=(not self.defender.has_skill(Skill.DODGE)) or self.attacker.has_skill(Skill.TACKLE),
                 blitz=self.blitz)
            return True

        elif self.selected_die == BBDieResult.PUSH:
            Push(self.game, self.attacker, self.defender, knock_down=False, blitz=self.blitz)
            return True

        return False

    def available_actions(self):

        actions = []

        if self.roll is not None and self.selected_die is None:
            disable_dice_pick = False
            if self.game.can_use_reroll(self.attacker.team) and not self.reroll_decision_made:
                actions.append(ActionChoice(ActionType.USE_REROLL, self.attacker.team))
                actions.append(ActionChoice(ActionType.DONT_USE_REROLL, self.attacker.team))
                if self.favor != self.attacker.team:
                    disable_dice_pick = True

            for die in self.roll.dice:
                if die.get_value() == BBDieResult.ATTACKER_DOWN:
                    actions.append(ActionChoice(ActionType.SELECT_ATTACKER_DOWN, self.favor, disabled=disable_dice_pick))
                if die.get_value() == BBDieResult.BOTH_DOWN:
                    actions.append(ActionChoice(ActionType.SELECT_BOTH_DOWN, self.favor, disabled=disable_dice_pick))
                if die.get_value() == BBDieResult.PUSH:
                    actions.append(ActionChoice(ActionType.SELECT_PUSH, self.favor, disabled=disable_dice_pick))
                if die.get_value() == BBDieResult.DEFENDER_STUMBLES:
                    actions.append(ActionChoice(ActionType.SELECT_DEFENDER_STUMBLES, self.favor, disabled=disable_dice_pick))
                if die.get_value() == BBDieResult.DEFENDER_DOWN:
                    actions.append(ActionChoice(ActionType.SELECT_DEFENDER_DOWN, self.favor, disabled=disable_dice_pick))

        return actions


class Bounce(Procedure):

    def __init__(self, game, ball, kick=False):
        super().__init__(game)
        self.ball = ball
        self.kick = kick

    def step(self, action):

        # Loose control
        self.ball.is_carried = False

        # Roll
        roll_scatter = DiceRoll([D8(self.game.rnd)], roll_type=RollType.BOUNCE_ROLL)
        result = roll_scatter.get_sum()

        # Bounce
        x = 0
        y = 0
        if result in [1, 4, 6]:
            x = -1
        if result in [3, 5, 8]:
            x = 1
        if result in [1, 2, 3]:
            y = -1
        if result in [6, 7, 8]:
            y = 1

        self.ball.move(x, y)
        self.game.report(Outcome(OutcomeType.BALL_BOUNCED, position=self.ball.position, rolls=[roll_scatter]))

        if self.kick:
            # Touchback
            if not self.game.is_team_side(self.ball.position, self.game.get_receiving_team()):
                Touchback(self.game, self.ball)
                self.game.report(Outcome(OutcomeType.TOUCHBACK, team=self.game.get_receiving_team(),
                                         rolls=[roll_scatter]))
                return True
        else:
            # Out of bounds
            if self.game.is_out_of_bounds(self.ball.position):
                ThrowIn(self.game, self.ball, Square(self.ball.position.x - x, self.ball.position.y - y))
                self.game.report(Outcome(OutcomeType.BALL_OUT_OF_BOUNDS))
                return True

        # On player -> Catch
        player_at = self.game.get_player_at(self.ball.position)
        if player_at is not None:
            Catch(self.game, player_at, self.ball)
            self.game.report(Outcome(OutcomeType.BALL_HIT_PLAYER, position=self.ball.position, player=player_at))
            return True

        self.game.report(Outcome(OutcomeType.BALL_ON_GROUND, position=self.ball.position))
        return True

    def available_actions(self):
        return []


class Casualty(Procedure):

    miss_next_game = [CasualtyEffect.MNG, CasualtyEffect.AG, CasualtyEffect.AV, CasualtyEffect.MA, CasualtyEffect.ST,
                      CasualtyEffect.NI]

    def __init__(self, game, player, roll, inflictor=None):
        super().__init__(game)
        self.game = game
        self.player = player
        self.inflictor = inflictor
        self.waiting_apothecary = False
        self.roll = roll
        self.casualty = None
        self.effect = None

    def step(self, action):

        self.roll = DiceRoll([D6(self.game.rnd), D8(self.game.rnd)], d68=True, roll_type=RollType.CASUALTY_ROLL)
        result = self.roll.get_sum()
        n = min(61, max(38, result))
        self.casualty = CasualtyType(n)
        self.effect = Rules.casualty_effect[self.casualty]

        if self.player.team.state.apothecary_available:
            self.game.report(
                Outcome(OutcomeType.CASUALTY, player=self.player, opp_player=self.inflictor, team=self.player.team, n=self.effect.name,
                        rolls=[self.roll]))
            Apothecary(self.game, self.player, roll=self.roll, outcome=OutcomeType.CASUALTY,
                       casualty=self.casualty, effect=self.effect, inflictor=self.inflictor)
        else:
            # Apply casualty
            self.game.pitch_to_casualties(self.player, self.casualty, self.effect)
            if self.effect == CasualtyEffect.NONE:
                self.game.report(Outcome(OutcomeType.BADLY_HURT, player=self.player, opp_player=self.inflictor, team=self.player.team,
                                         rolls=[self.roll]))
            elif self.effect in Casualty.miss_next_game:
                self.game.report(Outcome(OutcomeType.MISS_NEXT_GAME, player=self.player, opp_player=self.inflictor, team=self.player.team,
                                         rolls=[self.roll], n=self.effect.name))
            elif self.effect == CasualtyEffect.DEAD:
                self.game.report(Outcome(OutcomeType.DEAD, player=self.player, opp_player=self.inflictor, team=self.player.team,
                                         rolls=[self.roll]))

        return True

    def available_actions(self):
        return []


class Catch(Procedure):

    def __init__(self, game, player, ball, accurate=False, interception=False, handoff=False, kick=False):
        super().__init__(game)
        self.player = player
        self.ball = ball
        self.accurate = accurate
        self.handoff = handoff
        self.rolled = False
        self.reroll_used = False
        self.catch_used = False
        self.kick = kick
        self.waiting_for_reroll = False
        self.waiting_for_catch = False
        self.interception = interception

    def step(self, action):

        # Otherwise roll if player hasn't rolled
        if not self.rolled:

            # Can player even catch ball?
            if self.player.has_skill(Skill.NO_HANDS) or not self.player.can_catch():
                Bounce(self.game, self.ball)
                self.game.report(Outcome(OutcomeType.BALL_DROPPED, player=self.player))
                return True

            # Set modifiers
            modifiers = self.game.get_catch_modifiers(self.player, accurate=self.accurate,
                                              interception=self.interception, handoff=self.handoff)

            # Roll
            roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.AGILITY_ROLL)
            roll.modifiers = modifiers
            roll.target = Rules.agility_table[self.player.get_ag()]
            self.rolled = True
            if roll.is_d6_success():
                if self.interception:
                    self.game.report(Outcome(OutcomeType.INTERCEPTION, player=self.player, rolls=[roll]))
                    self.ball.move_to(self.player.position)
                    self.ball.is_carried = True
                    if self.game.is_touchdown(self.player):
                        Touchdown(self.game, self.player)
                    else:
                        Turnover(self.game)
                else:
                    self.game.report(Outcome(OutcomeType.CATCH, player=self.player, rolls=[roll]))
                    self.ball.is_carried = True
                    if self.game.is_touchdown(self.player):
                        Touchdown(self.game, self.player)
                return True
            else:

                if self.interception:
                    self.game.report(Outcome(OutcomeType.INTERCEPTION_FAILED, player=self.player, rolls=[roll]))
                    return True

                self.game.report(Outcome(OutcomeType.CATCH_FAILED, player=self.player, rolls=[roll]))

                # Check if catch
                if self.player.has_skill(Skill.CATCH) and not self.catch_used:
                    self.catch_used = True
                    self.waiting_for_catch = True
                    self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.player, skill=Skill.CATCH))
                    return False

                # Check if reroll available
                if self.game.can_use_reroll(self.player.team) and not self.catch_used:
                    self.waiting_for_reroll = True
                    if self.player.team != self.game.state.current_team:
                        self.game.add_secondary_clock(self.player.team)
                    return False

                Bounce(self.game, self.ball, kick=self.kick)
                return True

        # If catch used
        if self.waiting_for_catch:
            self.catch_used = True
            self.rolled = False
            return self.step(None)

        # If re-roll used
        if self.waiting_for_reroll:
            if self.player.team != self.game.state.current_team:
                self.game.remove_secondary_clocks()
            if action.action_type == ActionType.USE_REROLL:
                self.reroll_used = True
                self.player.team.state.use_reroll()
                self.game.report(Outcome(OutcomeType.REROLL_USED, team=self.player.team))
                self.rolled = False
                return self.step(None)
            else:
                Bounce(self.game, self.ball, kick=self.kick)

        return True

    def available_actions(self):
        if self.waiting_for_reroll:
            return [ActionChoice(ActionType.USE_REROLL, team=self.player.team),
                    ActionChoice(ActionType.DONT_USE_REROLL, team=self.player.team)]
        return []


class CoinTossFlip(Procedure):

    def __init__(self, game):
        super().__init__(game)

    def setup(self):
        self.game.add_secondary_clock(self.game.state.away_team)

    def step(self, action):
        if action.action_type == ActionType.HEADS:
            if self.game.rnd.rand(1)[0] >= 0.5:
                self.game.state.coin_toss_winner = self.game.state.away_team
                self.game.report(Outcome(OutcomeType.HEADS_WON))
            else:
                self.game.state.coin_toss_winner = self.game.state.home_team
                self.game.report(Outcome(OutcomeType.TAILS_LOSS))
        elif action.action_type == ActionType.TAILS:
            if self.game.rnd.rand(1)[0] >= 0.5:
                self.game.state.coin_toss_winner = self.game.state.away_team
                self.game.report(Outcome(OutcomeType.TAILS_WON))
            else:
                self.game.state.coin_toss_winner = self.game.state.home_team
                self.game.report(Outcome(OutcomeType.HEADS_LOSS))

        CoinTossKickReceive(self.game)
        self.game.remove_clocks()
        return True

    def available_actions(self):
        return [ActionChoice(ActionType.HEADS, team=self.game.state.away_team),
                   ActionChoice(ActionType.TAILS, team=self.game.state.away_team)]


class CoinTossKickReceive(Procedure):

    def __init__(self, game):
        super().__init__(game)
        self.aa = [ActionChoice(ActionType.KICK,
                                team=self.game.state.coin_toss_winner),
                   ActionChoice(ActionType.RECEIVE,
                                team=self.game.state.coin_toss_winner)]

    def setup(self):
        self.game.add_secondary_clock(self.game.state.coin_toss_winner)

    def step(self, action):
        kicking = None
        receiving = None
        if action.action_type == ActionType.KICK:
            kicking = self.game.state.coin_toss_winner
            receiving = self.game.get_opp_team(self.game.state.coin_toss_winner)
        elif action.action_type == ActionType.RECEIVE:
            kicking = self.game.get_opp_team(self.game.state.coin_toss_winner)
            receiving = self.game.state.coin_toss_winner
        self.game.state.kicking_first_half = kicking
        self.game.state.kicking_this_drive = kicking
        self.game.state.receiving_first_half = receiving
        self.game.state.receiving_this_drive = receiving
        if receiving == self.game.state.home_team:
            self.game.report(Outcome(OutcomeType.HOME_RECEIVE, team=receiving))
        else:
            self.game.report(Outcome(OutcomeType.AWAY_RECEIVE, team=receiving))
        self.game.remove_clocks()
        return True

    def available_actions(self):
        return self.aa


class Ejection(Procedure):

    def __init__(self, game, player):
        super().__init__(game)
        self.player = player

    def step(self, action):
        self.game.pitch_to_dungeon(self.player)
        return True

    def available_actions(self):
        return []


class Foul(Procedure):

    def __init__(self, game, fouler, defender):
        super().__init__(game)
        self.fouler = fouler
        self.defender = defender

    def step(self, action):

        # Assists
        assists_from = self.game.get_assisting_players(self.fouler, self.defender, foul=True)
        assists_to = self.game.get_assisting_players(self.defender, self.fouler, foul=True)
        modifier = len(assists_from) - len(assists_to)

        self.game.report(Outcome(OutcomeType.FOUL, player=self.fouler, opp_player=self.defender))

        # Armor roll
        Armor(self.game, self.defender, modifiers=modifier, inflictor=self.fouler, foul=True)

        return True

    def available_actions(self):
        return []


class ResetHalf(Procedure):

    def __init__(self, game):
        super().__init__(game)

    def step(self, action):
        for team in self.game.state.teams:
            team.state.rerolls = team.rerolls
            team.state.turn = 0
        return True

    def available_actions(self):
        return []


class Half(Procedure):

    def __init__(self, game, half):
        super().__init__(game)
        self.half = half
        self.kicked_off = False
        self.prepared = False

    def step(self, action):

        # If second half
        if not self.prepared and self.half > 1:
            PreKickOff(self.game, self.game.get_kicking_team(self.half))
            PreKickOff(self.game, self.game.get_receiving_team(self.half))
            self.prepared = True
            return False

        # Kickoff
        if not self.kicked_off:
            self.game.state.half = self.half
            self.game.state.round = 0
            self.game.state.kicking_this_drive = self.game.get_kicking_team(self.half)
            self.game.state.receiving_this_drive = self.game.get_receiving_team(self.half)
            self.kicked_off = True
            self.game.set_turn_order_from(self.game.get_receiving_team(self.half))
            Kickoff(self.game)
            Setup(self.game, team=self.game.get_receiving_team(self.half))
            Setup(self.game, team=self.game.get_kicking_team(self.half))
            ResetHalf(self.game)
            ClearBoard(self.game)
            return False

        # Add turn in round
        if self.game.state.round < self.game.config.rounds:
            self.game.state.round += 1
            for team in reversed(self.game.state.turn_order):
                Turn(self.game, team, self.half, self.game.state.round)
            return False

        if self.half == 1:
            self.game.report(Outcome(OutcomeType.END_OF_FIRST_HALF))
        elif self.half == 2:
            self.game.report(Outcome(OutcomeType.END_OF_SECOND_HALF))

        return True

    def available_actions(self):
        return []


class Injury(Procedure):

    def __init__(self, game, player, inflictor=None, foul=False, mighty_blow_used=False, dirty_player_used=False,
                 ejected=False, in_crowd=False):
        super().__init__(game)
        self.game = game
        self.player = player
        self.inflictor = inflictor
        self.injury_rolled = False
        self.foul = foul
        self.mighty_blow_used = mighty_blow_used
        self.dirty_player_used = dirty_player_used
        self.ejected = ejected
        self.apothecary_used = False
        self.in_crowd = in_crowd

    def step(self, action):

        # TODO: Necromancer

        # Roll
        roll = DiceRoll([D6(self.game.rnd), D6(self.game.rnd)], roll_type=RollType.INJURY_ROLL)
        result = roll.get_sum()
        self.injury_rolled = True

        # Skill modifiers
        thick_skull = -1 if self.player.has_skill(Skill.THICK_SKULL) else 0
        stunty = 1 if self.player.has_skill(Skill.STUNTY) else 0
        mighty_blow = 0
        dirty_player = 0
        if self.inflictor is not None:
            dirty_player = 1 if self.inflictor.has_skill(Skill.DIRTY_PLAYER) and not self.dirty_player_used and \
                                self.foul else 0
            mighty_blow = 1 if self.inflictor.has_skill(Skill.MIGHTY_BLOW) and not self.mighty_blow_used and not \
                self.foul else 0

        # EJECTION
        if self.foul and not self.ejected:
            if roll.same():
                if not self.inflictor.has_skill(Skill.SNEAKY_GIT):
                    self.game.report(Outcome(OutcomeType.PLAYER_EJECTED, player=self.inflictor))
                    Turnover(self.game)
                    Ejection(self.game, self.inflictor)

        # STUNNED
        if result + thick_skull + stunty + mighty_blow + dirty_player <= 7:
            roll.modifiers = thick_skull + stunty + mighty_blow + dirty_player
            if self.player.has_skill(Skill.BALL_AND_CHAIN):
                KnockOut(self.game, self.player, roll=roll, inflictor=self.inflictor)
            else:
                self.game.report(Outcome(OutcomeType.STUNNED, player=self.player, opp_player=self.inflictor,
                                         rolls=[roll]))
                if self.in_crowd:
                    self.game.pitch_to_reserves(self.player)
                else:
                    self.player.state.stunned = True
                    self.player.state.up = False

        # CASUALTY
        elif result + stunty + mighty_blow + dirty_player >= 10:
            roll.modifiers = stunty + mighty_blow + dirty_player
            self.game.report(Outcome(OutcomeType.CASUALTY, player=self.player, opp_player=self.inflictor, rolls=[roll]))
            Casualty(self.game, self.player, roll, inflictor=self.inflictor)

        # KOD
        else:
            roll.modifiers = thick_skull + stunty + mighty_blow + dirty_player
            KnockOut(self.game, self.player, roll=roll, inflictor=self.inflictor)

        return True

    def available_actions(self):
        return []


class Interception(Procedure):

    def __init__(self, game, team, ball, interceptors):
        super().__init__(game)
        self.team = team
        self.ball = ball
        self.interceptors = interceptors

    def setup(self):
        self.game.add_secondary_clock(self.team)

    def step(self, action):

        if action.action_type == ActionType.INTERCEPTION:
            self.ball.move_to(action.player.position)
            Catch(self.game, action.player, self.ball, interception=True)

        self.game.remove_secondary_clocks()
        return True

    def available_actions(self):
        agi_rolls = []
        for interceptor in self.interceptors:
            modifier = self.game.get_catch_modifiers(interceptor, interception=True)
            agi_rolls.append([max(2, min(6, Rules.agility_table[interceptor.get_ag()] - modifier))])
        return [ActionChoice(ActionType.INTERCEPTION, team=self.team,
                             players=self.interceptors, agi_rolls=agi_rolls),
                ActionChoice(ActionType.SELECT_NONE, team=self.team)]


class Touchback(Procedure):

    def __init__(self, game, ball):
        super().__init__(game)
        self.ball = ball

    def setup(self):
        self.game.add_secondary_clock(self.game.state.receiving_this_drive)
        self.players_on_pitch_standing = self.game.get_players_on_pitch(self.game.state.receiving_this_drive, up=True)

    def step(self, action):

        player = None
        if len(self.players_on_pitch_standing) == 0:
            position = self.game.rnd.choice(self.game.get_team_side(self.game.state.receiving_this_drive))
            self.ball.is_carried = False
        else:
            position = action.player.position
            self.ball.is_carried = True
            player = action.player

        self.ball.move_to(position)
        self.ball.on_ground = True
        self.game.report(Outcome(OutcomeType.TOUCHBACK_BALL_PLACED, player=player, position=position))
        self.game.remove_secondary_clocks()
        return True

    def available_actions(self):
        # Only allow standing players unless no players.
        if len(self.players_on_pitch_standing) > 0:
            return [ActionChoice(ActionType.SELECT_PLAYER, team=self.game.state.receiving_this_drive,
                                 players=self.players_on_pitch_standing)]
        else:
            return []


class LandKick(Procedure):

    def __init__(self, game, ball):
        super().__init__(game)
        self.ball = ball
        self.landed = False

    def step(self, action):

        if not self.game.is_team_side(self.ball.position, self.game.get_receiving_team()):
            Touchback(self.game, self.ball)
            self.game.report(Outcome(OutcomeType.TOUCHBACK, team=self.game.get_receiving_team()))
            return True

        # Gentle gust
        if self.game.state.gentle_gust:
            Scatter(self.game, self.ball, kick=True, gentle_gust=True)
            self.game.state.gentle_gust = False
            return False

        self.ball.on_ground = True
        player_at = self.game.get_player_at(self.ball.position)
        if player_at is None:
            Bounce(self.game, self.ball, kick=True)
            self.game.report(Outcome(OutcomeType.BALL_HIT_GROUND, position=self.ball.position))
            return True

        Catch(self.game, player_at, self.ball, kick=True)
        self.game.report(Outcome(OutcomeType.BALL_HIT_PLAYER, position=self.ball.position, player=player_at))

        return True

    def available_actions(self):
        return []


class Fans(Procedure):

    def __init__(self, game):
        super().__init__(game)

    def step(self, action):

        # Fans
        rolls = []
        spectators = []
        for team in self.game.state.teams:
            roll = DiceRoll([D6(self.game.rnd), D6(self.game.rnd)], roll_type=RollType.FANS_ROLL)
            rolls.append(roll)
            fans = (roll.get_sum() + team.fan_factor) * 1000
            spectators.append(fans)
            self.game.report(Outcome(OutcomeType.TEAM_SPECTATORS, n=fans, team=team, rolls=[roll]))
        self.game.state.spectators = int(np.sum(spectators))
        self.game.report(Outcome(OutcomeType.SPECTATORS, n=self.game.state.spectators))

        # FAME
        max_fans = int(np.max(spectators))
        min_fans = int(np.min(spectators))
        for i in range(len(self.game.state.teams)):
            team = self.game.state.teams[i]
            team_fans = spectators[i]
            fame = 0
            if team_fans == max_fans:
                if team_fans >= min_fans * 2:
                    fame = 2
                elif team_fans > min_fans:
                    fame = 1
            self.game.report(Outcome(OutcomeType.FAME, n=fame, team=team))
            team.state.fame = fame

        return True

    def available_actions(self):
        return []


class Kickoff(Procedure):

    def __init__(self, game):
        super().__init__(game)

    def step(self, action):
        self.game.state.gentle_gust = False
        # self.game.reset_kickoff()
        ball = Ball(None, on_ground=False)
        LandKick(self.game, ball)
        KickoffTable(self.game, ball)
        Scatter(self.game, ball, kick=True)
        PlaceBall(self.game, ball)
        return True

    def available_actions(self):
        return []


class GetTheRef(Procedure):

    def __init__(self, game):
        super().__init__(game)

    def step(self, action):
        for team in self.game.state.teams:
            team.state.bribes += 1
            self.game.report(Outcome(OutcomeType.EXTRA_BRIBE, team=team))
        return True

    def available_actions(self):
        return []


class Riot(Procedure):

    def __init__(self, game):
        super().__init__(game)
        self.effect = 0

    def step(self, action):
        roll = None
        receiving_turn = self.game.get_receiving_team().state.turn
        if receiving_turn == 7:
            self.effect = -1
        elif receiving_turn == 0:
            self.effect = 1
        else:
            roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.RIOT_ROLL)
            if roll.get_sum() <= 3:
                self.effect = 1
            else:
                self.effect = -1

        self.game.add_or_skip_turn(self.effect)
        if self.effect == -1:
            self.game.report(Outcome(OutcomeType.TURN_ADDED, rolls=[] if roll is None else [roll]))
        if self.effect == 1:
            self.game.report(Outcome(OutcomeType.TURN_SKIPPED, rolls=[] if roll is None else [roll]))

        return True

    def available_actions(self):
        return []


class HighKick(Procedure):

    def __init__(self, game, ball):
        super().__init__(game)
        self.ball = ball
        self.receiving_team = self.game.get_receiving_team()

    def setup(self):
        self.game.add_secondary_clock(self.receiving_team)
        self.available_players = []
        for player in self.game.get_players_on_pitch(self.receiving_team, up=True):
            if self.game.num_tackle_zones_in(player) == 0:
                self.available_players.append(player)

    def step(self, action):
        self.game.remove_secondary_clocks()
        if not self.game.is_team_side(self.ball.position, self.receiving_team) or \
            self.game.get_player_at(self.ball.position) is not None:
            return True
        if len(self.available_players) == 0:
            return True
        if action.action_type == ActionType.SELECT_PLAYER:
            self.game.move(action.player, self.ball.position)
            self.game.report(Outcome(OutcomeType.PLAYER_PLACED_HIGH_KICK, position=self.game.get_ball_position(), team=self.receiving_team))
        elif action.action_type == ActionType.END_SETUP:
            self.game.report(Outcome(OutcomeType.SETUP_DONE, team=self.receiving_team))
        return True

    def available_actions(self):
        if len(self.available_players) == 0:
            return []
        if self.game.is_team_side(self.ball.position, self.receiving_team) and \
                self.game.get_player_at(self.ball.position) is None:
            return [ActionChoice(ActionType.SELECT_PLAYER, team=self.receiving_team,
                                 players=self.available_players),
                    ActionChoice(ActionType.SELECT_NONE, team=self.receiving_team)]
        return []


class CheeringFans(Procedure):

    def __init__(self, game):
        super().__init__(game)

    def step(self, action):
        rolls = []
        cheers = []
        for team in self.game.state.teams:
            roll = DiceRoll([D3(self.game.rnd)], roll_type=RollType.CHEERING_FANS_ROLL)
            rolls.append(roll)
            roll.modifiers = team.state.fame + team.cheerleaders
            cheers.append(roll.get_result())
            self.game.report(Outcome(OutcomeType.CHEERING_FANS_ROLL, team=team, rolls=[roll]))

        max_cheers = np.max(cheers)
        for i in range(len(self.game.state.teams)):
            team = self.game.state.teams[i]
            if max_cheers == cheers[i]:
                team.state.rerolls += 1
                self.game.report(Outcome(OutcomeType.EXTRA_REROLL, team=team))

        return True

    def available_actions(self):
        return []


class BrilliantCoaching(Procedure):

    def __init__(self, game):
        super().__init__(game)

    def step(self, action):

        rolls = []
        brilliant_coaches = []
        for team in self.game.state.teams:
            roll = DiceRoll([D3(self.game.rnd)], roll_type=RollType.BRILLIANT_COACHING_ROLL)
            rolls.append(roll)
            roll.modifiers = team.state.fame + team.ass_coaches
            brilliant_coaches.append(roll.get_result())
            self.game.report(Outcome(OutcomeType.BRILLIANT_COACHING_ROLL, team=team, rolls=[roll]))

        max_cheers = np.max(brilliant_coaches)
        for i in range(len(self.game.state.teams)):
            team = self.game.state.teams[i]
            if max_cheers == brilliant_coaches[i]:
                team.state.rerolls += 1
                self.game.report(Outcome(OutcomeType.EXTRA_REROLL, team=team))

        return True

    def available_actions(self):
        return []


class ThrowARock(Procedure):

    def __init__(self, game):
        super().__init__(game)
        self.rolled = False

    def step(self, action):

        rolls = []
        rocks = []
        for team in self.game.state.teams:
            roll = DiceRoll([D3(self.game.rnd)], roll_type=RollType.THROW_A_ROCK_ROLL)
            rolls.append(roll)
            roll.modifiers = team.state.fame
            rocks.append(roll.get_result())
            self.game.report(Outcome(OutcomeType.THROW_A_ROCK_ROLL, team=team, rolls=[roll]))

        for i in range(len(self.game.state.teams)):
            team = self.game.state.teams[i]
            if rocks[i] == 1:
                players = self.game.get_players_on_pitch(team)
                if len(players) > 0:
                    player = self.game.rnd.choice(players)
                    KnockDown(self.game, player, armor_roll=False)
                    self.game.report(Outcome(OutcomeType.HIT_BY_ROCK, player=player))

        return True

    def available_actions(self):
        return []


class PitchInvasionRoll(Procedure):

    def __init__(self, game, team, player):
        super().__init__(game)
        self.team = team
        self.player = player

    def step(self, action):
        roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.PITCH_INVASION_ROLL)
        roll.modifiers = self.game.get_opp_team(self.team).state.fame
        if roll.get_result() >= 6 and roll.get_sum() != 1:
            if self.player.has_skill(Skill.BALL_AND_CHAIN):
                self.game.report(Outcome(OutcomeType.KNOCKED_OUT, rolls=[roll], player=self.player, team=self.team))
                KnockOut(self.game, self.player, roll=roll)
            else:
                self.player.state.stunned = True
                self.player.state.up = False
                self.game.report(Outcome(OutcomeType.STUNNED, rolls=[roll], player=self.player, team=self.team))
        else:
            self.game.report(Outcome(OutcomeType.PLAYER_READY, rolls=[roll], player=self.player, team=self.team))

        return True

    def available_actions(self):
        return []


class KickoffTable(Procedure):

    def __init__(self, game, ball):
        super().__init__(game)
        self.ball = ball
        self.rolled = False

    def step(self, action):

        roll = DiceRoll([D6(self.game.rnd), D6(self.game.rnd)], roll_type=RollType.KICKOFF_ROLL)
        roll.result = roll.get_sum()

        if roll.result == 2:  # Get the ref!
            GetTheRef(self.game)
            self.game.report(Outcome(OutcomeType.KICKOFF_GET_THE_REF, rolls=[roll]))
        elif roll.result == 3:  # Riot!
            Riot(self.game)
            self.game.report(Outcome(OutcomeType.KICKOFF_RIOT, rolls=[roll]))
        elif roll.result == 4:  # Perfect defense
            Setup(self.game, team=self.game.get_kicking_team(), reorganize=True)
            self.game.report(Outcome(OutcomeType.KICKOFF_PERFECT_DEFENSE, team=self.game.get_kicking_team(),
                                     rolls=[roll]))
        elif roll.result == 5:  # High Kick
            HighKick(self.game, self.ball)
            self.game.report(Outcome(OutcomeType.KICKOFF_HIGH_KICK, rolls=[roll]))
        elif roll.result == 6:  # Cheering fans
            CheeringFans(self.game)
            self.game.report(Outcome(OutcomeType.KICKOFF_CHEERING_FANS, rolls=[roll]))
        elif roll.result == 7:  # Changing Weather
            WeatherTable(self.game, kickoff=True)
            self.game.report(Outcome(OutcomeType.KICKOFF_CHANGING_WHEATHER, rolls=[roll]))
        elif roll.result == 8:  # Brilliant Coaching
            BrilliantCoaching(self.game)
            self.game.report(Outcome(OutcomeType.KICKOFF_BRILLIANT_COACHING, rolls=[roll]))
        elif roll.result == 9:  # Quick Snap
            Turn(self.game, self.game.get_receiving_team(), None, None, quick_snap=True)
            self.game.report(Outcome(OutcomeType.KICKOFF_QUICK_SNAP, rolls=[roll]))
        elif roll.result == 10:  # Blitz
            Turn(self.game, self.game.get_kicking_team(), None, None, blitz=True)
            self.game.report(Outcome(OutcomeType.KICKOFF_BLITZ, rolls=[roll]))
        elif roll.result == 11:  # Throw a Rock
            ThrowARock(self.game)
            self.game.report(Outcome(OutcomeType.KICKOFF_THROW_A_ROCK, rolls=[roll]))
        elif roll.result == 12:  # Pitch Invasion
            for team in reversed(self.game.state.teams):
                for player in sorted(self.game.get_players_on_pitch(team), key=lambda p: p.nr, reverse=True):
                    PitchInvasionRoll(self.game, team, player)
            self.game.report(Outcome(OutcomeType.KICKOFF_PITCH_INVASION, rolls=[roll]))

        return True

    def available_actions(self):
        return []


class KnockDown(Procedure):

    def __init__(self, game, player, armor_roll=True, injury_roll=True, modifiers=0, inflictor=None,
                 in_crowd=False, modifiers_opp=0, turnover=False):
        super().__init__(game)
        self.player = player
        self.armor_roll = armor_roll
        self.injury_roll = injury_roll
        self.modifiers = modifiers
        self.modifiers_opp = modifiers_opp
        self.inflictor = inflictor
        self.in_crowd = in_crowd
        self.turnover = turnover

    def step(self, action):

        # Knock down player
        self.player.state.up = False
        if self.player.team == self.game.state.current_team:
            self.player.state.used = True

        self.game.report(Outcome(OutcomeType.KNOCKED_DOWN, player=self.player, opp_player=self.inflictor))

        # Turnover
        if self.turnover:
            Turnover(self.game)

        # Check fumble - does not handle throw-in
        ball = self.game.get_ball_at(self.player.position)
        if ball is not None:
            if not self.game.is_out_of_bounds(self.player.position):
                Bounce(self.game, ball)
            self.game.report(Outcome(OutcomeType.FUMBLE, player=self.player, opp_player=self.inflictor))

        # If armor roll should be made. Injury is also nested in armor.
        if self.injury_roll and not self.armor_roll:
            Injury(self.game, self.player, inflictor=self.inflictor if not self.in_crowd else None,
                   in_crowd=self.in_crowd)
        elif self.armor_roll:
            Armor(self.game, self.player, modifiers=self.modifiers, inflictor=self.inflictor)

        return True

    def available_actions(self):
        return []


class KnockOut(Procedure):

    def __init__(self, game, player, roll, inflictor=None):
        super().__init__(game)
        self.game = game
        self.player = player
        self.inflictor = inflictor
        self.roll = roll

    def step(self, action):
        if self.player.team.state.apothecary_available:
            Apothecary(self.game, self.player, roll=self.roll, outcome=OutcomeType.KNOCKED_OUT,
                       inflictor=self.inflictor)
            return True
        else:
            # Knock out player
            self.game.pitch_to_kod(self.player)
            self.game.report(Outcome(OutcomeType.KNOCKED_OUT, rolls=[self.roll], player=self.player,
                                     opp_player=self.inflictor))

        return True

    def available_actions(self):
        return []


class Move(Procedure):

    def __init__(self, game, player, position, gfi, dodge):
        super().__init__(game)
        self.player = player
        self.position = position
        if dodge:
            Dodge(game, player, position)
        if gfi:
            GFI(game, player, position)

    def step(self, action):

        self.game.move(self.player, self.position)

        # Check if player moved onto the ball
        ball = self.game.get_ball_at(self.player.position)
        if ball is not None and not ball.is_carried:

            # Attempt to pick up the ball - unless no hands
            if self.player.has_skill(Skill.NO_HANDS):
                Bounce(self.game, ball)
                return True
            else:
                Pickup(self.game, ball, self.player, cause_turnover=True)
                return True

        elif self.game.has_ball(self.player) and self.game.is_touchdown(self.player):

            # Touchdown if player had the ball with him/her
            Touchdown(self.game, self.player)

        return True

    def available_actions(self):
        return []


class GFI(Procedure):

    def __init__(self, game, player, position):
        super().__init__(game)
        self.player = player
        self.position = position
        self.awaiting_reroll = False
        self.sure_feet_used = False
        self.reroll_used = False
        self.rolled = False

    def step(self, action):

        # If player hasn't rolled
        if not self.rolled:

            # Roll
            roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.GFI_ROLL)
            self.rolled = True
            roll.target = 2
            roll.modifiers = 1 if self.game.state.weather == WeatherType.BLIZZARD else 0

            if roll.is_d6_success():

                # Success
                self.game.report(Outcome(OutcomeType.SUCCESSFUL_GFI, player=self.player, position=self.position, rolls=[roll]))
                return True

            else:

                # Fail
                self.game.report(Outcome(OutcomeType.FAILED_GFI, player=self.player, position=self.position, rolls=[roll]))

                # Check if sure feet
                if self.player.has_skill(Skill.SURE_FEET) and not self.sure_feet_used:
                    self.sure_feet_used = True
                    self.rolled = False
                    self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.player, skill=Skill.SURE_FEET))
                    return self.step(None)

                # Check if reroll available
                if self.game.can_use_reroll(self.player.team) and not self.sure_feet_used:
                    self.awaiting_reroll = True
                    return False

                # Player trips
                if not self.player.position == self.position:
                    self.game.move(self.player, self.position)
                KnockDown(self.game, self.player, self.position, turnover=True)
                return True

        # If reroll used
        if self.awaiting_reroll:
            if action.action_type == ActionType.USE_REROLL:
                # Remove reroll and roll again - recursive call
                self.game.report(Outcome(OutcomeType.REROLL_USED, team=self.player.team))
                self.rolled = False
                self.player.team.state.use_reroll()
                return self.step(None)
            else:
                # Player trips
                if not self.player.position == self.position:
                    self.game.move(self.player, self.position)
                KnockDown(self.game, self.player, turnover=True)
                return True

    def available_actions(self):
        if self.awaiting_reroll:
            return [ActionChoice(ActionType.USE_REROLL, team=self.game.state.current_team),
                    ActionChoice(ActionType.DONT_USE_REROLL, team=self.game.state.current_team)]

        return []


class Dodge(Procedure):

    def __init__(self, game, player, position, cause_turnover=True):
        super().__init__(game)
        self.player = player
        self.position = position
        self.dodge_used = False
        self.awaiting_dodge = False
        self.awaiting_reroll = False
        self.rolled = False
        self.cause_turnover = cause_turnover

    def step(self, action):

        # If player hasn't rolled
        if not self.rolled:

            # TODO: Allow player to select if shadowing and diving tackle
            # TODO: Put diving tackle, shadowing proc or tentacles on stack
            # TODO: Auto-use other skills

            # Roll
            roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.AGILITY_ROLL)
            self.rolled = True
            roll.modifiers = self.game.get_dodge_modifiers(self.player, self.position)

            # Get players with tackle
            tacklers = self.game.get_adjacent_opponents(self.player, down=False, skill=Skill.TACKLE)

            # Break tackle - use st instead of ag
            attribute = self.player.get_ag()
            if self.player.has_skill(Skill.BREAK_TACKLE) and self.player.get_st() > self.player.get_st():
                attribute = self.player.get_st()

            roll.target = Rules.agility_table[attribute]

            if roll.is_d6_success():

                # Success
                self.game.report(Outcome(OutcomeType.SUCCESSFUL_DODGE, player=self.player, position=self.position, rolls=[roll]))
                return True

            else:

                # Fail
                self.game.report(Outcome(OutcomeType.FAILED_DODGE, player=self.player, position=self.position, rolls=[roll]))

                # Check if dodge
                if self.player.has_skill(Skill.DODGE) and not self.player.has_used_skill(Skill.DODGE) and not self.dodge_used and len(tacklers) == 0:
                    self.dodge_used = True
                    self.awaiting_dodge = True
                    self.player.state.used_skills.append(Skill.DODGE)
                    self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.player, skill=Skill.DODGE))
                    return False

                # Check if reroll available
                if self.game.can_use_reroll(self.player.team) and not self.dodge_used:
                    self.awaiting_reroll = True
                    return False

                # Player trips
                self.game.move(self.player, self.position)
                KnockDown(self.game, self.player, turnover=self.cause_turnover)

                return True

        # If has dodge
        if self.awaiting_dodge:
            self.dodge_used = True
            self.rolled = False
            return self.step(None)

        # If reroll used
        if self.awaiting_reroll:
            if action.action_type == ActionType.USE_REROLL:
                # Remove reroll and roll again - recursive call
                self.game.report(Outcome(OutcomeType.REROLL_USED, team=self.player.team))
                self.player.team.state.use_reroll()
                self.rolled = False
                return self.step(None)
            else:
                # Player trips
                self.game.move(self.player, self.position)
                KnockDown(self.game, self.player, turnover=self.cause_turnover)
                return True

    def available_actions(self):
        if self.awaiting_reroll:
            return [ActionChoice(ActionType.USE_REROLL, team=self.player.team),
                    ActionChoice(ActionType.DONT_USE_REROLL, team=self.player.team)]
        return []


class TurnoverIfPossessionLost(Procedure):

    def __init__(self, game, ball):
        super().__init__(game)
        self.ball = ball

    def step(self, action):
        player_at = self.game.get_player_at(self.ball.position)
        if player_at is None:
            Turnover(self.game)
        elif player_at.team != self.game.state.current_team:
            Turnover(self.game)
        return True

    def available_actions(self):
        return []


class Handoff(Procedure):

    def __init__(self, game, ball, player, pos_to, catcher):
        super().__init__(game)
        self.ball = ball
        self.player = player
        self.pos_to = pos_to
        self.catcher = catcher

    def step(self, action):
        self.ball.move_to(self.pos_to)
        TurnoverIfPossessionLost(self.game, self.ball)
        self.ball.move_to(self.catcher.position)
        Catch(self.game, self.catcher, self.ball, handoff=True)
        self.game.report(Outcome(OutcomeType.HANDOFF, player=self.player, opp_player=self.catcher))
        return True

    def available_actions(self):
        return []


class PassAction(Procedure):

    def __init__(self, game, ball, passer, catcher, position, pass_distance):
        super().__init__(game)
        self.ball = ball
        self.passer = passer
        self.catcher = catcher
        self.position = position
        self.pass_distance = pass_distance
        self.pass_roll = None
        self.reroll_used = False
        self.pass_used = False
        self.waiting_for_reroll = False
        self.waiting_for_pass = False
        self.fumble = False
        self.interception_tried = False

    def step(self, action):

        # Otherwise roll if player hasn't rolled
        if self.pass_roll is None:

            self.ball.is_carried = False

            # Check for interception
            if not self.interception_tried:
                interceptors = self.game.get_interceptors(self.passer.position, self.position, team=self.game.get_opp_team(self.passer.team))
                # TODO: Implement option to intercept for all opponent teams
                if len(interceptors) > 0:
                    Interception(self.game, interceptors[0].team, self.ball, interceptors)
                    self.interception_tried = True
                    return False

            self.ball.move_to(self.passer.position)

            # Roll
            self.pass_roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.AGILITY_ROLL)
            self.pass_roll.target = Rules.agility_table[self.passer.get_ag()]
            self.pass_roll.modifiers = self.game.get_pass_modifiers(self.passer, self.pass_distance)
            result = self.pass_roll.get_sum()
            mod_result = result + self.pass_roll.modifiers

            if result == 6 or (result != 1 and mod_result >= self.pass_roll.target):
                # Accurate pass
                self.game.report(Outcome(OutcomeType.ACCURATE_PASS, player=self.passer, rolls=[self.pass_roll]))
                self.ball.move_to(self.position)
                TurnoverIfPossessionLost(self.game, self.ball)
                if self.catcher is not None:
                    Catch(self.game, self.catcher, self.ball, accurate=True)
                else:
                    Bounce(self.game, self.ball)
                return True

            elif result == 1 or mod_result <= 1:
                # Fumble
                self.fumble = True
                self.game.report(Outcome(OutcomeType.FUMBLE, player=self.passer, rolls=[self.pass_roll]))
            else:
                # Inaccurate pass
                self.game.report(Outcome(OutcomeType.INACCURATE_PASS, player=self.passer, rolls=[self.pass_roll]))

            # Check if player has pass
            if self.passer.has_skill(Skill.PASS) and not self.pass_used:
                self.pass_used = True
                self.pass_roll = None
                self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.passer, skill=Skill.PASS))
                self.fumble = False
                return self.step(None)

            # Check if reroll available
            if self.game.can_use_reroll(self.passer.team) and not self.pass_used and not self.reroll_used:
                self.waiting_for_reroll = True
                return False

            # Effect
            if self.fumble:
                Turnover(self.game)
                Bounce(self.game, self.ball)
            else:
                TurnoverIfPossessionLost(self.game, self.ball)
                self.ball.move_to(self.position)
                Scatter(self.game, self.ball, is_pass=True)

            return True

        # If re-roll used
        if self.waiting_for_reroll:
            if action.action_type == ActionType.USE_REROLL:
                self.reroll_used = True
                self.game.report(Outcome(OutcomeType.REROLL_USED, team=self.passer.team))
                self.passer.team.state.use_reroll()
                self.fumble = False
                self.pass_roll = None
                return self.step(None)
            elif self.fumble:
                Turnover(self.game)
                Bounce(self.game, self.ball)
                return True

            TurnoverIfPossessionLost(self.game, self.ball)
            self.ball.move_to(self.position)
            Scatter(self.game, self.ball, is_pass=True)
            return True

        return True

    def available_actions(self):

        if self.waiting_for_reroll:
            return [ActionChoice(ActionType.USE_REROLL, team=self.passer.team),
                    ActionChoice(ActionType.DONT_USE_REROLL, team=self.passer.team)]
        return []


class Pickup(Procedure):

    def __init__(self, game, ball, player, cause_turnover=True):
        super().__init__(game)
        self.ball = ball
        self.player = player
        self.rolled = False
        self.sure_hands_used = False
        self.waiting_for_reroll = False
        self.waiting_for_sure_hands = False
        self.cause_turnover = cause_turnover

    def step(self, action):

        # Otherwise roll if player hasn't rolled
        if not self.rolled:

            roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.AGILITY_ROLL)
            roll.target = Rules.agility_table[self.player.get_ag()]
            roll.modifiers = self.game.get_pickup_modifiers(self.player, self.ball.position)

            # Roll
            self.rolled = True
            if roll.is_d6_success():
                self.game.report(Outcome(OutcomeType.SUCCESSFUL_PICKUP, player=self.player, rolls=[roll]))
                self.ball.is_carried = True
                if self.game.is_touchdown(self.player):
                    Touchdown(self.game, self.player)
                return True
            else:
                self.game.report(Outcome(OutcomeType.FAILED_PICKUP, player=self.player, position=self.player.position,
                                         rolls=[roll]))
                # Check if sure hands
                if self.player.has_skill(Skill.SURE_HANDS) and not self.sure_hands_used:
                    self.sure_hands_used = True
                    self.waiting_for_sure_hands = True
                    self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.player, position=self.player.position,
                                             rolls=[roll], skill=Skill.SURE_HANDS))
                    return False

                # Check if reroll available
                if self.game.can_use_reroll(self.player.team) and not self.sure_hands_used:
                    self.waiting_for_reroll = True
                    return False

                # Turnover?
                if self.cause_turnover:
                    Turnover(self.game)

                Bounce(self.game, self.ball)
                self.game.report(Outcome(OutcomeType.FAILED_PICKUP, player=self.player, rolls=[roll]))

                return True

        # If has sure hands
        if self.waiting_for_sure_hands:
            self.sure_hands_used = True
            self.rolled = False
            return self.step(None)

        # If re-roll used
        if self.waiting_for_reroll:
            if action.action_type == ActionType.USE_REROLL:
                self.player.team.state.use_reroll()
                self.game.report(Outcome(OutcomeType.REROLL_USED, team=self.player.team))
                self.rolled = False
                return self.step(None)
            else:
                # Turnover?
                if self.cause_turnover:
                    Turnover(self.game)
                Bounce(self.game, self.ball)

        return True

    def available_actions(self):
        if self.waiting_for_reroll:
            return [ActionChoice(ActionType.USE_REROLL, team=self.player.team),
                    ActionChoice(ActionType.DONT_USE_REROLL, team=self.player.team)]
        return []


class StandUp(Procedure):

    def __init__(self, game, player, roll=False):
        super().__init__(game)
        self.player = player
        self.roll = roll

    def step(self, action):

        if self.roll:
            roll = DiceRoll([D6(self.game.rnd)], target=4, roll_type=RollType.STAND_UP_ROLL)
            if roll.is_d6_success():
                self.player.state.up = True
                self.game.report(Outcome(OutcomeType.PLAYER_STAND_UP_SUCCESS, rolls=[roll], player=self.player))
            else:
                self.player.state.up = False
                self.game.report(Outcome(OutcomeType.PLAYER_STAND_UP_FAILURE, rolls=[roll], player=self.player))
        else:
            self.player.state.up = True

        return True

    def available_actions(self):
        return []


class PlaceBall(Procedure):

    def __init__(self, game, ball):
        super().__init__(game)
        self.ball = ball
        self.aa = None

    def setup(self):
        self.game.add_secondary_clock(self.game.get_kicking_team())
        self.aa = [ActionChoice(ActionType.PLACE_BALL, team=self.game.get_kicking_team(),
                                positions=self.game.get_team_side(self.game.get_receiving_team()))]

    def step(self, action):
        self.game.state.pitch.balls.append(self.ball)
        self.ball.on_ground = False
        self.ball.move_to(action.position)
        self.game.report(Outcome(OutcomeType.BALL_PLACED, position=action.position, team=self.game.get_kicking_team()))
        self.game.remove_secondary_clocks()
        return True

    def available_actions(self):
        return self.aa


class EndPlayerTurn(Procedure):

    def __init__(self, game, player):
        super().__init__(game)
        self.player = player

    def step(self, action):
        self.player.state.used = True
        self.player.state.moves = 0
        self.game.report(Outcome(OutcomeType.END_PLAYER_TURN, player=self.player))
        self.game.state.active_player = None
        self.player.state.squares_moved.clear()
        return True

    def available_actions(self):
        return []


class PlayerAction(Procedure):

    def __init__(self, game, player, player_action_type, turn):
        super().__init__(game)
        self.player = player
        self.player_action_type = player_action_type
        self.blitz_block = False
        self.turn = turn

    def step(self, action):

        if len(self.player.state.squares_moved) == 0:
            self.player.state.squares_moved.append(self.player.position)

        if action.action_type == ActionType.END_PLAYER_TURN:
            EndPlayerTurn(self.game, self.player)
            return True

        if action.action_type == ActionType.STAND_UP:

            StandUp(self.game, self.player, roll=self.player.get_ma() < 3)
            self.player.state.moves += min(self.player.get_ma(), 3)
            for i in range(3):
                self.player.state.squares_moved.append(self.player.position)

            return False

        # Action attributes
        player_to = self.game.get_player_at(action.position)

        if action.action_type == ActionType.MOVE:

            # Check GFI
            gfi = self.player.state.moves + 1 > self.player.get_ma()

            # Check dodge
            dodge = False
            if not self.turn.quick_snap:
                if self.player.has_skill(Skill.BALL_AND_CHAIN):
                    # Ball and chain -> Auto-dodge
                    dodge = True
                    gfi = False
                else:
                    tackle_zones_from = self.game.num_tackle_zones_in(self.player)
                    dodge = tackle_zones_from > 0

            # Add proc
            Move(self.game, self.player, action.position, gfi, dodge)
            self.player.state.squares_moved.append(action.position)

            self.player.state.moves += 1

            return False

        elif action.action_type == ActionType.BLOCK:

            self.blitz_block = self.player_action_type == PlayerActionType.BLITZ

            # Check GFI
            gfi = False
            gfi_frenzy = False
            frenzy_allowed = True
            if self.player_action_type == PlayerActionType.BLITZ:
                move_needed = 3 if not self.player.state.up else 1
                gfis = 3 if self.player.has_skill(Skill.SPRINT) else 2
                gfi = self.player.state.moves + move_needed > self.player.get_ma()
                gfi_frenzy = self.player.state.moves + move_needed + 1 > self.player.get_ma()
                frenzy_allowed = self.player.state.moves + move_needed + 1 < self.player.get_ma() + gfis
                self.player.state.moves += move_needed
                if not self.player.state.up:
                    for i in range(move_needed-1):
                        self.player.state.squares_moved.append(self.player.position)
                self.player.state.up = True

            # End turn after block - if not a blitz action
            if self.player_action_type == PlayerActionType.BLOCK:
                EndPlayerTurn(self.game, self.player)

            # Frenzy second block
            if self.player.has_skill(Skill.FRENZY) and frenzy_allowed:
                Block(self.game, self.player, player_to, blitz=self.blitz_block, gfi=gfi_frenzy, frenzy_block=True)

            # Regular block
            Block(self.game, self.player, player_to, blitz=self.blitz_block, gfi=gfi)

            if self.player_action_type == PlayerActionType.BLOCK:
                return True

            return False

        elif action.action_type == ActionType.FOUL:

            EndPlayerTurn(self.game, self.player)
            Foul(self.game, self.player, player_to)

            return True

        elif action.action_type == ActionType.HANDOFF:

            EndPlayerTurn(self.game, self.player)
            Handoff(self.game, self.game.get_ball_at(self.player.position), self.player, action.position,
                    self.game.get_player_at(action.position))

            return True

        elif action.action_type == ActionType.PASS:

            # Check distance
            pass_distance = self.game.get_pass_distance(self.player.position, action.position)
            EndPlayerTurn(self.game, self.player)
            PassAction(self.game, self.game.get_ball_at(self.player.position), self.player, player_to, action.position,
                       pass_distance)
            self.turn.pass_available = False

            return True

    def available_actions(self):
        actions = []

        # Move actions
        if self.player_action_type != PlayerActionType.BLOCK:
            move_positions = []
            agi_rolls = []
            move_needed = 1 if not self.player.state.up else 1
            gfi = self.player.state.moves + move_needed > self.player.get_ma()
            sprints = 3 if self.player.has_skill(Skill.SPRINT) else 2
            if not self.player.state.up:
                if self.player.get_ma() < 3:
                    agi_rolls.append([4])
                else:
                    agi_rolls.append([])
                actions.append(ActionChoice(ActionType.STAND_UP, team=self.player.team, agi_rolls=agi_rolls))
                # TODO: Check if position is necessary here? ^^
            elif (not self.turn.quick_snap
                  and self.player.state.moves + move_needed <= self.player.get_ma() + sprints) \
                    or (self.turn.quick_snap and self.player.state.moves == 0):
                for square in self.game.get_adjacent_squares(self.player.position, occupied=False):
                    ball_at = self.game.get_ball_at(square)
                    move_positions.append(square)
                    rolls = []
                    if not self.turn.quick_snap:
                        if gfi:
                            rolls.append(2)
                        if self.game.num_tackle_zones_in(self.player) > 0:
                            modifiers = self.game.get_dodge_modifiers(self.player, square)
                            target = Rules.agility_table[self.player.get_ag()]
                            rolls.append(min(6, max(2, target - modifiers)))
                        if ball_at is not None and ball_at.on_ground:
                            target = Rules.agility_table[self.player.get_ag()]
                            modifiers = self.game.get_pickup_modifiers(self.player, square)
                            rolls.append(min(6, max(2, target - modifiers)))
                    agi_rolls.append(rolls)
                if len(move_positions) > 0:
                    actions.append(ActionChoice(ActionType.MOVE, team=self.player.team,
                                                positions=move_positions, agi_rolls=agi_rolls))

        # Block actions
        if self.player_action_type == PlayerActionType.BLOCK or (self.player_action_type == PlayerActionType.BLITZ
                                                                 and not self.blitz_block):

            can_block = self.player.state.up    # Ignore JUMP_UP for now
            # Check movement left if blitz,
            gfi = False
            if self.player_action_type == PlayerActionType.BLITZ:
                move_needed = 1
                gfi_allowed = 3 if self.player.has_skill(Skill.SPRINT) else 2
                if self.player.state.moves + move_needed > self.player.get_ma() + gfi_allowed:
                    can_block = False
                gfi = self.player.state.moves + move_needed > self.player.get_ma()

            # Find adjacent enemies to block
            if can_block:
                block_positions = []
                block_rolls = []
                for player_to in self.game.get_adjacent_opponents(self.player, down=False):
                    block_positions.append(player_to.position)
                    dice = self.game.num_block_dice(attacker=self.player, defender=player_to,
                                                       blitz=self.player_action_type == PlayerActionType.BLITZ,
                                                       dauntless_success=False)
                    block_rolls.append(dice)
                if len(block_positions) > 0:
                    agi_rolls = [([2] if gfi else []) for _ in block_positions]
                    actions.append(ActionChoice(ActionType.BLOCK, team=self.player.team,
                                                positions=block_positions, block_rolls=block_rolls,
                                                agi_rolls=agi_rolls))

        # Foul actions
        if self.player_action_type == PlayerActionType.FOUL:
            foul_positions = []
            foul_rolls = []
            for player_to in self.game.get_adjacent_opponents(self.player, standing=False, down=True):
                foul_positions.append(player_to.position)
                armor = player_to.get_av()
                assists_from = self.game.get_assisting_players(self.player, player_to, foul=True)
                assists_to = self.game.get_assisting_players(player_to, self.player, foul=True)
                foul_rolls.append(min(0, armor + 1 - len(assists_from) + len(assists_to)))

            if len(foul_positions) > 0:
                actions.append(ActionChoice(ActionType.FOUL, team=self.player.team,
                                            positions=foul_positions, block_rolls=foul_rolls))

        # Handoff actions
        if self.player_action_type == PlayerActionType.HANDOFF and self.game.has_ball(self.player):
            hand_off_positions = []
            agi_rolls = []
            for player_to in self.game.get_adjacent_teammates(self.player):
                if player_to.can_catch():
                    hand_off_positions.append(player_to.position)
                    modifiers = self.game.get_catch_modifiers(self.player, player_to.position)
                    target = Rules.agility_table[self.player.get_ag()]
                    agi_rolls.append([min(6, max(2, target - modifiers))])

            if len(hand_off_positions) > 0:
                actions.append(ActionChoice(ActionType.HANDOFF, team=self.player.team,
                                            positions=hand_off_positions, agi_rolls=agi_rolls))

        # Pass actions
        if self.player_action_type == PlayerActionType.PASS and self.game.has_ball(self.player):
            positions, distances = self.game.get_pass_distances(self.player)
            agi_rolls = []
            cache = {}
            for i in range(len(distances)):
                distance = distances[i]
                position = positions[i]
                if distance not in cache:
                    modifiers = self.game.get_pass_modifiers(self.player, distance)
                    target = Rules.agility_table[self.player.get_ag()]
                    cache[distance] = min(6, max(2, target - modifiers))
                rolls = [cache[distance]]
                player_to = self.game.get_player_at(position)
                if player_to is not None and player_to.team == self.player.team \
                        and player_to.can_catch():
                    catch_target = Rules.agility_table[player_to.get_ag()]
                    catch_modifiers = self.game.get_catch_modifiers(player_to, accurate=True)
                    rolls.append(min(6, max(2, catch_target - catch_modifiers)))
                agi_rolls.append(rolls)
            if len(positions) > 0:
                actions.append(ActionChoice(ActionType.PASS, team=self.player.team,
                                            positions=positions, agi_rolls=agi_rolls))

        actions.append(ActionChoice(ActionType.END_PLAYER_TURN, team=self.player.team))
        return actions


class StartGame(Procedure):

    def __init__(self, game):
        super().__init__(game)

    def step(self, action):
        for team in self.game.state.teams:
            for player in team.players:
                self.game.get_reserves(team).append(player)
        self.game.start_time = time.time()
        self.game.report(Outcome(OutcomeType.GAME_STARTED))
        return True

    def available_actions(self):
        return [ActionChoice(ActionType.START_GAME, team=self.game.state.away_team)]


class EndGame(Procedure):

    def __init__(self, game):
        super().__init__(game)

    def step(self, action):
        self.game.state.game_over = True
        winner = self.game.get_winning_team()
        if winner is not None:
            self.game.report(Outcome(OutcomeType.END_OF_GAME_WINNER, team=winner))
        else:
            self.game.report(Outcome(OutcomeType.END_OF_GAME_DRAW))
        return True

    def available_actions(self):
        return []


class Pregame(Procedure):

    def __init__(self, game):
        super().__init__(game)
        CoinTossFlip(self.game)
        # self.game.state.stack.push(Inducements(self.game, True))
        # self.game.state.stack.push(Inducements(self.game, False))
        # self.game.state.stack.push(GoldToPettyCash(self.game))
        Fans(self.game)
        WeatherTable(self.game)
        StartGame(self.game)

    def step(self, action):
        Half(self.game, 2)
        Half(self.game, 1)
        self.game.state.half = 1
        for team in self.game.state.teams:
            team.state.turn = 0
        return True

    def available_actions(self):
        return []


class PreKickOff(Procedure):

    def __init__(self, game, team):
        super().__init__(game)
        self.team = team
        self.checked = []

    def step(self, action):
        # Check KOed
        for player in self.game.get_knocked_out(self.team):
            if player not in self.checked:
                roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.KO_READY_ROLL)
                if roll.get_sum() >= 4:
                    self.game.kod_to_reserves(player)
                    self.checked.append(player)
                    self.game.report(Outcome(OutcomeType.PLAYER_READY, player=player, rolls=[roll]))
                    return False
                self.checked.append(player)
                self.game.report(Outcome(OutcomeType.PLAYER_NOT_READY, player=player, rolls=[roll]))
                return False
        return True

    def available_actions(self):
        return []


class FollowUp(Procedure):

    def __init__(self, game, attacker, defender, pos_to):
        super().__init__(game)
        self.attacker = attacker
        self.defender = defender
        self.pos_to = pos_to

    def step(self, action):
        if self.defender.has_skill(Skill.FEND):
            self.game.report(Outcome(OutcomeType.SKILL_USED, skill=Skill.FEND, player=self.defender))
            self.attacker.state.squares_moved.append(self.attacker.position)
        elif self.attacker.has_skill(Skill.FRENZY) or action.position == self.pos_to:
            self.game.move(self.attacker, self.pos_to)
            self.attacker.state.squares_moved.append(self.pos_to)
            self.game.report(Outcome(OutcomeType.FOLLOW_UP, position=self.pos_to, player=self.attacker))
        else:
            self.attacker.state.squares_moved.append(self.attacker.position)
        return True

    def available_actions(self):
        if self.attacker.has_skill(Skill.FRENZY) or self.defender.has_skill(Skill.FEND):
            return []
        else:
            return [ActionChoice(ActionType.FOLLOW_UP, team=self.attacker.team,
                                 positions=[self.attacker.position, self.pos_to])]


class Push(Procedure):

    def __init__(self, game, pusher, player, knock_down=False, blitz=False, chain=False):
        super().__init__(game)
        self.pusher = pusher
        self.player = player
        self.knock_down = knock_down
        self.blitz = blitz
        self.waiting_stand_firm = False
        self.stand_firm_used = False
        self.chain = chain
        self.waiting_for_move = False
        self.player_chain = None
        self.push_to = None
        self.follow_to = None
        self.squares = None
        self.selector = pusher
        self.crowd = False

    def step(self, action):

        # When proceeding pushes are over, move player(s)
        if self.waiting_for_move:

            # Stop secondary clocks
            self.game.remove_secondary_clocks()

            # Move pushed player
            self.game.move(self.player, self.push_to)

            # Ball
            if self.game.has_ball(self.player):
                if self.game.is_touchdown(self.player):
                    Touchdown(self.game, self.player)
                elif self.game.is_out_of_bounds(self.player.position):
                    ball = self.game.get_ball_at(self.player.position)
                    ThrowIn(self.game, ball, position=self.follow_to)
            elif self.game.get_ball_at(self.push_to) is not None:
                Bounce(self.game, self.game.get_ball_at(self.push_to))

            # Knock down
            if self.knock_down or self.crowd:
                KnockDown(self.game, self.player, in_crowd=self.crowd, armor_roll=not self.crowd)

            # Chain push
            if not self.chain:
                FollowUp(self.game, self.pusher, self.player, self.follow_to)

            return True

        # Use stand firm
        if self.waiting_stand_firm:
            if action.action_type == ActionType.USE_STAND_FIRM:
                return True
            else:
                self.waiting_stand_firm = False
                self.stand_firm_used = True

        # Stand firm
        if self.player.has_skill(Skill.STAND_FIRM) and not self.stand_firm_used:
            if not self.pusher.has_skill(Skill.JUGGERNAUT):
                self.waiting_stand_firm = True
                return False

        # Get possible squares
        if self.squares is None:
            # Sidestep and grab cancels out eachother - otherwise let the grabber or sidestepper select adjacent square
            if self.player.has_skill(Skill.SIDE_STEP) and not self.pusher.has_skill(Skill.GRAB):
                self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.player, skill=Skill.SIDE_STEP))
                self.squares = self.game.get_adjacent_squares(self.player.position, occupied=False)
                self.selector = self.player
                if len(self.squares) > 0:
                    if self.player.team != self.game.state.current_team:
                        self.game.add_secondary_clock(self.player.team)
            elif self.pusher.has_skill(Skill.GRAB) and not self.player.has_skill(Skill.SIDE_STEP):
                self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.player, skill=Skill.GRAB))
                self.squares = self.game.get_adjacent_squares(self.player.position, occupied=False)
            # If no free squares
            if self.squares is None or len(self.squares) == 0:
                self.squares = self.game.get_push_squares(self.pusher.position, self.player.position)
            return False

        if action.action_type == ActionType.PUSH:

            # Push to crowd?
            self.crowd = self.game.is_out_of_bounds(action.position)

            # Report
            if self.crowd:
                self.game.report(Outcome(OutcomeType.PUSHED_INTO_CROWD, player=self.player))
            else:
                self.game.report(Outcome(OutcomeType.PUSHED, player=self.player, position=action.position))

            # Follow up - wait if push is delayed
            player_at = self.game.get_player_at(action.position)

            # Move players in next step - after chaining
            self.waiting_for_move = True
            self.squares = None

            # Save positions before chaining
            self.push_to = Square(action.position.x, action.position.y)
            self.follow_to = Square(self.player.position.x, self.player.position.y)

            assert self.push_to != self.follow_to

            # Chain push
            if player_at is not None:
                self.player_chain = player_at
                Push(self.game, self.player, self.player_chain, knock_down=False, chain=True)

            return False

        raise Exception("Unknown push sequence")

    def available_actions(self):
        actions = []
        if self.squares is not None:
            actions.append(ActionChoice(ActionType.PUSH, team=self.selector.team, positions=self.squares))
        elif self.waiting_stand_firm:
            actions.append(ActionChoice(ActionType.USE_SKILL, team=self.player.team))
            actions.append(ActionChoice(ActionType.DONT_USE_SKILL, team=self.player.team))
        return actions


class Scatter(Procedure):

    def __init__(self, game, ball, kick=False, is_pass=False, gentle_gust=False):
        super().__init__(game)
        self.ball = ball
        self.kick = kick
        self.is_pass = is_pass
        self.gentle_gust = gentle_gust

    def step(self, action):

        # Don't scatter if ball is out
        if self.game.is_out_of_bounds(self.ball.position):
            return True

        n = 3 if self.is_pass else 1
        rolls = [DiceRoll([D8(self.game.rnd)], roll_type=RollType.SCATTER_ROLL) for _ in range(n)]

        for s in range(n):

            # Roll
            roll_scatter = rolls[s]
            if self.kick and not self.gentle_gust:
                if self.game.config.kick_scatter_dice == 'd6':
                    distance_dice = D6(self.game.rnd)
                elif self.game.config.kick_scatter_dice == 'd3':
                    distance_dice = D6(self.game.rnd)
                else:
                    raise Exception("Unknown kick_roll_distance")
                roll_distance = DiceRoll([distance_dice], roll_type=RollType.DISTANCE_ROLL)
                rolls += [roll_distance]

            x = 0
            y = 0
            if roll_scatter.get_sum() in [1, 4, 6]:
                x = -1
            if roll_scatter.get_sum() in [3, 5, 8]:
                x = 1
            if roll_scatter.get_sum() in [1, 2, 3]:
                y = -1
            if roll_scatter.get_sum() in [6, 7, 8]:
                y = 1
            distance = 1 if not self.kick or self.gentle_gust else roll_distance.get_sum()

            for i in range(distance):
                # Move ball on square
                self.ball.move(x, y)

                if self.kick and i == 0 and not self.gentle_gust:
                    self.game.report(Outcome(OutcomeType.BALL_SCATTER, rolls=rolls))

                # Check out of bounds
                if self.kick:
                    if self.game.is_out_of_bounds(self.ball.position):
                        if self.gentle_gust:
                            # Touchback will be enforced after kick-off table when ball lands
                            self.game.report(Outcome(OutcomeType.GENTLE_GUST_OUT_OF_BOUNDS, position=self.ball.position,
                                                     rolls=rolls))
                        else:
                            # Touchback will be enforced after kick-off table when ball lands
                            self.game.report(Outcome(OutcomeType.KICK_OUT_OF_BOUNDS, position=self.ball.position,
                                                     rolls=rolls))
                        return True
                    elif self.game.is_team_side(self.ball.position, self.game.get_kicking_team()):
                        if self.gentle_gust:
                            # Touchback will be enforced after kick-off table when ball lands
                            self.game.report(Outcome(OutcomeType.GENTLE_GUST_OPP_HALF, position=self.ball.position,
                                                     rolls=rolls))
                        else:
                            # Touchback will be enforced after kick-off table when ball lands
                            self.game.report(Outcome(OutcomeType.KICK_OPP_HALF, position=self.ball.position, rolls=rolls))
                        return True
                else:
                    # Throw in
                    if self.game.is_out_of_bounds(self.ball.position):
                        ThrowIn(self.game, self.ball, Square(self.ball.position.x-x, self.ball.position.y-y))
                        self.game.report(Outcome(OutcomeType.BALL_SCATTER, rolls=rolls))
                        self.game.report(Outcome(OutcomeType.BALL_OUT_OF_BOUNDS,
                                                 position=self.ball.position))
                        return True

                    # Passes are scattered three times
                    if self.is_pass and s < n-1:
                        continue

                    self.game.report(Outcome(OutcomeType.BALL_SCATTER, rolls=rolls))

                    # On player -> Catch
                    player = self.game.get_player_at(self.ball.position)
                    if player is not None:
                        Catch(self.game, player, self.ball)
                        self.game.report(Outcome(OutcomeType.BALL_HIT_PLAYER, position=self.ball.position,
                                                 player=player, rolls=[roll_scatter]))
                        return True

        if self.kick:
            if self.gentle_gust:
                # Wait for ball to land
                self.game.report(Outcome(OutcomeType.GENTLE_GUST_IN_BOUNDS, position=self.ball.position,
                                         rolls=[roll_scatter]))
            else:
                # Wait for ball to land
                self.game.report(Outcome(OutcomeType.KICK_IN_BOUNDS, position=self.ball.position,
                                         rolls=[roll_scatter, roll_distance]))
        else:
            # Bounce ball
            Bounce(self.game, self.ball)

        return True

    def available_actions(self):
        return []


class ClearBoard(Procedure):

    def __init__(self, game):
        super().__init__(game)

    def step(self, action):
        self.game.state.pitch.balls.clear()
        for team in self.game.state.teams:
            for player in team.players:
                # If player not in reserves. move it to it
                if player.position is not None:
                    # Set to ready
                    player.state.reset()
                    player.state.up = True
                    # Remove from pitch
                    self.game.pitch_to_reserves(player)
                    # Check if heat exhausted
                    if self.game.state.weather == WeatherType.SWELTERING_HEAT:
                        roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.SWELTERING_HEAT_ROLL)
                        if roll.get_sum() == 1:
                            player.state.heated = True
                            self.game.report(Outcome(OutcomeType.PLAYER_HEATED, player=player, rolls=[roll]))
                        else:
                            self.game.report(Outcome(OutcomeType.PLAYER_NOT_HEATED, player=player, rolls=[roll]))
        return True

    def available_actions(self):
        return []


class Setup(Procedure):

    def __init__(self, game, team, reorganize=False):
        super().__init__(game)
        self.team = team
        self.reorganize = reorganize
        self.selected_player = None
        self.formations = game.config.defensive_formations if team == game.get_kicking_team() \
            else game.config.offensive_formations
        self.started = False
        self.aa = []

    def setup(self):
        # Set current team
        self.game.state.current_team = self.team
        # Add clock
        self.game.add_primary_clock(self.team)

    def step(self, action):
        formation = None
        if action.action_type == ActionType.SETUP_FORMATION_WEDGE:
            formation = [formation for formation in self.formations if formation.name == "Wedge"][0]
        if action.action_type == ActionType.SETUP_FORMATION_ZONE:
            formation = [formation for formation in self.formations if formation.name == "Zone"][0]
        if action.action_type == ActionType.SETUP_FORMATION_LINE:
            formation = [formation for formation in self.formations if formation.name == "Line"][0]
        if action.action_type == ActionType.SETUP_FORMATION_SPREAD:
            formation = [formation for formation in self.formations if formation.name == "Spread"][0]

        if formation is not None:
            actions = formation.actions(self.game, self.team)
            for action in actions:
                self.step(action)
            return False

        if action.action_type == ActionType.END_SETUP:
            if not self.game.is_setup_legal_count(self.team, max_players=self.game.config.pitch_max,
                                            min_players=self.game.config.pitch_min):
                self.game.report(Outcome(OutcomeType.ILLEGAL_SETUP_NUM, team=self.team))
                return False
            elif not self.game.is_setup_legal_scrimmage(self.team, min_players=self.game.config.scrimmage_min):
                self.game.report(Outcome(OutcomeType.ILLEGAL_SETUP_SCRIMMAGE, team=self.team))
                return False
            elif not self.game.is_setup_legal_wings(self.team, max_players=self.game.config.wing_max):
                self.game.report(Outcome(OutcomeType.ILLEGAL_SETUP_WINGS, team=self.team))
                return False
            self.game.state.current_team = None
            self.game.remove_clocks()
            self.game.report(Outcome(OutcomeType.SETUP_DONE, team=self.team))
            return True

        if action.action_type == ActionType.PLACE_PLAYER:
            if action.position is None:
                self.game.pitch_to_reserves(action.player)
            elif action.player.position is None:
                self.game.reserves_to_pitch(action.player, action.position)
            else:
                player_at = self.game.get_player_at(action.position)
                if player_at is not None:
                    self.game.swap(action.player, player_at)
                else:
                    self.game.move(action.player, action.position)
            self.game.report(Outcome(OutcomeType.PLAYER_PLACED, position=action.position, player=action.player))
            return False

    def available_actions(self):
        # Set actions
        if self.reorganize:
            positions = self.game.get_team_side(self.team)
        else:
            positions = self.game.get_team_side(self.team) + [None]

        aa = [
            ActionChoice(ActionType.PLACE_PLAYER, team=self.team,
                         players=self.game.get_players_on_pitch(
                             self.team) if self.reorganize else self.game.get_players_on_pitch(
                             self.team) +self.game.get_reserves(self.team),
                         positions=positions),
            ActionChoice(ActionType.END_SETUP, team=self.team)
        ]
        if not self.reorganize:
            for formation in self.formations:
                if formation.name == "Wedge":
                    aa.append(ActionChoice(ActionType.SETUP_FORMATION_WEDGE, team=self.team))
                if formation.name == "Line":
                    aa.append(ActionChoice(ActionType.SETUP_FORMATION_LINE, team=self.team))
                if formation.name == "Spread":
                    aa.append(ActionChoice(ActionType.SETUP_FORMATION_SPREAD, team=self.team))
                if formation.name == "Zone":
                    aa.append(ActionChoice(ActionType.SETUP_FORMATION_ZONE, team=self.team))

        return aa


class ThrowIn(Procedure):

    def __init__(self, game, ball, position):
        super().__init__(game)
        self.ball = ball
        self.position = position

    def step(self, action):

        # Move ball to throw-in position
        self.ball.move_to(self.position)
        self.ball.carried = False

        # Roll
        roll_direction = DiceRoll([D3(self.game.rnd)], roll_type=RollType.SCATTER_ROLL)
        roll_distance = DiceRoll([D6(self.game.rnd), D6(self.game.rnd)], roll_type=RollType.DISTANCE_ROLL)

        # Scatter
        x = 0
        y = 0

        if self.ball.position.y == 1 and self.ball.position.x == 1:  # Top left corner
            y = 0 if roll_direction.get_sum() == 1 else 1
            x = 0 if roll_direction.get_sum() == 3 else 1
        elif self.ball.position.y == 1 and self.ball.position.x == len(self.game.arena.board[0])-2:  # Top right corner
            y = 0 if roll_direction.get_sum() == 3 else 1
            x = 0 if roll_direction.get_sum() == 1 else -1
        elif self.ball.position.y == len(self.game.arena.board)-2 and self.ball.position.x == len(self.game.arena.board[0])-2:  # Bottom right corner
            y = 0 if roll_direction.get_sum() == 1 else -1
            x = 0 if roll_direction.get_sum() == 3 else -1
        elif self.ball.position.y == len(self.game.arena.board)-2 and self.ball.position.x == 1:  # Bottom left corner
            y = 0 if roll_direction.get_sum() == 3 else -1
            x = 0 if roll_direction.get_sum() == 1 else 1
        elif self.ball.position.y == 1:  # Above
            y = 1
            x = -1 if roll_direction.get_sum() == 3 else 1
        elif self.ball.position.y == len(self.game.arena.board)-2:  # Below
            y = -1
            x = -1 if roll_direction.get_sum() == 1 else 1
        elif self.ball.position.x == 1:  # Right
            x = 1
            y = -1 if roll_direction.get_sum() == 3 else 1
        elif self.ball.position.x == len(self.game.arena.board[0])-2:  # Left
            x = -1
            y = -1 if roll_direction.get_sum() == 1 else 1

        for i in range(roll_distance.get_sum()):
            self.ball.move(x, y)
            if self.game.is_out_of_bounds(self.ball.position):
                ThrowIn(self.game, self.ball, Square(self.ball.position.x - x, self.ball.position.y - y))
                self.game.report(Outcome(OutcomeType.THROW_IN_OUT_OF_BOUNDS, position=self.ball.position,
                                         rolls=[roll_direction, roll_distance]))
                return True

        self.game.report(Outcome(OutcomeType.THROW_IN, position=self.ball.position, rolls=[roll_direction,  roll_distance]))

        # On player -> Catch
        player = self.game.get_player_at(self.ball.position)
        if player is not None:
            Catch(self.game, player, self.ball)
        else:
            Bounce(self.game, self.ball)

        return True

    def available_actions(self):
        return []


class Turnover(Procedure):

    def __init__(self, game):
        super().__init__(game)

    def step(self, action):
        self.game.report(Outcome(OutcomeType.TURNOVER, team=self.game.state.current_team))
        self.game.state.active_player = None
        EndTurn(self.game)
        return True

    def available_actions(self):
        return []


class Touchdown(Procedure):

    def __init__(self, game, player):
        super().__init__(game)
        self.game = game
        self.player = player

    def step(self, action):
        self.game.report(Outcome(OutcomeType.TOUCHDOWN, team=self.player.team, player=self.player))
        self.player.team.state.score += 1
        self.game.state.kicking_this_drive = self.player.team
        self.game.state.receiving_this_drive = self.game.get_opp_team(self.player.team)
        EndTurn(self.game, kickoff=True)

        return True

    def available_actions(self):
        return []


class TurnStunned(Procedure):

    def __init__(self, game, team):
        super().__init__(game)
        self.team = team

    def step(self, action):
        players = []
        for player in self.team.players:
            if player.state.stunned:
                player.state.stunned = False
                player.state.used = True
                players.append(player)
        self.game.report(Outcome(OutcomeType.STUNNED_TURNED))
        return True

    def available_actions(self):
        return []


class EndTurn(Procedure):

    def __init__(self, game, kickoff=False):
        super().__init__(game)
        self.kickoff = kickoff

    def step(self, action):

        # Remove all procs in the current turn - including the current turn proc.
        x = 0
        for i in reversed(range(self.game.state.stack.size())):
            x += 1
            if isinstance(self.game.state.stack.items[i], Turn):
                break
        for i in range(x):
            self.game.state.stack.pop()

        # Reset turn
        if self.game.state.current_team is not None:
            self.game.state.current_team.state.reset_turn()
            for player in self.game.state.current_team.players:
                player.state.reset_turn()


        # Add kickoff procedure - if there are more turns left
        if self.kickoff and self.game.get_opp_team(self.game.state.current_team).state.turn < self.game.config.rounds:
            Kickoff(self.game)

            # Setup in turn order from after scoring team
            #self.game.set_turn_order_after(self.game.state.current_team)

            Setup(self.game, self.game.state.receiving_this_drive)
            Setup(self.game, self.game.state.kicking_this_drive)

            for team in self.game.state.teams:
                PreKickOff(self.game, team)

            ClearBoard(self.game)

        self.game.state.current_team = None
        self.game.remove_clocks()

        return True

    def available_actions(self):
        return []


class Turn(Procedure):

    def __init__(self, game, team, half, turn, blitz=False, quick_snap=False):
        super().__init__(game)
        self.team = team
        self.half = half
        self.turn = turn
        self.blitz = blitz
        self.started = False
        self.quick_snap = quick_snap
        self.blitz_available = not quick_snap
        self.pass_available = not quick_snap
        self.handoff_available = not quick_snap
        self.foul_available = not quick_snap

    def setup(self):
        self.game.state.current_team = self.team

    def start_player_action(self, outcome_type, player_action_type, player):

        # Start action
        self.game.state.active_player = player
        player_action = PlayerAction(self.game, player, player_action_type, turn=self)
        self.game.report(Outcome(outcome_type, player=player))
        if player.has_skill(Skill.BONE_HEAD):
            Bonehead(self.game, player, player_action)
        if player.has_skill(Skill.REALLY_STUPID):
            ReallyStupid(self.game, player, player_action)
        if player.has_skill(Skill.WILD_ANIMAL):
            WildAnimal(self.game, player, player_action)

    def step(self, action):

        # Update state
        if not self.started:
            self.started = True
            self.game.add_primary_clock(self.team)

            if self.blitz:
                self.game.report(Outcome(OutcomeType.BLITZ_START, team=self.team))
            elif self.quick_snap:
                self.game.report(Outcome(OutcomeType.QUICK_SNAP_START, team=self.team))
            else:
                self.game.report(Outcome(OutcomeType.TURN_START, team=self.team, n=self.turn))
                self.team.state.turn = self.turn
                TurnStunned(self.game, self.team)
            return False

        # Handle End Turn action
        if action.action_type == ActionType.END_TURN:
            if self.blitz:
                self.game.report(Outcome(OutcomeType.END_OF_BLITZ, team=self.team))
            elif self.quick_snap:
                self.game.report(Outcome(OutcomeType.END_OF_QUICK_SNAP, team=self.team))
            else:
                self.game.report(Outcome(OutcomeType.END_OF_TURN, team=self.team))
            self.game.state.active_player = None
            EndTurn(self.game)
            
            return True

        # Start movement action
        if action.action_type == ActionType.START_MOVE:
            self.start_player_action(OutcomeType.MOVE_ACTION_STARTED, PlayerActionType.MOVE, action.player)
            return False

        # Start blitz action
        if action.action_type == ActionType.START_BLITZ:
            self.blitz_available = False
            self.start_player_action(OutcomeType.BLITZ_ACTION_STARTED, PlayerActionType.BLITZ, action.player)
            return False

        # Start foul action
        if action.action_type == ActionType.START_FOUL:
            self.foul_available = False
            self.start_player_action(OutcomeType.FOUL_ACTION_STARTED, PlayerActionType.FOUL, action.player)
            return False

        # Start block action
        if action.action_type == ActionType.START_BLOCK:
            self.start_player_action(OutcomeType.BLOCK_ACTION_STARTED, PlayerActionType.BLOCK, action.player)
            return False

        # Start pass action
        if action.action_type == ActionType.START_PASS:
            self.pass_available = False
            self.start_player_action(OutcomeType.PASS_ACTION_STARTED, PlayerActionType.PASS, action.player)
            return False

        # Start handoff action
        if action.action_type == ActionType.START_HANDOFF:
            self.handoff_available = False
            self.start_player_action(OutcomeType.HANDOFF_ACTION_STARTED, PlayerActionType.HANDOFF, action.player)
            return False

    def available_actions(self):
        if not self.started:
            return []
        move_players = []
        block_players = []
        blitz_players = []
        pass_players = []
        handoff_players = []
        foul_players = []
        for player in self.game.get_players_on_pitch(self.team):
            if self.blitz:
                if self.game.num_tackle_zones_in(player) > 0:
                    continue
            if not player.state.used:
                move_players.append(player)
                if self.blitz_available:
                    blitz_players.append(player)
                if self.pass_available:
                    pass_players.append(player)
                if self.handoff_available:
                    handoff_players.append(player)
                if self.foul_available:
                    foul_players.append(player)
                if not self.quick_snap and not self.blitz and player.state.up:
                    if self.game.get_adjacent_opponents(player, down=False):
                        block_players.append(player)

        actions = []
        if len(move_players) > 0:
            actions.append(ActionChoice(ActionType.START_MOVE, team=self.team, players=move_players))
        if len(block_players) > 0:
            actions.append(ActionChoice(ActionType.START_BLOCK, team=self.team, players=block_players))
        if len(blitz_players) > 0:
            actions.append(ActionChoice(ActionType.START_BLITZ, team=self.team, players=blitz_players))
        if len(pass_players) > 0:
            actions.append(ActionChoice(ActionType.START_PASS, team=self.team, players=pass_players))
        if len(handoff_players) > 0:
            actions.append(ActionChoice(ActionType.START_HANDOFF, team=self.team, players=handoff_players))
        if len(foul_players) > 0:
            actions.append(ActionChoice(ActionType.START_FOUL, team=self.team, players=foul_players))
        actions.append(ActionChoice(ActionType.END_TURN, team=self.team))

        return actions


class WeatherTable(Procedure):

    def __init__(self, game, kickoff=False):
        super().__init__(game)
        self.kickoff = kickoff

    def step(self, action):
        roll = DiceRoll([D6(self.game.rnd), D6(self.game.rnd)], roll_type=RollType.WEATHER_ROLL)
        if roll.get_sum() == 2:
            self.game.state.weather = WeatherType.SWELTERING_HEAT
            self.game.report(Outcome(OutcomeType.WEATHER_SWELTERING_HEAT, rolls=[roll]))
        if roll.get_sum() == 3:
            self.game.state.weather = WeatherType.VERY_SUNNY
            self.game.report(Outcome(OutcomeType.WEATHER_VERY_SUNNY, rolls=[roll]))
        if 4 <= roll.get_sum() <= 10:
            if self.kickoff and self.game.state.weather == WeatherType.NICE:
                self.game.state.gentle_gust = True
            self.game.state.weather = WeatherType.NICE
            self.game.report(Outcome(OutcomeType.WEATHER_NICE, rolls=[roll]))
        if roll.get_sum() == 11:
            self.game.state.weather = WeatherType.POURING_RAIN
            self.game.report(Outcome(OutcomeType.WEATHER_POURING_RAIN, rolls=[roll]))
        if roll.get_sum() == 12:
            self.game.state.weather = WeatherType.BLIZZARD
            self.game.report(Outcome(OutcomeType.WEATHER_BLIZZARD, rolls=[roll]))
        return True

    def available_actions(self):
        return []


class Negatrait(Procedure, metaclass=ABCMeta):
    def __init__(self, game, player, player_action):
        super().__init__(game)
        self.game = game
        self.player = player
        self.awaiting_reroll = False
        self.reroll_used = False
        self.rolled = False
        self.roll = None
        self.player_action = player_action

        self.roll_type = None
        self.skill = None
        self.success_outcome = None
        self.fail_outcome = None

    @abstractmethod
    def get_target(self):
        pass

    @abstractmethod
    def apply_fail_state(self):
        pass

    @abstractmethod
    def remove_fail_state(self):
        pass

    def trigger_failure(self):
        self.apply_fail_state()
        self.end_turn()

    def step(self, action):
        # If player hasn't rolled
        if not self.rolled:
            # Roll
            self.roll = DiceRoll([D6(self.game.rnd)], roll_type=self.roll_type)
            self.roll.target = self.get_target()
            self.rolled = True

            if self.roll.is_d6_success():
                # Success
                self.remove_fail_state()
                self.game.report(Outcome(self.success_outcome, player=self.player, rolls=[self.roll]))
                return True
            else:
                self.game.report(Outcome(self.fail_outcome, player=self.player, skill=self.skill, rolls=[self.roll]))
                # Check if reroll available
                if self.game.can_use_reroll(self.player.team):
                    self.awaiting_reroll = True
                    return False

                # Player forgets what to do
                self.trigger_failure()
                return True

        # If reroll used
        if self.awaiting_reroll:
            if action.action_type == ActionType.USE_REROLL:
                # Remove reroll and roll again - recursive call
                self.game.report(Outcome(OutcomeType.REROLL_USED, team=self.player.team))
                self.player.team.state.use_reroll()
                self.rolled = False
                return self.step(None)
            else:
                # Player forgets what to do
                self.trigger_failure()
                return True

    def end_turn(self):
        # mark player turn as over
        EndPlayerTurn(self.game, self.player)
        # don't let the intended action happen
        self.player_action.done = True

    def available_actions(self):
        if self.awaiting_reroll:
            return [ActionChoice(ActionType.USE_REROLL, team=self.game.state.current_team),
                    ActionChoice(ActionType.DONT_USE_REROLL, team=self.game.state.current_team)]

        return []


class Bonehead(Negatrait):
    def __init__(self, game, player, player_action):
        super().__init__(game, player, player_action)
        self.roll_type = RollType.BONE_HEAD_ROLL
        self.skill = Skill.BONE_HEAD
        self.success_outcome = OutcomeType.SUCCESSFUL_BONE_HEAD
        self.fail_outcome = OutcomeType.FAILED_BONE_HEAD

    def get_target(self):
        return 2

    def apply_fail_state(self):
        self.player.state.bone_headed = True

    def remove_fail_state(self):
        self.player.state.bone_headed = False


class ReallyStupid(Negatrait):
    def __init__(self, game, player, player_action):
        super().__init__(game, player, player_action)
        self.roll_type = RollType.REALLY_STUPID_ROLL
        self.skill = Skill.REALLY_STUPID
        self.success_outcome = OutcomeType.SUCCESSFUL_REALLY_STUPID
        self.fail_outcome = OutcomeType.FAILED_REALLY_STUPID

    def get_target(self):
        target = 4  # default target for player alone
        team_mates = self.game.get_adjacent_teammates(self.player)
        if team_mates:
            for team_mate in team_mates:
                if not team_mate.has_skill(Skill.REALLY_STUPID):
                    target = 2  # a non-stupid neighbour improves the odds
        return target

    def apply_fail_state(self):
        self.player.state.really_stupid = True

    def remove_fail_state(self):
        self.player.state.really_stupid = False


class WildAnimal(Negatrait):
    def __init__(self, game, player, player_action):
        super().__init__(game, player, player_action)
        self.roll_type = RollType.WILD_ANIMAL_ROLL
        self.skill = Skill.WILD_ANIMAL
        self.success_outcome = OutcomeType.SUCCESSFUL_WILD_ANIMAL
        self.fail_outcome = OutcomeType.FAILED_WILD_ANIMAL

    def get_target(self):
        target = 4
        if self.player_action.player_action_type is PlayerActionType.BLITZ or self.player_action.player_action_type is PlayerActionType.BLOCK:
            target = 2
        return target

    def apply_fail_state(self):
        self.player.state.wild_animal = True

    def remove_fail_state(self):
        self.player.state.wild_animal = False
