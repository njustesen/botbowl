"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains all the procedures. Procedures contain the core part of the rules in botbowl. Each procedure are
responsible for an isolated part of the game. Procedures are added to a stack, and the top-most procedure must finish
before other procedures are run. Procedures can add other procedures to the stack simply by instantiating procedures.
"""


from botbowl.core.pathfinding import Pathfinder
from botbowl.core.util import compare_object
from botbowl.core.model import *
from botbowl.core.table import *

import time
from abc import abstractmethod, ABCMeta


class Procedure(Reversible):

    def __init__(self, game, context=None, ignored_keys=None):
        ignored_keys = [] if ignored_keys is None else ignored_keys

        super().__init__(ignored_keys=["game"]+ignored_keys)
        self.game = game
        self.context = context
        self.done = False
        self.started = False
        self.game.state.stack.push(self)  # Note: The stack will call this object's set_logger

    def start(self):
        """
        Is called just before step() is called the first time. The Game class will set initialized to True.
        """
        pass

    def end(self):
        """
        Is called when the procedure is done, i.e. right after step() returns True.
        """
        pass

    def step(self, action):
        """
        Is called when this procedure is on the top of the stack until the procedure is done.
        :param action: an action performed by an agent - will be None if there are no available actions.
        :return: a bool indicating whether the procedure is done.
        """
        raise Exception("Must be implemented by subclass")

    def available_actions(self):
        """
        :return: a list of available actions. This can vary based on the state of the procedure.
        """
        return []

    def compare(self, other, path=""):
        return compare_object(self, other, path, ignored_keys={"game"}, ignored_types={Procedure})

    def __repr__(self):
        return f"{self.__class__.__name__}(done={self.done}, started={self.started})"


class Regeneration(Procedure):

    def __init__(self, game, player):
        super().__init__(game)
        self.player = player
        self.regenerates = False

    def step(self, action):
        if self.player.has_skill(Skill.REGENERATION):
            regen_roll = DiceRoll([D6(self.game.rnd)], target=4, roll_type=RollType.REGENERATION_ROLL)
            if regen_roll.is_d6_success():
                self.game.report(Outcome(OutcomeType.SUCCESSFUL_REGENERATION, player=self.player, rolls=[regen_roll]))
                # self.game.pitch_to_reserves(self.player)
                self.regenerates = True
            else:
                self.game.report(Outcome(OutcomeType.FAILED_REGENERATION, player=self.player, rolls=[regen_roll]))
        return True


class Apothecary(Procedure):

    def __init__(self, game, player, roll, outcome, inflictor, casualty=None, effect=None, decay=False):
        super().__init__(game)
        self.player = player
        self.inflictor = inflictor
        self.decay = decay
        self.waiting_apothecary = False
        self.roll_first = roll
        self.roll_second = roll
        self.outcome = outcome
        self.casualty_first = casualty
        self.effect_first = effect
        self.casualty_second = None
        self.effect_second = None
        self.regeneration = None
        self.effect = None
        self.casualty = None
        self.roll = None

    def start(self):
        if self.game.state.current_team != self.player.team:
            self.game.add_opp_clock(self.player.team)

    def step(self, action):

        if self.outcome == OutcomeType.KNOCKED_OUT:

            if action.action_type == ActionType.USE_APOTHECARY:

                # Player is moved to reserves
                self.player.team.state.apothecaries -= 1
                self.player.state.stunned = True
                self.player.place_prone()
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
                self.player.team.state.apothecaries -= 1
                self.game.report(Outcome(OutcomeType.CASUALTY_APOTHECARY, player=self.player, team=self.player.team,
                                         rolls=[self.roll_first, self.roll_second]))
                self.waiting_apothecary = True

                return False

            if action.action_type == ActionType.SELECT_FIRST_ROLL or ActionType.SELECT_SECOND_ROLL:

                self.effect = self.effect_first if action.action_type == ActionType.SELECT_FIRST_ROLL else self.effect_second
                self.casualty = self.casualty_first if action.action_type == ActionType.SELECT_FIRST_ROLL else self.casualty_second
                self.roll = self.roll_first if action.action_type == ActionType.SELECT_FIRST_ROLL else self.roll_second

                # Regeneration
                if self.player.has_skill(Skill.REGENERATION) and not self.regeneration:
                    self.regeneration = Regeneration(self.game, self.player)
                    return False

                if self.regeneration and self.regeneration.regenerates:
                    self.game.pitch_to_reserves(self.player)
                    return True

                # Apply casualty
                self.game.apply_casualty(self.player, self.inflictor, self.casualty, self.effect, self.roll)

                # Decay
                if self.decay:
                    Casualty(self.game, self.player)

        return True

    def available_actions(self):
        if self.regeneration:
            return []
        elif not self.waiting_apothecary and self.player.team.state.apothecaries > 0:
            return [ActionChoice(ActionType.USE_APOTHECARY, team=self.player.team),
                    ActionChoice(ActionType.DONT_USE_APOTHECARY, team=self.player.team)]
        elif self.waiting_apothecary:
            return [ActionChoice(ActionType.SELECT_FIRST_ROLL, team=self.player.team),
                    ActionChoice(ActionType.SELECT_SECOND_ROLL, team=self.player.team)]
        return []


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
        self.ejected = False

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
        if self.foul and roll.same():
            # Sneaky git
            if not self.inflictor.has_skill(Skill.SNEAKY_GIT) or armor_broken:
                self.ejected = True

        # Break armor - roll injury
        if armor_broken:
            Injury(self.game, self.player, self.inflictor, foul=self.foul,
                   mighty_blow_used=mighty_blow_used, dirty_player_used=dirty_player_used)
            self.game.report(Outcome(OutcomeType.ARMOR_BROKEN, player=self.player, opp_player=self.inflictor,
                                     rolls=[roll]))
        else:
            self.game.report(Outcome(OutcomeType.ARMOR_NOT_BROKEN, player=self.player, opp_player=self.inflictor,
                                     rolls=[roll]))

        return True

    def end(self):
        if self.ejected:
            Ejection(self.game, self.inflictor)


class Stab(Procedure):

    def __init__(self, game, attacker, defender, blitz=False, gfi=False):
        super().__init__(game)
        self.attacker = attacker
        self.defender = defender
        self.roll = None
        self.reroll = None
        self.blitz = blitz
        self.gfi = gfi

    def end(self):
        self.attacker.state.has_blocked = True

    def step(self, action):
        # GfI
        if self.gfi:
            self.gfi = False
            GFI(self.game, self.attacker, self.attacker.position)
            return False

        # Stab!
        self.roll = DiceRoll([D6(self.game.rnd), D6(self.game.rnd)], lowest_fail=False, highest_succeed=False)
        self.roll.target = self.defender.get_av()
        if self.attacker.has_skill(Skill.STAKES) and self.defender.team.race in \
                ['Khemri', 'Necromantic', 'Undead', 'Vampire']:
            self.game.report(Outcome(OutcomeType.SKILL_USED, skill=Skill.STAB, player=self.attacker))
            self.roll.modifiers += 1
        if self.roll.is_d6_success():
            KnockDown(self.game, player=self.defender, armor_roll=False, inflictor=self.attacker)
        self.game.report(Outcome(OutcomeType.SKILL_USED, skill=Skill.STAB, player=self.attacker, rolls=[self.roll]))
        return True


class FoulAppearance(Procedure):

    def __init__(self, game, attacker, defender):
        super().__init__(game)
        self.attacker = attacker
        self.defender = defender
        self.roll = None
        self.reroll = None
        self.revolted = False

    def step(self, action):
        if self.roll is None:
            self.roll = DiceRoll([D6(self.game.rnd)])
            self.roll.target = 2
            self.game.report(
                Outcome(OutcomeType.SKILL_USED, skill=Skill.FOUL_APPEARANCE, player=self.attacker, rolls=[self.roll]))
            if self.roll.is_d6_success():
                return True
            else:
                self.revolted = True
                self.reroll = Reroll(self.game, self.attacker, self)
                return False

        assert self.reroll is not None

        if self.reroll.use_reroll:
            self.reroll = None
            self.roll = None
            self.revolted = False
            return False
        return True


class Block(Procedure):

    def __init__(self, game, attacker, defender, blitz=False, frenzy_block=False, gfi=False):
        super().__init__(game)
        assert attacker != defender
        self.attacker = attacker
        self.defender = defender
        self.roll = None
        self.reroll = None
        self.blitz = blitz
        self.gfi = gfi
        self.frenzy_block = frenzy_block
        self.waiting_wrestle_attacker = False
        self.waiting_wrestle_defender = False
        self.waiting_juggernaut = False
        self.juggernaut_checked = False
        self.selected_die = None
        self.favor = None
        self.dauntless_roll = None
        self.dauntless_success = False
        self.frenzy_checked = False
        self.waiting_dump_off = False
        self.waiting_foul_appearance = False
        self.foul_appearance = None

    def start(self):
        self.waiting_dump_off = self.game.get_ball_carrier() == self.defender and self.defender.has_skill(Skill.DUMP_OFF) and not self.frenzy_block
        self.waiting_foul_appearance = self.defender.has_skill(Skill.FOUL_APPEARANCE)

    def end(self):
        self.attacker.state.has_blocked = True

    def step(self, action):

        # GfI
        if self.gfi:
            self.gfi = False
            GFI(self.game, self.attacker, self.attacker.position)
            return False

        # Foul appearance
        if self.waiting_foul_appearance:
            if not self.foul_appearance:
                self.foul_appearance = FoulAppearance(self.game, self.attacker, self.defender)
                return False
            elif self.foul_appearance.revolted:
                return True
            else:
                self.waiting_foul_appearance = False
                self.foul_appearance = None

        # Dump-off
        if self.waiting_dump_off:
            if action.action_type == ActionType.USE_SKILL:
                PassAction(self.game, self.defender, dump_off=True)
            self.waiting_dump_off = False
            return False

        # Frenzy check
        if self.frenzy_block and not self.frenzy_checked:
            # End frenzy block procedure if player is not on the field or not standing
            if self.defender.position is None or not self.defender.state.up:
                return True
            # If not adjacent to defender - e.g. in case of fend
            if self.defender not in self.game.get_adjacent_opponents(self.attacker):
                return True
            self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.attacker, skill=Skill.FRENZY))
            self.attacker.use_skill(Skill.FRENZY)
            self.frenzy_checked = True

        # Juggernaut
        if self.waiting_juggernaut:
            if action.action_type == ActionType.USE_SKILL:
                self.selected_die = BBDieResult.PUSH
                self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.attacker, skill=Skill.JUGGERNAUT))
            self.waiting_juggernaut = False
            self.juggernaut_checked = True
            return False

        # Attacker wrestle action
        if self.waiting_wrestle_attacker:
            if action.action_type == ActionType.USE_SKILL:
                KnockDown(self.game, self.attacker, armor_roll=False, injury_roll=False, inflictor=self.defender)
                KnockDown(self.game, self.defender, armor_roll=False, injury_roll=False, inflictor=self.attacker)
                self.waiting_wrestle_attacker = False
            elif self.waiting_wrestle_defender:
                return False
            else:
                return self.both_down()

        # Defender wrestle action
        if self.waiting_wrestle_defender:
            if action.action_type == ActionType.USE_SKILL:
                KnockDown(self.game, self.defender, armor_roll=False, injury_roll=False,
                          inflictor=self.attacker)
                KnockDown(self.game, self.attacker, armor_roll=False, injury_roll=False,
                          inflictor=self.defender)
                self.waiting_wrestle_defender = False
            else:
                return self.both_down()

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

            # check for re-roll
            self.reroll = Reroll(self.game, self.attacker, context=self)
            return False

        # If re-roll decision is made
        elif self.reroll is not None:
            if self.reroll.use_reroll:
                # reroll!
                self.roll = None
                self.reroll = None
                return False
            elif self.reroll.block_action:
                # If block die was selected in Reroll
                action = self.reroll.block_action
            else:
                # Add secondary clock if opponent has the favor
                if self.favor == self.defender.team:
                    self.game.add_secondary_clock(self.favor)
                self.reroll = None
                return False

        return self.handle_block_die(action.action_type)

    def handle_block_die(self, action_type):

        # Select die
        if action_type == ActionType.SELECT_ATTACKER_DOWN:
            self.selected_die = BBDieResult.ATTACKER_DOWN
        elif action_type == ActionType.SELECT_BOTH_DOWN:
            self.selected_die = BBDieResult.BOTH_DOWN
        elif action_type == ActionType.SELECT_PUSH:
            self.selected_die = BBDieResult.PUSH
        elif action_type == ActionType.SELECT_DEFENDER_STUMBLES:
            self.selected_die = BBDieResult.DEFENDER_STUMBLES
        elif action_type == ActionType.SELECT_DEFENDER_DOWN:
            self.selected_die = BBDieResult.DEFENDER_DOWN

        # Remove secondary clocks
        assert self.selected_die is not None

        BBDie.FixedRolls.append(self.selected_die)
        die = BBDie(None)
        self.game.report(Outcome(OutcomeType.ACTION_SELECT_DIE, team=self.favor, rolls=[DiceRoll([die])]))

        self.game.remove_secondary_clocks()

        # Dice result
        if self.selected_die == BBDieResult.ATTACKER_DOWN:
            # Turnover(self.game)
            KnockDown(self.game, self.attacker, inflictor=self.defender, turnover=True)
            return True

        elif self.selected_die == BBDieResult.BOTH_DOWN:

            if self.blitz and self.attacker.has_skill(Skill.JUGGERNAUT) and not self.juggernaut_checked:
                self.waiting_juggernaut = True
                return False

            # Attacker wrestle - Don't use if only one with Block skill
            if self.attacker.has_skill(Skill.WRESTLE) and not (self.attacker.has_skill(Skill.BLOCK) and not self.defender.has_skill(Skill.BLOCK)):
                self.waiting_wrestle_attacker = True

            # Defender wrestle - Don't use if only one with Block skill
            if self.defender.has_skill(Skill.WRESTLE) and not (self.defender.has_skill(Skill.BLOCK) and not self.attacker.has_skill(Skill.BLOCK)):
                if self.blitz and self.attacker.has_skill(Skill.JUGGERNAUT):
                    self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.attacker, skill=Skill.JUGGERNAUT))
                else:
                    self.waiting_wrestle_defender = True

            if self.waiting_wrestle_attacker or self.waiting_wrestle_defender:
                return False

            return self.both_down()

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

    def both_down(self):
        # Attacker down
        if not self.attacker.has_skill(Skill.BLOCK):
            KnockDown(self.game, self.attacker, inflictor=self.defender, turnover=True)
        else:
            self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.attacker, skill=Skill.BLOCK))
        # Defender down
        if not self.defender.has_skill(Skill.BLOCK):
            KnockDown(self.game, self.defender, inflictor=self.attacker)
        else:
            self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.defender, skill=Skill.BLOCK))
        return True

    def available_actions(self):
        if not self.attacker.state.up:
            return []
        if self.waiting_dump_off:
            return [
                ActionChoice(ActionType.USE_SKILL, skill=Skill.DUMP_OFF, players=[self.defender],
                             team=self.defender.team),
                ActionChoice(ActionType.DONT_USE_SKILL, skill=Skill.DUMP_OFF, players=[self.defender],
                             team=self.defender.team)
            ]
        if self.roll is None:
            return []
        if self.reroll is not None and self.reroll.done:
            return []
        if self.waiting_juggernaut:
            return [
                ActionChoice(ActionType.USE_SKILL, self.attacker.team, skill=Skill.JUGGERNAUT),
                ActionChoice(ActionType.DONT_USE_SKILL, self.attacker.team, skill=Skill.JUGGERNAUT)
            ]
        if self.waiting_wrestle_attacker:
            return [
                ActionChoice(ActionType.USE_SKILL, self.attacker.team, skill=Skill.WRESTLE),
                ActionChoice(ActionType.DONT_USE_SKILL, self.attacker.team, skill=Skill.WRESTLE)
            ]
        if self.waiting_wrestle_defender:
            return [
                ActionChoice(ActionType.USE_SKILL, self.defender.team, skill=Skill.WRESTLE),
                ActionChoice(ActionType.DONT_USE_SKILL, self.defender.team, skill=Skill.WRESTLE)
            ]
        assert self.selected_die is None
        disable_dice_pick = False
        if self.reroll is not None and self.reroll.use_reroll is None and self.favor != self.attacker.team:
            disable_dice_pick = True
        actions = []
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

    def __init__(self, game, piece, kick=False):
        super().__init__(game)
        self.piece = piece
        self.kick = kick

    def step(self, action):

        # Loose control
        if self.piece.is_catchable():
            self.piece.is_carried = False

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

        self.game.shove(self.piece, x, y)
        if type(self.piece) is Ball:
            self.game.report(Outcome(OutcomeType.BALL_BOUNCED, position=self.piece.position, rolls=[roll_scatter]))
        elif type(self.piece) is Player:
            self.game.report(Outcome(OutcomeType.PLAYER_BOUNCED, position=self.piece.position, rolls=[roll_scatter]))

        if self.kick:
            # Touchback
            if not self.game.is_team_side(self.piece.position, self.game.get_receiving_team()):
                Touchback(self.game, self.piece)
                self.game.report(Outcome(OutcomeType.TOUCHBACK, team=self.game.get_receiving_team(),
                                         rolls=[roll_scatter]))
                return True
        else:
            # Out of bounds
            if self.game.is_out_of_bounds(self.piece.position):
                if type(self.piece) is Ball:
                    ThrowIn(self.game, self.piece, self.game.get_square(self.piece.position.x - x, self.piece.position.y - y))
                    self.game.report(Outcome(OutcomeType.BALL_OUT_OF_BOUNDS))
                elif type(self.piece) is Player:
                    KnockDown(self.game, self.piece, in_crowd=True)
                    self.game.report(Outcome(OutcomeType.PLAYER_OUT_OF_BOUNDS, player=self.piece))
                return True

        # On player -> Catch
        player_at = self.game.get_player_at(self.piece.position)
        if player_at is not None:
            if self.piece.is_catchable():
                Catch(self.game, player_at, self.piece)
            if type(self.piece) is Ball:
                self.game.report(Outcome(OutcomeType.BALL_BOUNCE_PLAYER, position=self.piece.position, player=player_at))
            elif type(self.piece) is Player:
                self.game.report(Outcome(OutcomeType.PLAYER_BOUNCE_PLAYER, position=self.piece.position, player=player_at))
                Bounce(self.game, self.piece)
            return True

        # On empty square
        if type(self.piece) is Ball:
            self.game.report(Outcome(OutcomeType.BALL_BOUNCE_GROUND, position=self.piece.position))
        elif type(self.piece) is Player:
            self.game.report(Outcome(OutcomeType.PLAYER_BOUNCE_GROUND, player=self.piece, position=self.piece.position))
            Land(self.game, self.piece)
        return True

    def available_actions(self):
        return []


class Casualty(Procedure):

    def __init__(self, game, player, inflictor=None, decay=False, blood_lust=False, always_hungry=False):
        super().__init__(game)
        self.player = player
        self.inflictor = inflictor
        self.waiting_apothecary = False
        self.roll = None
        self.casualty = None
        self.effect = None
        self.decay = decay
        self.regeneration = None
        self.blood_lust = blood_lust
        self.always_hungry = always_hungry

    def step(self, action):

        if self.regeneration:
            if self.regeneration.regenerates:
                if self.player.position is not None:
                    self.game.pitch_to_reserves(self.player)
                return True
            self.regeneration = None

        if self.roll is None:
            self.roll = DiceRoll([D6(self.game.rnd), D8(self.game.rnd)], d68=True, roll_type=RollType.CASUALTY_ROLL)
            cas_rolls = None
            if self.blood_lust: 
                result = 38
            elif self.always_hungry:
                result = 61
            else:
                result = self.roll.get_sum()
                cas_rolls = [self.roll]
            
            n = min(61, max(38, result))
            self.casualty = CasualtyType(n)
            self.effect = Rules.casualty_effect[self.casualty]

            self.game.report(
                Outcome(OutcomeType.CASUALTY, player=self.player, opp_player=self.inflictor, team=self.player.team,
                        n=self.effect.name,
                        rolls=cas_rolls))

            if not self.always_hungry:
                if self.player.team.state.apothecaries > 0:
                    Apothecary(self.game, self.player, roll=self.roll, outcome=OutcomeType.CASUALTY,
                               casualty=self.casualty, effect=self.effect, inflictor=self.inflictor)
                    return True
                else:
                    # Regeneration
                    if self.player.has_skill(Skill.REGENERATION) and not self.regeneration:
                        self.regeneration = Regeneration(self.game, self.player)
                        return False

        # Apply casualty
        self.game.apply_casualty(self.player, self.inflictor, self.casualty, self.effect, self.roll)

        # Decay
        if self.decay:
            self.game.report(Outcome(OutcomeType.DECAYING, player=self.player))
            Casualty(self.game, self.player)

        return True


class Catch(Procedure):

    def __init__(self, game, player, piece, accurate=False, handoff=False, kick=False, passer=None):
        super().__init__(game)
        self.player = player
        self.piece = piece
        self.accurate = accurate
        self.handoff = handoff
        self.roll = None
        self.reroll = None
        self.kick = kick
        self.passer = passer
        self.diving = player.position != piece.position
        self.bomb_choice = None

    def start(self):
        if self.diving:
            self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.player, skill=Skill.DIVING_CATCH))

    def step(self, action):

        # You can decide not to catch a bomb
        if type(self.piece) is Bomb and self.bomb_choice is None:
            if action.action_type == ActionType.CATCH_BOMB:
                self.bomb_choice = True
                self.game.report(Outcome(OutcomeType.WILL_CATCH_BOMB, player=self.player))
            else:
                Explode(self.game, self.piece)
                self.game.report(Outcome(OutcomeType.WONT_CATCH_BOMB, player=self.player))
                return

        # Otherwise roll if player hasn't rolled
        if self.roll is None:

            # Can player even catch ball?
            if not self.player.can_catch():
                if type(self.piece) is Ball:
                    Bounce(self.game, self.piece)
                    self.game.report(Outcome(OutcomeType.FAILED_CATCH, player=self.player))
                elif type(self.piece) is Bomb:
                    Explode(self.game, self.piece)
                    self.game.report(Outcome(OutcomeType.FAILED_CATCH_BOMB, player=self.player))
                return True

            # Roll
            self.roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.AGILITY_ROLL)
            self.roll.modifiers = self.game.get_catch_modifiers(self.player, accurate=self.accurate, handoff=self.handoff)
            self.roll.target = Rules.agility_table[self.player.get_ag()]
            if self.roll.is_d6_success():
                if type(self.piece) is Ball:
                    self.game.report(Outcome(OutcomeType.SUCCESSFUL_CATCH, player=self.player, rolls=[self.roll]))
                elif type(self.piece) is Bomb:
                    self.game.report(Outcome(OutcomeType.SUCCESSFUL_CATCH_BOMB, player=self.player, rolls=[self.roll]))
                    ThrowBombAction(self.game, self.player)
                if self.diving:
                    self.game.move(self.piece, self.player.position)
                self.piece.is_carried = True
                if self.game.is_touchdown(self.player):
                    Touchdown(self.game, self.player)
                return True
            else:
                self.game.report(Outcome(OutcomeType.FAILED_CATCH, player=self.player, rolls=[self.roll]))
                # Check for re-roll
                self.reroll = Reroll(self.game, self.player, context=self)
                return False

        assert self.reroll is not None

        # If re-roll used
        if self.reroll.use_reroll:
            self.roll = None
            self.reroll = None
            return False
        else:
            # If diving tackle was used - bounce from player's square
            if self.diving:
                self.game.move(self.piece, self.player.position)
            Bounce(self.game, self.piece, kick=self.kick)

        return True

    def available_actions(self):
        if type(self.piece) is Bomb and self.bomb_choice is None:
            return [ActionChoice(ActionType.CATCH_BOMB, team=self.player.team), ActionChoice(ActionType.DONT_CATCH_BOMB, team=self.player.team)]
        return []


class Intercept(Procedure):

    def __init__(self, game, interceptor, ball, passer=None):
        super().__init__(game)
        self.interceptor = interceptor
        self.ball = ball
        self.roll = None
        self.reroll = None
        self.safe_throw_roll = None
        self.safe_throw_reroll = None
        self.passer = passer
        self.waiting_safe_throw = False

    def step(self, action):

        # If waiting for interception re-roll due to Safe Throw skill
        if self.waiting_safe_throw:
            # Make agility roll for passer
            self.safe_throw_roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.AGILITY_ROLL)
            self.safe_throw_roll.target = Rules.agility_table[self.passer.get_ag()]
            self.game.report(
                Outcome(OutcomeType.SKILL_USED, player=self.passer, skill=Skill.SAFE_THROW, rolls=[self.safe_throw_roll]))
            if self.safe_throw_roll.is_d6_success():
                self.game.report(Outcome(OutcomeType.FAILED_INTERCEPTION, player=self.interceptor))
                return True
            else:
                # Check for re-roll
                self.safe_throw_reroll = Reroll(self.game, self.passer, context=self)
                self.waiting_safe_throw = False
                return False
                # TODO: Check for safe-throw in Reroll

        # If safe throw re-roll decision made
        if self.safe_throw_reroll is not None:
            if self.safe_throw_reroll.use_reroll:
                self.safe_throw_reroll = None
                self.waiting_safe_throw = True
                self.roll = None
                return False
            else:
                self.ball.move_to(self.interceptor.position)
                self.ball.is_carried = True
                if self.game.is_touchdown(self.interceptor):
                    Touchdown(self.game, self.interceptor)
                else:
                    Turnover(self.game)
                return True

        # Otherwise roll if player hasn't rolled
        if self.roll is None:

            # Can player even catch ball?
            if self.interceptor.has_skill(Skill.NO_HANDS) or not self.interceptor.can_catch():
                return True

            # Roll
            self.roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.AGILITY_ROLL)
            self.roll.modifiers = self.game.get_catch_modifiers(self.interceptor, interception=True)
            self.roll.target = Rules.agility_table[self.interceptor.get_ag()]
            if self.roll.is_d6_success():
                self.game.report(Outcome(OutcomeType.INTERCEPTION, player=self.interceptor, rolls=[self.roll]))
                if self.passer is not None and self.passer.has_skill(Skill.SAFE_THROW):
                    if self.interceptor.has_skill(Skill.VERY_LONG_LEGS):
                        self.game.report(
                            Outcome(OutcomeType.SKILL_USED, player=self.passer, skill=Skill.SAFE_THROW))
                        self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.interceptor, skill=Skill.VERY_LONG_LEGS))
                    else:
                        self.waiting_safe_throw = True
                        return False
                self.ball.move_to(self.interceptor.position)
                self.ball.is_carried = True
                if self.game.is_touchdown(self.interceptor):
                    Touchdown(self.game, self.interceptor)
                else:
                    Turnover(self.game)
                return True

            else:

                self.game.report(Outcome(OutcomeType.FAILED_INTERCEPTION, player=self.interceptor, rolls=[self.roll]))

                # Check for re-roll
                self.reroll = Reroll(self.game, self.interceptor, context=self)
                return False

        assert self.reroll is not None

        # If re-roll used
        if self.reroll.use_reroll:
            self.roll = None
            self.reroll = None
            return False

        return True


class CoinTossFlip(Procedure):

    def __init__(self, game):
        super().__init__(game)

    def start(self):
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

    def start(self):
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
        self.awaiting_bribe = False

    def start(self):
        self.awaiting_bribe = self.player.team.state.bribes > 0

    def step(self, action):
        # Check if already ejected
        if self.player.state.ejected:
            return True
        if self.awaiting_bribe:
            if action.action_type == ActionType.USE_BRIBE:
                self.player.team.state.bribes -= 1
                self.game.report(Outcome(OutcomeType.BRIBE_USED, team=self.player.team, player=self.player))
                die = D6(self.game.rnd)
                roll = DiceRoll([die], roll_type=RollType.BRIBE_ROLL)
                roll.target = 2
                if roll.is_d6_success():
                    self.game.report(Outcome(OutcomeType.SUCCESSFUL_BRIBE, team=self.player.team, player=self.player, rolls=[roll]))
                    return True
                else:
                    self.game.report(Outcome(OutcomeType.FAILED_BRIBE, team=self.player.team, player=self.player, rolls=[roll]))
                    # Allow team to use another bribe
                    self.awaiting_bribe = self.player.team.state.bribes > 0
                    if self.awaiting_bribe:
                        return False

        # Turnover
        Turnover(self.game)

        # Drop ball
        ball = self.game.get_ball_at(self.player.position)
        if ball is not None:
            Bounce(self.game, ball)

        # Eject player
        self.game.pitch_to_dungeon(self.player)
        self.game.report(Outcome(OutcomeType.PLAYER_EJECTED, player=self.player))

        return True

    def available_actions(self):
        if self.awaiting_bribe:
            return [ActionChoice(ActionType.USE_BRIBE, team=self.player.team),
                    ActionChoice(ActionType.DONT_USE_BRIBE, team=self.player.team)]
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
            PreKickoff(self.game, self.game.get_kicking_team(self.half))
            PreKickoff(self.game, self.game.get_receiving_team(self.half))
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

    def __init__(self, game, player, inflictor=None, foul=False, mighty_blow_used=False, dirty_player_used=False, in_crowd=False, blood_lust=False):
        super().__init__(game)
        self.player = player
        self.inflictor = inflictor
        self.injury_rolled = False
        self.foul = foul
        self.mighty_blow_used = mighty_blow_used
        self.dirty_player_used = dirty_player_used
        self.ejected = False
        self.in_crowd = in_crowd
        self.blood_lust = blood_lust

    def step(self, action):

        # TODO: Necromancer

        # Roll
        roll = DiceRoll([D6(self.game.rnd), D6(self.game.rnd)], roll_type=RollType.INJURY_ROLL)
        self.injury_rolled = True

        # Skill modifiers
        thick_skull = -1 if self.player.has_skill(Skill.THICK_SKULL) else 0
        stunty = 1 if self.player.has_skill(Skill.STUNTY) else 0
        mighty_blow = 0
        dirty_player = 0
        niggling = self.player.num_niggling_injuries()
        if self.inflictor is not None:
            dirty_player = 1 if self.inflictor.has_skill(Skill.DIRTY_PLAYER) and not self.dirty_player_used and \
                                self.foul else 0
            mighty_blow = 1 if self.inflictor.has_skill(Skill.MIGHTY_BLOW) and not self.mighty_blow_used and not \
                self.foul else 0

        # EJECTION
        if self.foul and roll.same() and not self.inflictor.has_skill(Skill.SNEAKY_GIT):
            self.ejected = True

        # CASUALTY
        roll.modifiers = stunty + mighty_blow + dirty_player + niggling
        if roll.get_result() >= 10:
            roll.modifiers = stunty + mighty_blow + dirty_player
            self.game.report(Outcome(OutcomeType.CASUALTY, player=self.player, opp_player=self.inflictor, rolls=[roll]))
            Casualty(self.game, self.player, inflictor=self.inflictor, decay=self.player.has_skill(Skill.DECAY),
                     blood_lust=self.blood_lust)
            return True

        # KOD
        roll.modifiers = thick_skull + stunty + mighty_blow + dirty_player + niggling
        if roll.get_result() >= 8:
            KnockOut(self.game, self.player, roll=roll, inflictor=self.inflictor)
            return True

        # STUNNED
        roll.modifiers = thick_skull + stunty + mighty_blow + dirty_player + niggling
        if self.player.has_skill(Skill.BALL_AND_CHAIN):
            KnockOut(self.game, self.player, roll=roll, inflictor=self.inflictor)
        else:
            self.game.report(Outcome(OutcomeType.STUNNED, player=self.player, opp_player=self.inflictor,
                                     rolls=[roll]))
            if self.in_crowd:
                self.game.pitch_to_reserves(self.player)
            else:
                self.player.place_prone()
                self.player.state.stunned = True

        return True

    def end(self):
        if self.ejected:
            Ejection(self.game, self.inflictor)

    def available_actions(self):
        return []


class Interception(Procedure):

    def __init__(self, game, team, ball, interceptors, passer):
        super().__init__(game)
        self.team = team
        self.passer = passer
        self.ball = ball
        self.interceptors = interceptors

    def start(self):
        self.game.add_secondary_clock(self.team)

    def step(self, action):
        if action.action_type == ActionType.SELECT_PLAYER:
            Intercept(self.game, action.player, self.ball, passer=self.passer)
        self.game.remove_secondary_clocks()
        return True

    def available_actions(self):
        rolls = []
        for interceptor in self.interceptors:
            modifier = self.game.get_catch_modifiers(interceptor, interception=True)
            rolls.append([max(2, min(6, Rules.agility_table[interceptor.get_ag()] - modifier))])
        return [ActionChoice(ActionType.SELECT_PLAYER, team=self.team,
                             players=self.interceptors, rolls=rolls),
                ActionChoice(ActionType.SELECT_NONE, team=self.team)]


class Touchback(Procedure):

    def __init__(self, game, ball):
        super().__init__(game)
        self.ball = ball

    def start(self):
        self.game.add_secondary_clock(self.game.state.receiving_this_drive)
        self.players_on_pitch_standing = self.game.get_players_on_pitch(self.game.state.receiving_this_drive, up=True)

    def step(self, action):
        player = None
        if len(self.players_on_pitch_standing) == 0:
            position = self.game.rnd.choice(self.game.get_team_side(self.game.state.receiving_this_drive))
            self.ball.is_carried = False
        else:
            player = action.player
            position = player.position
            self.ball.is_carried = True
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
        catcher = self.game.get_catcher(self.ball.position)
        if catcher is None:
            Bounce(self.game, self.ball, kick=True)
            self.game.report(Outcome(OutcomeType.BALL_HIT_GROUND, position=self.ball.position))
            return True

        Catch(self.game, catcher, self.ball, kick=True)
        self.game.report(Outcome(OutcomeType.BALL_HIT_PLAYER, position=self.ball.position, player=catcher))

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

    def start(self):
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
        if len(self.available_players) == 0 or action.action_type == ActionType.SELECT_NONE:
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
            return [ActionChoice(ActionType.SELECT_PLAYER, team=self.receiving_team, players=self.available_players),
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
        for team in self.game.state.teams:
            roll = DiceRoll([D3(self.game.rnd)], roll_type=RollType.THROW_A_ROCK_ROLL)
            roll.modifiers = team.state.fame
            rolls.append(roll.get_result())
            self.game.report(Outcome(OutcomeType.THROW_A_ROCK_ROLL, team=team, rolls=[roll]))

        for i in range(len(self.game.state.teams)):
            team = self.game.state.teams[i]
            if rolls[i] == np.min(rolls):
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
                self.player.place_prone()
                self.player.state.stunned = True
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
                 in_crowd=False, modifiers_opp=0, turnover=False, blood_lust=False):
        super().__init__(game)
        self.player = player
        self.armor_roll = armor_roll
        self.injury_roll = injury_roll
        self.modifiers = modifiers
        self.modifiers_opp = modifiers_opp
        self.inflictor = inflictor
        self.in_crowd = in_crowd
        self.turnover = turnover
        self.blood_lust = blood_lust

    def step(self, action):

        # Knock down player
        self.player.place_prone()
        if self.player == self.game.get_active_player():
            self.player.state.used = True

        self.game.report(Outcome(OutcomeType.KNOCKED_DOWN, player=self.player, opp_player=self.inflictor))

        # Turnover
        if self.turnover:
            Turnover(self.game)

        # Check fumble - does not handle throw-in
        ball = self.game.get_ball_at(self.player.position)
        if ball is not None:
            if not self.player.position.out_of_bounds:
                Bounce(self.game, ball)
            self.game.report(Outcome(OutcomeType.FUMBLE, player=self.player, opp_player=self.inflictor))

        # If armor roll should be made. Injury is also nested in armor.
        if self.injury_roll and not self.armor_roll:
            Injury(self.game, self.player, inflictor=self.inflictor if not self.in_crowd else None,
                   in_crowd=self.in_crowd, blood_lust=self.blood_lust)
        elif self.armor_roll:
            Armor(self.game, self.player, modifiers=self.modifiers, inflictor=self.inflictor)

        return True


class KnockOut(Procedure):

    def __init__(self, game, player, roll, inflictor=None):
        super().__init__(game)
        self.player = player
        self.inflictor = inflictor
        self.roll = roll

    def step(self, action):
        if self.player.team.state.apothecaries > 0:
            Apothecary(self.game, self.player, roll=self.roll, outcome=OutcomeType.KNOCKED_OUT,
                       inflictor=self.inflictor)
            return True
        else:
            # Knock out player
            self.game.pitch_to_kod(self.player)
            self.game.report(Outcome(OutcomeType.KNOCKED_OUT, rolls=[self.roll], player=self.player,
                                     opp_player=self.inflictor))

        return True


class Leap(Procedure):

    def __init__(self, game, player, position, gfis):
        super().__init__(game)
        self.player = player
        self.position = position
        self.roll = None
        self.reroll = None
        for i in range(gfis):
            GFI(game, player, position)

    def start(self):
        assert Skill.LEAP not in self.player.state.used_skills
        self.player.state.used_skills.add(Skill.LEAP)

    def step(self, action):

        if self.roll is None:

            # Agility roll
            self.roll = DiceRoll([D6(self.game.rnd)])
            self.roll.target = Rules.agility_table[self.player.get_ag()]
            self.roll.modifiers = self.game.get_leap_modifiers(self.player)

            if self.roll.is_d6_success():
                self.game.move(self.player, self.position)
                self.game.report(Outcome(OutcomeType.SUCCESSFUL_LEAP, player=self.player, rolls=[self.roll]))
                # Check if player moved onto the ball
                ball = self.game.get_ball_at(self.player.position)
                if ball is not None and not ball.is_carried:
                    # Attempt to pick up the ball - unless no hands
                    if self.player.has_skill(Skill.NO_HANDS):
                        Bounce(self.game, ball)
                    else:
                        Pickup(self.game, ball, self.player)
                elif self.game.has_ball(self.player) and self.game.is_touchdown(self.player):
                    # Touchdown if player had the ball with him/her
                    Touchdown(self.game, self.player)
                return True
            else:
                self.game.report(Outcome(OutcomeType.FAILED_LEAP, player=self.player, rolls=[self.roll]))
                self.reroll = Reroll(self.game, self.player, self)
                return False

        if self.reroll is not None:
            if self.reroll.use_reroll:
                self.roll = None
                self.reroll = None
                return False
            else:
                self.game.report(Outcome(OutcomeType.FAILED_LEAP))
                self.game.move(self.player, self.position)
                KnockDown(self.game, self.player, turnover=True)
        return True


class Shadowing(Procedure):

    def __init__(self, game, player, position, shadowers):
        super().__init__(game)
        self.player = player
        self.position = position
        self.shadowers = shadowers
        self.shadower = None
        self.team = self.game.get_opp_team(player.team)
        self.roll = None
        self.reroll = None

    def start(self):
        able_players = []
        for player in self.shadowers:
            if player.has_tackle_zone() and not player.state.taken_root:
                able_players.append(player)
        self.shadowers = able_players
        self.shadower = self.shadowers[0] if self.shadowers else None
        self.game.add_secondary_clock(self.team)

    def step(self, action):
        if not self.shadowers:
            return True

        if self.roll is None and action.action_type == ActionType.USE_SKILL:
            self.roll = DiceRoll(dice=[D6(self.game.rnd), D6(self.game.rnd)], roll_type=RollType.SHADOWING_ROLL, highest_succeed=False)
            self.roll.target = 7
            self.roll.modifiers = self.player.get_ma() - self.shadower.get_ma()
            self.roll.target_higher = False
            self.roll.target_lower = True
            self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.shadower, skill=Skill.SHADOWING, rolls=[self.roll]))
            if self.roll.is_d6_success():
                self.game.move(self.shadower, self.position)
                return True
            else:
                self.reroll = Reroll(self.game, self.shadower, context=self)
                return False

        if self.reroll is not None:
            if self.reroll.use_reroll:
                self.roll = None
                self.reroll = None
                return False
            return True

        # Dont use shadow
        self.shadowers.remove(self.shadower)
        self.game.remove_secondary_clocks()
        if self.shadowers:
            self.shadower = self.shadowers[0]
            self.game.add_secondary_clock(self.team)
            return False

        return True

    def end(self):
        self.game.remove_secondary_clocks()

    def available_actions(self):
        if self.shadower is None or self.reroll is not None:
            return []
        return [
            ActionChoice(ActionType.USE_SKILL, team=self.team, players=[self.shadower], skill=Skill.SHADOWING),
            ActionChoice(ActionType.DONT_USE_SKILL, team=self.team, players=[self.shadower], skill=Skill.SHADOWING)
        ]


class Tentacles(Procedure):

    def __init__(self, game, player, position, tentacler, move_proc):
        super().__init__(game)
        self.player = player
        self.position = position
        self.tentacler = tentacler
        self.move_proc = move_proc
        self.roll = None
        self.reroll = None

    def step(self, action):

        if self.roll is None:
            self.roll = DiceRoll([D6(self.game.rnd), D6(self.game.rnd)])
            self.roll.target = 5
            self.roll.modifiers = self.player.get_st() - self.tentacler.get_st()
            if self.roll.is_d6_success():
                return True
            else:
                self.reroll = Reroll(self.game, self.player, self)
                return False

        assert self.reroll is not None

        if self.reroll.use_reroll:
            self.roll = None
            self.reroll = None
            return False

        self.move_proc.tentacles_used = True
        return True


class Move(Procedure):

    def __init__(self, game, player, position, gfi, dodge):
        super().__init__(game)
        self.player = player
        self.position = position
        self.dodge = dodge
        self.dodge_proc = None
        self.gfi = gfi
        self.tentaclers = [tentacler for tentacler in self.game.get_adjacent_opponents(self.player, skill=Skill.TENTACLES, standing=True) if tentacler.has_tackle_zone()]
        self.tentacles_used = False

    def step(self, action):

        if self.tentacles_used:
            return True

        if self.tentaclers:
            next_tentacler = self.tentaclers[0]
            Tentacles(self.game, self.player, self.position, next_tentacler, self)
            self.tentaclers.remove(next_tentacler)
            return False

        if self.dodge:
            self.dodge_proc = Dodge(self.game, self.player, self.position)

        if self.gfi:
            GFI(self.game, self.player, self.position)

        if self.dodge or self.gfi:
            self.dodge = False
            self.gfi = False
            return False

        # If this is reached then Dodge and GFI was passed

        # Shadowing
        shadowers = self.game.get_adjacent_opponents(self.player, down=False, skill=Skill.SHADOWING)
        position = self.player.position

        self.game.move(self.player, self.position)
        if self.dodge_proc is not None and self.dodge_proc.diving_tackler:
            self.game.move(self.dodge_proc.diving_tackler, self.dodge_proc.from_position)
            KnockDown(self.game, self.dodge_proc.diving_tackler, armor_roll=False, injury_roll=False, turnover=False)

        # Check if player moved onto the ball
        ball = self.game.get_ball_at(self.player.position)
        if ball is not None and not ball.is_carried:

            # Attempt to pick up the ball - unless no hands
            if self.player.has_skill(Skill.NO_HANDS):
                Bounce(self.game, ball)
            else:
                Pickup(self.game, ball, self.player)

        elif self.game.has_ball(self.player) and self.game.is_touchdown(self.player):

            # Touchdown if player had the ball with him/her
            Touchdown(self.game, self.player)
            shadowers = None

        # Shadowing
        if shadowers:
            Shadowing(self.game, self.player, position, shadowers)

        return True


class GFI(Procedure):

    def __init__(self, game, player, position):
        super().__init__(game)
        self.player = player
        self.position = position
        self.reroll = None
        self.roll = None

    def step(self, action):

        # If player hasn't rolled
        if self.roll is None:

            # Roll
            self.roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.GFI_ROLL)
            self.roll.target = 2
            self.roll.modifiers = 1 if self.game.state.weather == WeatherType.BLIZZARD else 0

            if self.roll.is_d6_success():

                # Success
                self.game.report(Outcome(OutcomeType.SUCCESSFUL_GFI, player=self.player, position=self.position, rolls=[self.roll]))
                return True

            else:

                # Fail
                self.game.report(Outcome(OutcomeType.FAILED_GFI, player=self.player, position=self.position, rolls=[self.roll]))

                # Check re-roll
                self.reroll = Reroll(self.game, self.player, context=self)
                return False

        assert self.reroll is not None

        # If reroll used
        if self.reroll.use_reroll:
            self.roll = None
            self.reroll = None
            return False

        # Shadowing
        shadowers = self.game.get_adjacent_opponents(self.player, down=False, skill=Skill.SHADOWING)
        shadow_position = self.player.position

        # Player trips
        if not self.player.position == self.position:
            self.game.move(self.player, self.position)

        KnockDown(self.game, self.player, turnover=True)

        if shadowers:
            Shadowing(self.game, self.player, shadow_position, shadowers)

        return True


class Dodge(Procedure):

    def __init__(self, game, player, position):
        super().__init__(game)
        self.player = player
        self.position = position
        self.reroll = None
        self.roll = None
        self.waiting_break_tackle = False
        self.break_tackle_target = None
        self.diving_tacklers = []
        self.diving_tackler = None
        self.from_position = self.game.state.pitch.squares[self.player.position.y][self.player.position.x]
        self.diving_tacklers = self.game.get_adjacent_opponents(self.player, down=False, skill=Skill.DIVING_TACKLE)
        if self.diving_tacklers:
            self.game.add_secondary_clock(self.diving_tacklers[0].team)

    def step(self, action):

        # Diving tackle
        if self.diving_tacklers:
            if action.action_type == ActionType.USE_SKILL:
                self.diving_tackler = self.diving_tacklers[0]
                self.diving_tacklers.clear()
            else:
                self.diving_tacklers.pop(0)
            # Time management
            if self.diving_tacklers:
                self.game.add_secondary_clock(self.diving_tacklers[0].team)
            else:
                self.game.remove_secondary_clocks()
            return False

        # Break tackle
        if self.waiting_break_tackle:
            if action.action_type == ActionType.DONT_USE_SKILL:
                # Check for re-roll
                self.reroll = Reroll(self.game, self.player, context=self)
                self.waiting_break_tackle = False
                return False
            elif action.action_type == ActionType.USE_SKILL:
                # Success
                self.player.use_skill(Skill.BREAK_TACKLE)
                self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.player, skill=Skill.BREAK_TACKLE))
                self.roll.target = self.break_tackle_target
                self.game.report(Outcome(OutcomeType.SUCCESSFUL_DODGE, player=self.player, position=self.position,
                                         rolls=[self.roll]))
                return True

        # If player hasn't rolled
        if self.roll is None:

            # TODO: Allow player to select if shadowing and diving tackle
            # TODO: Put diving tackle, shadowing proc or tentacles on stack
            # TODO: Auto-use other skills

            # Roll
            self.roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.AGILITY_ROLL)
            self.roll.modifiers = self.game.get_dodge_modifiers(self.player, self.position, include_diving_tackle=(self.diving_tackler is not None))

            agility_target = Rules.agility_table[self.player.get_ag()]
            self.roll.target = agility_target
            if self.player.can_use_skill(Skill.BREAK_TACKLE) and self.player.get_st() > self.player.get_ag():
                self.break_tackle_target = Rules.agility_table[self.player.get_st()]

            if self.roll.is_d6_success():
                # Success
                self.game.report(Outcome(OutcomeType.SUCCESSFUL_DODGE, player=self.player, position=self.position, rolls=[self.roll]))
                return True
            else:
                # Fail
                self.game.report(Outcome(OutcomeType.FAILED_DODGE, player=self.player, position=self.position, rolls=[self.roll]))
                if self.player.can_use_skill(Skill.BREAK_TACKLE) and self.player.get_st() > self.player.get_ag() and self.roll.get_result() >= self.break_tackle_target:
                    self.waiting_break_tackle = True
                    return False

                # Check for re-roll
                self.reroll = Reroll(self.game, self.player, context=self)
                return False

        assert self.reroll is not None

        # If dodge failed and re-roll decision made
        if self.reroll.use_reroll:
            self.roll = None
            self.reroll = None
            return False

        # Shadowing
        shadowers = self.game.get_adjacent_opponents(self.player, down=False, skill=Skill.SHADOWING)
        shadow_position = self.player.position

        # If dodge failed and no re-roll -> Player trips
        self.game.move(self.player, self.position)
        if self.diving_tackler:
            self.game.move(self.diving_tackler, self.from_position)
            KnockDown(self.game, self.diving_tackler, armor_roll=False, injury_roll=False, turnover=False)
        KnockDown(self.game, self.player, turnover=True)

        if shadowers:
            Shadowing(self.game, self.player, shadow_position, shadowers)

        return True

    def available_actions(self):
        if self.diving_tacklers:
            return [ActionChoice(ActionType.USE_SKILL, skill=Skill.DIVING_TACKLE, team=self.diving_tacklers[0].team, players=[self.diving_tacklers[0]]),
                    ActionChoice(ActionType.DONT_USE_SKILL, skill=Skill.DIVING_TACKLE, team=self.diving_tacklers[0].team, players=[self.diving_tacklers[0]])]
        if self.waiting_break_tackle:
            return [ActionChoice(ActionType.USE_SKILL, skill=Skill.BREAK_TACKLE, team=self.player.team, players=[self.player]),
                    ActionChoice(ActionType.DONT_USE_SKILL, skill=Skill.BREAK_TACKLE, team=self.player.team, players=[self.player])]
        return []


class TurnoverIfPossessionLost(Procedure):

    def __init__(self, game, ball):
        super().__init__(game)
        self.ball = ball

    def step(self, action):
        player_at = self.game.get_player_at(self.ball.position)
        if player_at is None or player_at.team != self.game.state.current_team:
            Turnover(self.game)
        return True


class Handoff(Procedure):

    def __init__(self, game, ball, player, pos_to, catcher):
        super().__init__(game)
        self.ball = ball
        self.player = player
        self.pos_to = pos_to
        self.catcher = catcher
        self.eat_thrall = None
        
    def step(self, action):
        if self.player.state.blood_lust and self.eat_thrall is None:
            self.eat_thrall = EatThrall(self.game, self.player)
            return False 
        
        if self.eat_thrall is not None and self.eat_thrall.failed: 
            return True 
        
        self.ball.move_to(self.pos_to)
        TurnoverIfPossessionLost(self.game, self.ball)
        self.ball.move_to(self.catcher.position)
        Catch(self.game, self.catcher, self.ball, handoff=True)
        self.game.report(Outcome(OutcomeType.HANDOFF, player=self.player, opp_player=self.catcher))
        return True


class Explode(Procedure):

    def __init__(self, game, bomb):
        super().__init__(game)
        self.bomb = bomb
        self.player = self.game.get_player_at(bomb.position)

    def step(self, action):
        self.game.report(Outcome(OutcomeType.BOMB_EXPLODED, position=self.bomb.position, player=self.player))
        if self.player:
            self.game.report(Outcome(OutcomeType.BOMB_HIT, position=self.bomb.position, player=self.player))
            KnockDown(self.game, self.player)
        position = self.bomb.position
        self.game.remove(self.bomb)
        for player in self.game.get_adjacent_players(position):
            die = D6(self.game.rnd)
            roll = DiceRoll(die, RollType.BOMB_ROLL, target=4)
            if roll.is_d6_success():
                self.game.report(Outcome(OutcomeType.BOMB_HIT, position=self.bomb.position, player=self.player, rolls=[roll]))
                KnockDown(self.game, player)
        return True


class Land(Procedure):

    def __init__(self, game, player):
        super().__init__(game)
        self.player = player
        self.roll = None
        self.reroll = None

    def start(self):
        self.game.put_down(self.player)

    def step(self, action):
        if self.roll is None:
            self.roll = DiceRoll([D6(self.game.rnd)], RollType.LAND_ROLL)
            self.roll.target = Rules.agility_table[self.player.get_ag()]
            self.roll.modifiers = self.game.get_landing_modifiers(self.player)
            if self.roll.is_d6_success():
                self.game.report(Outcome(OutcomeType.SUCCESSFUL_LAND, position=self.player.position, player=self.player, rolls=[self.roll]))
                if self.game.is_touchdown(self.player):
                    Touchdown(self.game, self.player)
                # bounce ball if you land on the ball; this will be different in BB2020
                elif self.game.get_ball_position() == self.player.position and not self.game.has_ball(self.player):
                    Bounce(self.game, self.game.get_ball())
                return True
            else:
                self.game.report(Outcome(OutcomeType.FAILED_LAND, position=self.player.position, player=self.player, rolls=[self.roll]))
                self.reroll = Reroll(self.game, self.player, context=self)
                return False

        # If re-roll used
        if self.reroll.use_reroll:
            self.reroll = None
            self.roll = None
            return False

        # Failed landing
        KnockDown(self.game, self.player, turnover=self.game.has_ball(self.player))
        return True


class PassAttempt(Procedure):

    def __init__(self, game, passer, piece, position, pass_distance, dump_off=False):
        super().__init__(game)
        self.piece = piece
        self.passer = passer
        self.position = position
        self.pass_distance = pass_distance
        self.roll = None
        self.reroll = None
        self.fumble = False
        self.interception_tried = False
        self.dump_off = dump_off
        self.catcher = game.get_player_at(position)
        self.eat_thrall = None
        self.ttm = type(piece) == Player
        self.turnover = False
        self.safe_throw_used = False

    def start(self):
        if self.dump_off:
            self.game.report(Outcome(OutcomeType.SKILL_USED, skill=Skill.DUMP_OFF, player=self.passer))
        self.catcher = self.game.get_catcher(self.position)

    def end(self):
        if self.turnover:
            Turnover(self.game)

    def step(self, action):

        if self.passer.state.blood_lust and self.eat_thrall is None:
            self.eat_thrall = EatThrall(self.game, self.passer)
            return False 
        
        if self.eat_thrall is not None and self.eat_thrall.failed: 
            return True
        
        # Otherwise roll if player hasn't rolled
        if self.roll is None:

            if self.piece.is_catchable():
                self.piece.is_carried = False

            # Check for interception
            if not self.interception_tried and self.piece.is_catchable():
                interceptors = self.game.get_interceptors(self.passer.position, self.position, team=self.game.get_opp_team(self.passer.team))
                if len(interceptors) > 0:
                    Interception(self.game, interceptors[0].team, self.piece, interceptors, self.passer)
                    self.interception_tried = True
                    return False

            # Roll
            self.roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.AGILITY_ROLL)
            self.roll.target = Rules.agility_table[self.passer.get_ag()]
            self.roll.modifiers = self.game.get_pass_modifiers(self.passer, self.pass_distance, ttm=self.ttm)
            result = self.roll.get_sum()
            mod_result = result + self.roll.modifiers

            if result == 6 or (result != 1 and mod_result >= self.roll.target):
                # Treat as inaccurate if throw team-mate
                if type(self.piece) is Player:
                    self.game.report(Outcome(OutcomeType.INACCURATE_PASS, player=self.passer, rolls=[self.roll]))
                    self._inaccurate_pass()
                    return True
                # Accurate pass
                self.game.report(Outcome(OutcomeType.ACCURATE_PASS, player=self.passer, rolls=[self.roll]))
                self.game.move(self.piece, self.position)
                if type(self.piece) is Ball and not self.dump_off:
                    TurnoverIfPossessionLost(self.game, self.piece)
                if self.piece.is_catchable() and self.catcher is not None:
                    Catch(self.game, self.catcher, self.piece, accurate=True)
                elif type(self.piece) is Ball:
                    Bounce(self.game, self.piece)
                elif type(self.piece) is Bomb:
                    Explode(self.game, self.piece)
                elif type(self.piece) is Player:
                    if self.catcher is not None:
                        Bounce(self.game, self.piece)
                        KnockDown(self.game, self.catcher, self.piece)
                    else:
                        Land(self.game, self.piece)
                return True

            safe_throw = self.passer.has_skill(Skill.SAFE_THROW) and type(self.piece) == Ball

            if result == 1 or (mod_result <= 1 and not safe_throw):
                # Fumble
                self.fumble = True
                self.game.report(Outcome(OutcomeType.FUMBLE, player=self.passer, rolls=[self.roll]))
            elif mod_result <= 1 and safe_throw:
                # Inaccurate pass - Safe Throw
                self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.passer, skill=Skill.SAFE_THROW))
                self.safe_throw_used = True
            else:
                # Inaccurate pass
                self.game.report(Outcome(OutcomeType.INACCURATE_PASS, player=self.passer, rolls=[self.roll]))

            # Treat as inaccurate pass as a success if throw team-mate
            if not self.fumble and type(self.piece) is Player:
                self._inaccurate_pass()
                return True

            # Check if re-roll available
            self.reroll = Reroll(self.game, self.passer, context=self)
            return False

        # If re-roll used
        assert self.reroll is not None

        if self.reroll.use_reroll:
            self.reroll = None
            self.roll = None
            self.fumble = False
            return False

        if self.safe_throw_used:
            return True

        if self.fumble:
            self._fumble()
        else:
            self._inaccurate_pass()
        return True

    def _fumble(self):
        if type(self.piece) is Ball:
            if not self.dump_off:
                Turnover(self.game)
            Bounce(self.game, self.piece)
        elif type(self.piece) is Player:
            Land(self.game, self.piece)
        elif type(self.piece) is Bomb:
            Explode(self.game, self.piece)

    def _inaccurate_pass(self):
        if type(self.piece) is Ball and not self.dump_off:
            TurnoverIfPossessionLost(self.game, self.piece)
        self.game.move(self.piece, self.position)
        Scatter(self.game, self.piece, is_pass=True)


class Pickup(Procedure):

    def __init__(self, game, ball, player):
        super().__init__(game)
        self.ball = ball
        self.player = player
        self.roll = None
        self.reroll = None

    def step(self, action):

        # Otherwise roll if player hasn't rolled
        if self.roll is None:

            self.roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.AGILITY_ROLL)
            self.roll.target = Rules.agility_table[self.player.get_ag()]
            self.roll.modifiers = self.game.get_pickup_modifiers(self.player, self.ball.position)

            # Roll
            if self.roll.is_d6_success():
                self.game.report(Outcome(OutcomeType.SUCCESSFUL_PICKUP, player=self.player, rolls=[self.roll]))
                self.ball.is_carried = True
                if self.game.is_touchdown(self.player):
                    Touchdown(self.game, self.player)
                return True
            else:
                self.game.report(Outcome(OutcomeType.FAILED_PICKUP, player=self.player, position=self.player.position,
                                         rolls=[self.roll]))
                # Check re-roll
                self.reroll = Reroll(self.game, self.player, context=self)
                return False

        assert self.reroll is not None

        # If re-roll is used
        if self.reroll.use_reroll:
            self.roll = None
            self.reroll = None
            return False

        Turnover(self.game)
        Bounce(self.game, self.ball)
        return True


class StandUp(Procedure):

    def __init__(self, game, player):
        super().__init__(game)
        self.player = player
        self.sroll_required = False
        self.moves_required = 3
        self.reroll = None
        self.roll = None
        self.moves_required = 0 if self.player.has_skill(Skill.JUMP_UP) else 3
        self.roll_required = self.player.get_ma() < self.moves_required

    def step(self, action):
        if self.roll_required and self.roll is None:
            modifier = self.game.get_stand_up_modifier(self.player)
            self.roll = DiceRoll([D6(self.game.rnd)], target=4, modifiers=modifier, roll_type=RollType.STAND_UP_ROLL)
            if self.roll.is_d6_success():
                self.player.state.up = True
                self.game.report(Outcome(OutcomeType.STAND_UP, rolls=[self.roll], player=self.player))
                return True
            else:
                self.game.report(Outcome(OutcomeType.FAILED_STAND_UP, rolls=[self.roll], player=self.player))
                self.reroll = Reroll(self.game, self.player, self)
                return False
        elif not self.roll_required:
            self.game.report(Outcome(OutcomeType.STAND_UP, player=self.player))
            self.player.state.up = True
            return True

        assert self.reroll is not None

        if self.reroll.use_reroll:
            self.reroll = None
            self.roll = None
            return False
        else:
            self.player.place_prone()
            self.game.report(Outcome(OutcomeType.FAILED_STAND_UP, rolls=[self.roll], player=self.player))
            EndPlayerTurn(self.game, self.player)
        return True


class PlaceBall(Procedure):

    def __init__(self, game, ball):
        super().__init__(game)
        self.ball = ball
        self.aa = None

    def start(self):
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
        
        if self.player.state.blood_lust: 
            EatThrall(self.game, self.player)
            return False

        self.game.purge_stack_until(Turn, inclusive=False)
        self.player.state.used = True
        self.player.state.moves = 0
        self.game.report(Outcome(OutcomeType.END_PLAYER_TURN, player=self.player))
        self.game.state.active_player = None
        self.player.state.squares_moved.clear()
        self.game.state.player_action_type = None
        self.player.state.has_blocked = False
        return True


class JumpUpToBlock(Procedure):

    def __init__(self, game, player):
        super().__init__(game)
        self.player = player
        self.roll = None
        self.reroll = None

    def step(self, action):

        if self.roll is None:
            self.roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.AGILITY_ROLL)
            self.roll.target = Rules.agility_table[self.player.get_ag()]
            self.roll.modifiers = 2
            if self.roll.is_d6_success():
                self.player.state.up = True
                self.game.report(Outcome(OutcomeType.SKILL_USED, skill=Skill.JUMP_UP, player=self.player, rolls=[self.roll]))
                return True
            else:
                self.game.report(Outcome(OutcomeType.FAILED_JUMP_UP, player=self.player, rolls=[self.roll]))
                self.reroll = Reroll(self.game, self.player, self)
                return False

        assert self.reroll is not None

        if self.reroll.use_reroll:
            self.roll = None
            self.reroll = None
            return False
        EndPlayerTurn(self.game, self.player)
        return True


class EscapeBeingEaten(Procedure):

    def __init__(self, game, hungry_player, delicious_player):
        super().__init__(game)
        self.hungry_player = hungry_player
        self.delicious_player = delicious_player
        self.roll = None
        self.reroll = None

    def step(self, action):

        if self.roll is None:
            self.roll = DiceRoll([D6(self.game.rnd)])
            self.roll.target = 2
            if self.roll.is_d6_success():
                self.game.report(Outcome(OutcomeType.SUCCESSFUL_ESCAPE_BEING_EATEN, player=self.hungry_player,
                                         opp_player=self.delicious_player, rolls=[self.roll]))
                Land(self.game, self.delicious_player)
                return True
            self.game.report(Outcome(OutcomeType.FAILED_ESCAPE_BEING_EATEN, player=self.hungry_player,
                                     opp_player=self.delicious_player, rolls=[self.roll]))
            self.game.report(Outcome(OutcomeType.EATEN_DURING_ALWAYS_HUNGRY, player=self.delicious_player, opp_player=self.hungry_player))
            self.reroll = Reroll(self.game, self.delicious_player, self)
            return False

        if self.reroll.use_reroll:
            self.roll = None
            self.reroll = None
            return False
        else:
            EndPlayerTurn(self.game, self.hungry_player)
            ball = self.game.get_ball_at(self.delicious_player.position)
            if ball is not None:
                TurnoverIfPossessionLost(self.game, ball)
                Bounce(self.game, ball)
            Casualty(self.game, player=self.delicious_player, inflictor=self.hungry_player, always_hungry=True)
        return True


class AlwaysHungry(Procedure):

    def __init__(self, game, hungry_player, delicious_player):
        super().__init__(game)
        self.hungry_player = hungry_player
        self.delicious_player = delicious_player
        self.roll = None
        self.reroll = None

    def step(self, action):

        if self.roll is None:
            self.roll = DiceRoll([D6(self.game.rnd)])
            self.roll.target = 2
            if self.roll.is_d6_success():
                self.game.report(Outcome(OutcomeType.SUCCESSFUL_ALWAYS_HUNGRY, player=self.hungry_player,
                                         opp_player=self.delicious_player, rolls=[self.roll]))
                return True
            self.game.report(Outcome(OutcomeType.FAILED_ALWAYS_HUNGRY, player=self.hungry_player,
                                     opp_player=self.delicious_player, rolls=[self.roll]))
            self.reroll = Reroll(self.game, self.hungry_player, self)
            return False

        assert self.reroll is not None

        if self.reroll.use_reroll:
            self.roll = None
            self.reroll = None
            return False
        else:
            EndPlayerTurn(self.game, self.hungry_player)
            EscapeBeingEaten(self.game, self.hungry_player, self.delicious_player)
        return True


class UndoPlayerAction(Procedure):

    def __init__(self, game, player):
        super().__init__(game)
        self.game = game
        self.player = player

    def step(self, action):
        self.game.state.player_action_type = None
        self.game.state.active_player = None
        return True


class MoveAction(Procedure):

    def __init__(self, game, player, player_action_type=PlayerActionType.MOVE):
        super().__init__(game)
        self.player = player
        self.player_action_type = player_action_type
        self.paths = {}
        self.steps = None
        self.orig_action_type = None
        self.can_undo = False

    def start(self):
        if self.player_action_type == PlayerActionType.MOVE:
            self.game.report(Outcome(OutcomeType.MOVE_ACTION_STARTED, player=self.player))
            self.game.state.player_action_type = PlayerActionType.MOVE
        self.can_undo = self.game.get_team_agent(self.player.team).human and not self.player.state.failed_nega_trait_this_turn

    def step(self, action):

        # Follow the selected path
        if action is None:
            pos = self.steps.pop(0)
            if pos == self.player.position:
                action = Action(ActionType.STAND_UP)
            else:
                action = Action(ActionType.MOVE, position=pos)

        if action.action_type == ActionType.UNDO:
            UndoPlayerAction(self.game, self.player)
            return True

        self.can_undo = False

        if len(self.player.state.squares_moved) == 0:
            self.player.state.squares_moved.append(self.player.position)

        if action.action_type == ActionType.END_PLAYER_TURN:
            EndPlayerTurn(self.game, self.player)
            return True

        if action.action_type == ActionType.STAND_UP:

            # if this is the last step in a path
            if self.steps is not None and len(self.steps) == 0:
                self.steps = None

            self._stand_up()
            return False

        if action.action_type == ActionType.LEAP:

            distance = self.player.position.distance(action.position)
            gfis = (self.player.state.moves + distance) - self.player.get_ma() if self.player.state.moves + distance > self.player.get_ma() else 0
            Leap(self.game, self.player, action.position, gfis)
            self.player.state.squares_moved.append(action.position)
            self.player.state.moves += distance
            return False

        if action.action_type == ActionType.HYPNOTIC_GAZE and self.player_action_type == PlayerActionType.MOVE:

            EndPlayerTurn(self.game, self.player)
            target_opponent = self.game.get_player_at(action.position)
            HypnoticGaze(self.game, self.player, target_opponent)
            return True

        if not self.game.is_quick_snap() and self.game.config.pathfinding_enabled:

            # Start new path?
            if self.steps is None:
                self.steps = self.paths[action.position].steps
                self.orig_action_type = action.action_type
                return False

            # if this is the last step in a path
            if self.steps is not None and len(self.steps) == 0:
                self.steps = None

        self._move(action.position)

        return False

    def _stand_up(self):
        moves = 3
        if self.player.has_skill(Skill.JUMP_UP):
            moves = 0
            self.game.report(Outcome(OutcomeType.SKILL_USED, skill=Skill.JUMP_UP, player=self.player))
        StandUp(self.game, self.player)
        self.player.state.moves += min(self.player.get_ma(), moves)
        for i in range(moves):
            self.player.state.squares_moved.append(self.player.position)

    def _move(self, position):
        gfi = self.player.state.moves + 1 > self.player.get_ma()
        if self.game.is_quick_snap() or self.player.has_skill(Skill.BALL_AND_CHAIN):
            dodge = False
        else:
            dodge = self.game.num_tackle_zones_in(self.player) > 0
        Move(self.game, self.player, position, gfi, dodge)
        self.player.state.squares_moved.append(position)
        self.player.state.moves += 1

    def _get_actions_from_paths(self, paths):
        paths_by_type = {
            ActionType.MOVE: [],
            ActionType.BLOCK: [],
            ActionType.STAB: [],
            ActionType.HANDOFF: [],
            ActionType.FOUL: []
        }
        for path in paths:
            if path.block_dice:
                paths_by_type[ActionType.BLOCK].append(path)
                if self.player.has_skill(Skill.STAB):
                    paths_by_type[ActionType.STAB].append(path)
            elif path.handoff_roll:
                paths_by_type[ActionType.HANDOFF].append(path)
            elif path.foul_roll:
                paths_by_type[ActionType.FOUL].append(path)
            else:
                paths_by_type[ActionType.MOVE].append(path)
        actions = []
        for action_type, action_paths in paths_by_type.items():
            if len(action_paths) > 0:
                positions = [path.get_last_step() for path in action_paths]
                block_dice = [path.block_dice for path in action_paths]
                rolls = []
                if action_type == ActionType.HANDOFF:
                    rolls = [path.handoff_roll for path in action_paths]
                elif action_type == ActionType.FOUL:
                    rolls = [path.foul_roll for path in action_paths]
                actions.append(ActionChoice(action_type=action_type, team=self.player.team, positions=positions,
                                            paths=action_paths, block_dice=block_dice, rolls=rolls))
        return actions

    def available_actions(self):
        if self.steps is not None and len(self.steps) > 0:
            return []  # No actions -> Continue path
        actions = []
        if self.game.is_quick_snap() or not self.game.config.pathfinding_enabled:
            actions = self.game.get_adjacent_move_actions(self.player)
        else:
            can_block = self.player_action_type == PlayerActionType.BLITZ and not self.player.state.has_blocked
            can_foul = self.player_action_type == PlayerActionType.FOUL
            can_handoff = self.player_action_type == PlayerActionType.HANDOFF and self.game.has_ball(self.player)
            pathfinder = Pathfinder(self.game,
                                    self.player,
                                    directly_to_adjacent=self.game.config.pathfinding_directly_to_adjacent,
                                    can_block=can_block,
                                    can_handoff=can_handoff,
                                    can_foul=can_foul,
                                    trr=False)
            paths = pathfinder.get_paths()
            self.paths = {path.get_last_step(): path for path in paths}
            actions += self._get_actions_from_paths(paths)
        actions += self.game.get_stand_up_actions(self.player)
        actions += self.game.get_leap_actions(self.player)
        if self.player.has_skill(Skill.HYPNOTIC_GAZE) and self.player_action_type == PlayerActionType.MOVE:
            actions += self.game.get_hypnotic_gaze_actions(self.player)
        actions.append(ActionChoice(ActionType.END_PLAYER_TURN, team=self.player.team))

        if self.can_undo:
            actions.append(ActionChoice(ActionType.UNDO, team=self.player.team))

        return actions


class HandoffAction(MoveAction):

    def __init__(self, game, player):
        super().__init__(game, player, player_action_type=PlayerActionType.HANDOFF)

    def start(self):
        self.game.use_handoff_action()
        self.game.report(Outcome(OutcomeType.HANDOFF_ACTION_STARTED, player=self.player))
        self.game.state.player_action_type = PlayerActionType.HANDOFF

    def step(self, action):
        # Continue moving along path
        if action is None:
            player_at = self.game.get_player_at(self.steps[0])
            if player_at is None or (player_at == self.player and not player_at.state.up):
                return super().step(action)
            action = Action(ActionType.HANDOFF, position=self.steps.pop(0))
            self.steps = None

        if action.action_type is ActionType.UNDO:
            self.game.unuse_handoff_action()
            return super().step(action)

        self.can_undo = False

        if action.action_type is ActionType.HANDOFF:
            if self.player.position.distance(action.position) == 1:
                EndPlayerTurn(self.game, self.player)
                ball = self.game.get_ball_at(self.player.position)
                assert ball is not None
                Handoff(self.game, ball, self.player, action.position,
                        self.game.get_player_at(action.position))
                return True

        # It's a move action
        return super().step(action)

    def available_actions(self):
        # If MoveAction is following a path -> continue
        if self.steps is not None:
            return []
        # Path-assisted moves
        actions = super().available_actions()
        # If pathfinding not enabled add them
        if not self.game.config.pathfinding_enabled and self.game.has_ball(self.player):
            actions.extend(self.game.get_handoff_actions(self.player))
        return actions


class PassAction(MoveAction):

    def __init__(self, game, player, dump_off=False):
        super().__init__(game, player, player_action_type=PlayerActionType.PASS)
        self.dump_off = dump_off
        self.picked_up_teammate = None

    def start(self):
        self.game.use_pass_action()
        self.game.report(Outcome(OutcomeType.PASS_ACTION_STARTED, player=self.player))
        self.game.state.player_action_type = PlayerActionType.PASS

    def step(self, action):

        # Continue moving along path
        if action is None:
            return super().step(action)

        if action.action_type is ActionType.UNDO:
            self.game.unuse_pass_action()
            return super().step(action)

        self.can_undo = False

        if action.action_type is ActionType.PASS:
            pass_distance = self.game.get_pass_distance(self.player.position, action.position)
            if not self.dump_off:
                self.game.use_pass_action()
                EndPlayerTurn(self.game, self.player)
            piece = self.game.get_ball_at(self.player.position)
            PassAttempt(self.game, self.player, piece, action.position, pass_distance, dump_off=self.dump_off)
            return True

        if action.action_type == ActionType.PICKUP_TEAM_MATE:
            self.picked_up_teammate = self.game.get_player_at(action.position)
            self.game.lift(self.picked_up_teammate)
            if self.player.has_skill(Skill.ALWAYS_HUNGRY):
                AlwaysHungry(self.game, self.player, self.picked_up_teammate)
            return False

        if action.action_type == ActionType.THROW_TEAM_MATE:
            pass_distance = self.game.get_pass_distance(self.player.position, action.position)
            EndPlayerTurn(self.game, self.player)
            PassAttempt(self.game, self.player, self.picked_up_teammate, action.position, pass_distance)
            return True

        return super().step(action)

    def available_actions(self):
        # If MoveAction is following a path -> continue
        if self.steps is not None:
            return []

        actions = []
        if not self.picked_up_teammate:
            actions.extend(super().available_actions())

        if self.player.has_skill(Skill.THROW_TEAM_MATE) and self.player.state.up:
            if self.picked_up_teammate is None:
                # Pickup team-mate
                ptm_actions = self.game.get_pickup_teammate_actions(self.player)
                actions.extend(ptm_actions)
            else:
                # Throw team-mate actions
                pass_actions = self.game.get_pass_actions(self.player, self.picked_up_teammate)
                actions.extend(pass_actions)

        # Pass actions
        if self.game.has_ball(self.player) and self.picked_up_teammate is None:
            piece = self.game.get_ball_at(self.player.position)
            pass_actions = self.game.get_pass_actions(self.player, piece, dump_off=self.dump_off)
            actions.extend(pass_actions)

        return actions


class ThrowBombAction(Procedure):

    def __init__(self, game, player):
        super().__init__(game)
        self.player = player
        self.game.put_bomb(Bomb(self.player.position))
        self.can_undo = False

    def start(self):
        self.game.report(Outcome(OutcomeType.THROW_BOMB_ACTION_STARTED, player=self.player))
        self.game.state.player_action_type = PlayerActionType.THROW_BOMB
        self.can_undo = self.game.get_team_agent(self.player.team).human

    def step(self, action):
        if action.action_type == ActionType.THROW_BOMB:
            pass_distance = self.game.get_pass_distance(self.player.position, action.position)
            EndPlayerTurn(self.game, self.player)
            PassAttempt(self.game, self.player, self.game.get_bomb(), action.position, pass_distance)
            return True
        EndPlayerTurn(self.game, self.player)
        return True

    def end(self):
        if self.game.get_bomb() is not None:
            self.game.remove_bomb()

    def available_actions(self):
        actions = self.game.get_pass_actions(self.player, self.game.get_bomb())
        actions.append(ActionChoice(ActionType.END_PLAYER_TURN, team=self.player.team))
        return actions


class FoulAction(MoveAction):

    def __init__(self, game, player):
        super().__init__(game, player, player_action_type=PlayerActionType.FOUL)

    def start(self):
        self.game.use_foul_action()
        self.game.report(Outcome(OutcomeType.FOUL_ACTION_STARTED, player=self.player))
        self.game.state.player_action_type = PlayerActionType.FOUL

    def step(self, action):
        # Continue moving along path
        if action is None:
            player_at = self.game.get_player_at(self.steps[0])
            if player_at is None or (player_at == self.player and not player_at.state.up):
                return super().step(action)
            # Last step is a foul action
            action = Action(ActionType.FOUL, position=self.steps.pop(0))
            self.steps = None

        if action.action_type is ActionType.UNDO:
            self.game.unuse_foul_action()
            return super().step(action)

        self.can_undo = False

        if action.action_type == ActionType.FOUL:
            if self.player.position.distance(action.position) == 1:
                player_to = self.game.get_player_at(action.position)
                EndPlayerTurn(self.game, self.player)
                Foul(self.game, self.player, player_to)
                return True

        # It's a move action
        return super().step(action)

    def available_actions(self):
        # If MoveAction is following a path -> continue
        if self.steps is not None:
            return []
        # Will include foul actions if pathfinding is enabled
        actions = super().available_actions()
        # Else find them for this square
        if not self.game.config.pathfinding_enabled:
            actions += self.game.get_foul_actions(self.player)
        return actions


class BlockAction(Procedure):

    def __init__(self, game, player):
        super().__init__(game)
        self.player = player
        self.can_undo = False

    def start(self):
        self.game.report(Outcome(OutcomeType.BLOCK_ACTION_STARTED, player=self.player))
        self.game.state.player_action_type = PlayerActionType.BLOCK
        self.can_undo = self.game.get_team_agent(self.player.team).human

    def step(self, action):

        if action.action_type == ActionType.END_PLAYER_TURN:
            EndPlayerTurn(self.game, self.player)
            return True

        if action.action_type == ActionType.UNDO:
            UndoPlayerAction(self.game, self.player)
            return True

        self.can_undo = False

        defender = self.game.get_player_at(action.position)

        EndPlayerTurn(self.game, self.player)

        # Stab
        if action.action_type == ActionType.STAB:

            Stab(self.game, self.player, defender)

        if action.action_type == ActionType.BLOCK:

            if self.player.has_skill(Skill.FRENZY):
                # TODO: Second block can also be a stab?
                Block(self.game, self.player, defender, frenzy_block=True)

            # Regular block
            Block(self.game, self.player, defender)

        if not self.player.state.up:
            JumpUpToBlock(self.game, self.player)

        return True

    def available_actions(self):
        actions = self.game.get_block_actions(self.player, blitz=False)
        actions.append(ActionChoice(ActionType.END_PLAYER_TURN, team=self.player.team))
        if self.can_undo:
            actions.append(ActionChoice(ActionType.UNDO, team=self.player.team))
        return actions


class BlitzAction(MoveAction):

    def __init__(self, game, player):
        super().__init__(game, player, player_action_type=PlayerActionType.BLITZ)
        self.player = player

    def start(self):
        self.game.report(Outcome(OutcomeType.BLITZ_ACTION_STARTED, player=self.player))
        self.game.state.player_action_type = PlayerActionType.BLITZ
        self.game.use_blitz_action()

    def step(self, action):

        # Continue moving along path
        if action is None:
            player_at = self.game.get_player_at(self.steps[0])
            if player_at is None or (player_at == self.player and not player_at.state.up):
                return super().step(action)
            # Last path step is a block or stab action
            action = Action(self.orig_action_type, position=self.steps.pop(0))
            self.steps = None

        if action.action_type is ActionType.UNDO:
            self.game.unuse_blitz_action()
            return super().step(action)

        self.can_undo = False

        if action.action_type == ActionType.END_PLAYER_TURN:
            EndPlayerTurn(self.game, self.player)
            return True

        if action.action_type == ActionType.BLOCK or action.action_type == ActionType.STAB:
            if self.player.position.distance(action.position) == 1:
                defender = self.game.get_player_at(action.position)
                move_needed = 1
                if not self.player.state.up:
                    move_needed += 0 if self.player.has_skill(Skill.JUMP_UP) else 3
                else:
                    self.player.state.moves += move_needed

                gfis = 3 if self.player.has_skill(Skill.SPRINT) else 2
                gfi = self.player.state.moves + move_needed > self.player.get_ma()
                gfi_frenzy = self.player.state.moves + move_needed + 1 > self.player.get_ma()
                frenzy_allowed = self.player.state.moves + move_needed + 1 < self.player.get_ma() + gfis

                if action.action_type == ActionType.BLOCK:
                    # Frenzy second block
                    if self.player.has_skill(Skill.FRENZY) and frenzy_allowed:
                        # TODO: Option to block or stab: add second block inside block?
                        Block(self.game, self.player, defender, blitz=True, gfi=gfi_frenzy, frenzy_block=True)
                    # Regular block
                    Block(self.game, self.player, defender, blitz=True, gfi=gfi)
                    if not self.player.state.up:
                        self._stand_up()
                    return False
                elif action.action_type == ActionType.STAB:
                    EndPlayerTurn(self.game, self.player)
                    Stab(self.game, self.player, defender, blitz=True, gfi=gfi)
                    if not self.player.state.up:
                        self._stand_up()
                    return True

        # It's a move action
        return super().step(action)

    def available_actions(self):
        # If MoveAction is following a path -> continue
        if self.steps is not None:
            return []
        # Move actions will include block moves if pathfinding is enabled
        actions = super().available_actions()
        # Else find them for this square
        if not self.game.config.pathfinding_enabled:
            actions += self.game.get_block_actions(self.player, blitz=True)
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


class PreKickoff(Procedure):

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
        elif self.attacker.has_skill(Skill.FRENZY) or (action and action.position == self.pos_to):
            shadowers = self.game.get_adjacent_opponents(self.attacker, down=False, skill=Skill.SHADOWING)
            position = self.attacker.position
            self.game.move(self.attacker, self.pos_to)
            self.attacker.state.squares_moved.append(self.pos_to)
            self.game.report(Outcome(OutcomeType.FOLLOW_UP, position=self.pos_to, player=self.attacker))
            # Touchdown?
            if self.game.has_ball(self.attacker) and self.game.is_touchdown(self.attacker):
                Touchdown(self.game, self.attacker)
                shadowers = None
            # Shadowing
            if shadowers:
                Shadowing(self.game, self.attacker, position, shadowers)
        else:
            self.attacker.state.squares_moved.append(self.attacker.position)
        return True

    def available_actions(self):
        if self.must_follow_up(self.attacker) or not self.can_follow_up(self.attacker, self.defender):
            return []
        else:
            return [ActionChoice(ActionType.FOLLOW_UP, team=self.attacker.team,
                                 positions=[self.attacker.position, self.pos_to])]

    def can_follow_up(self, attacker, defender):
        return not defender.has_skill(Skill.FEND) and not attacker.state.taken_root

    def must_follow_up(self, attacker):
        return attacker.has_skill(Skill.FRENZY)


class Push(Procedure):

    def __init__(self, game, pusher, player, knock_down=False, blitz=False, chain=False):
        super().__init__(game)
        assert pusher != player
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
        self.selector = game.get_active_player()
        self.crowd = False
        self.strip_ball_condition = self.pusher.has_skill(Skill.STRIP_BALL) and not self.player.has_skill(Skill.SURE_HANDS) and not self.knock_down and not self.chain and self.game.has_ball(self.player) 
        
    def step(self, action):

        # When proceeding pushes are over, move player(s)
        if self.waiting_for_move:

            # Stop secondary clocks
            self.game.remove_secondary_clocks()

            # Move pushed player
            self.game.move(self.player, self.push_to)

            # Ball
            if self.game.has_ball(self.player):
                if self.player.position.out_of_bounds:
                    ball = self.game.get_ball_at(self.player.position)
                    ThrowIn(self.game, ball, position=self.follow_to)
                elif self.strip_ball_condition: 
                    self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.pusher, skill=Skill.STRIP_BALL))
                    self.game.report(Outcome(OutcomeType.FUMBLE, player=self.player, opp_player=self.pusher))
                    Bounce(self.game, self.game.get_ball_at(self.player.position) )
                elif self.game.is_touchdown(self.player):
                    Touchdown(self.game, self.player)
                    
            elif self.game.get_ball_at(self.push_to) is not None:
                Bounce(self.game, self.game.get_ball_at(self.push_to))

            # Knock down
            if self.knock_down or self.crowd:
                KnockDown(self.game, self.player, in_crowd=self.crowd, armor_roll=not self.crowd)

            # Follow up
            if not self.chain:
                FollowUp(self.game, self.pusher, self.player, self.follow_to)

            return True

        # Taken root players cannot be pushed
        if self.player.state.taken_root:
            self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.player, skill=Skill.TAKE_ROOT))
            if self.knock_down:
                KnockDown(self.game, self.player, in_crowd=False, armor_roll=True)
            elif self.strip_ball_condition: 
                self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.pusher, skill=Skill.STRIP_BALL))
                self.game.report(Outcome(OutcomeType.FUMBLE, player=self.player, opp_player=self.pusher))
                Bounce(self.game, self.game.get_ball_at(self.player.position) )
            return True

        # Use stand firm
        if self.waiting_stand_firm:
            if action.action_type == ActionType.USE_SKILL:
                self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.player, skill=Skill.STAND_FIRM))
                if self.knock_down:
                    KnockDown(self.game, self.player, in_crowd=False, armor_roll=True)
                elif self.strip_ball_condition: 
                    self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.pusher, skill=Skill.STRIP_BALL))
                    self.game.report(Outcome(OutcomeType.FUMBLE, player=self.player, opp_player=self.pusher))
                    Bounce(self.game, self.game.get_ball_at(self.player.position) )
                return True
            else:
                self.waiting_stand_firm = False
                self.stand_firm_used = True

        # Stand firm
        if self.player.has_skill(Skill.STAND_FIRM) and not self.stand_firm_used:
            if self.pusher.has_skill(Skill.JUGGERNAUT) and self.blitz:
                self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.pusher, skill=Skill.JUGGERNAUT))
            else:
                self.waiting_stand_firm = True
                return False

        # Get possible squares
        if self.squares is None:
            # Sidestep and grab cancels out eachother - otherwise let the grabber or sidestepper select adjacent square
            if self.player.has_skill(Skill.SIDE_STEP) and (self.chain or not self.pusher.has_skill(Skill.GRAB)):
                self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.player, skill=Skill.SIDE_STEP))
                self.squares = self.game.get_adjacent_squares(self.player.position, occupied=False)
                self.selector = self.player
                if len(self.squares) > 0:
                    if self.player.team != self.game.state.current_team:
                        self.game.add_secondary_clock(self.player.team)
            elif not self.chain and self.pusher.has_skill(Skill.GRAB) and not self.player.has_skill(Skill.SIDE_STEP):
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
            self.push_to = action.position
            self.follow_to = self.player.position

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
            actions.append(ActionChoice(ActionType.USE_SKILL, skill=Skill.STAND_FIRM, team=self.player.team, players=[self.player]))
            actions.append(ActionChoice(ActionType.DONT_USE_SKILL, skill=Skill.STAND_FIRM, team=self.player.team, players=[self.player]))
        return actions


class Scatter(Procedure):

    def __init__(self, game, piece, kick=False, is_pass=False, gentle_gust=False):
        super().__init__(game)
        self.piece = piece
        self.kick = kick
        self.is_pass = is_pass
        self.gentle_gust = gentle_gust

    def step(self, action):

        # Don't scatter if ball is out
        if self.game.is_out_of_bounds(self.piece.position):
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
                self.game.shove(self.piece, x, y)

                if self.kick and i == 0 and not self.gentle_gust:
                    self.game.report(Outcome(OutcomeType.BALL_SCATTER, rolls=rolls))

                # Check out of bounds
                if self.kick:
                    if self.game.is_out_of_bounds(self.piece.position):
                        if self.gentle_gust:
                            # Touchback will be enforced after kick-off table when ball lands
                            self.game.report(Outcome(OutcomeType.GENTLE_GUST_OUT_OF_BOUNDS, position=self.piece.position,
                                                     rolls=rolls))
                        else:
                            # Touchback will be enforced after kick-off table when ball lands
                            self.game.report(Outcome(OutcomeType.KICK_OUT_OF_BOUNDS, position=self.piece.position,
                                                     rolls=rolls))
                        return True
                    elif self.game.is_team_side(self.piece.position, self.game.get_kicking_team()):
                        if self.gentle_gust:
                            # Touchback will be enforced after kick-off table when ball lands
                            self.game.report(Outcome(OutcomeType.GENTLE_GUST_OPP_HALF, position=self.piece.position,
                                                     rolls=rolls))
                        else:
                            # Touchback will be enforced after kick-off table when ball lands
                            self.game.report(Outcome(OutcomeType.KICK_OPP_HALF, position=self.piece.position, rolls=rolls))
                        return True
                else:
                    # Throw in
                    if self.game.is_out_of_bounds(self.piece.position):
                        if type(self.piece) is Ball:
                            ThrowIn(self.game, self.piece, self.game.get_square(self.piece.position.x-x, self.piece.position.y-y))
                            self.game.report(Outcome(OutcomeType.BALL_SCATTER, rolls=rolls))
                            self.game.report(Outcome(OutcomeType.BALL_OUT_OF_BOUNDS,
                                                     position=self.piece.position))
                        elif type(self.piece) is Player:
                            KnockDown(self.game, self.piece, in_crowd=True, armor_roll=False)
                            self.game.report(Outcome(OutcomeType.PLAYER_OUT_OF_BOUNDS, player=self.piece, rolls=rolls))
                        elif type(self.piece) is Bomb:
                            self.game.report(Outcome(OutcomeType.BOMB_OUT_OF_BOUNDS, rolls=rolls))
                            self.game.remove_bomb()
                        return True

                    # Passes are scattered three times
                    if self.is_pass and s < n-1:
                        continue

                    if type(self.piece) is Ball:
                        self.game.report(Outcome(OutcomeType.BALL_SCATTER, rolls=rolls))
                    elif type(self.piece) is Bomb:
                        self.game.report(Outcome(OutcomeType.BOMB_SCATTER, rolls=rolls))
                    elif type(self.piece) is Player:
                        self.game.report(Outcome(OutcomeType.PLAYER_SCATTER, player=self.piece, rolls=rolls))

                    # On player -> Catch
                    player_at = self.game.get_player_at(self.piece.position)
                    catcher = self.game.get_catcher(self.piece.position)

                    if type(self.piece) is Player and player_at is not None:
                        self.game.report(Outcome(OutcomeType.PLAYER_HIT_PLAYER, position=self.piece.position,
                                                 player=player_at, rolls=[roll_scatter]))
                        Bounce(self.game, self.piece)
                        KnockDown(self.game, player=player_at, inflictor=self.piece)
                        if self.piece.team == player_at.team:
                            self.game.get_proc(PassAttempt).turnover = True  # Will cause a turnover after pass attempt
                        return True
                    if catcher is not None and self.piece.is_catchable():
                        Catch(self.game, catcher, self.piece)
                        self.game.report(Outcome(OutcomeType.BALL_HIT_PLAYER, position=self.piece.position,
                                                 player=catcher, rolls=[roll_scatter]))
                        return True

        if self.kick:
            if self.gentle_gust:
                # Wait for ball to land
                self.game.report(Outcome(OutcomeType.GENTLE_GUST_IN_BOUNDS, position=self.piece.position,
                                         rolls=[roll_scatter]))
            else:
                # Wait for ball to land
                self.game.report(Outcome(OutcomeType.KICK_IN_BOUNDS, position=self.piece.position,
                                         rolls=[roll_scatter, roll_distance]))
        elif type(self.piece) is Player:
            Land(self.game, player=self.piece)
        elif type(self.piece) is Bomb:
            Explode(self.game, self.piece)
        else:
            Bounce(self.game, self.piece)
        return True


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


class Setup(Procedure):

    def __init__(self, game, team, reorganize=False):
        super().__init__(game)
        self.team = team
        self.reorganize = reorganize
        self.selected_player = None
        self.formations = game.config.defensive_formations if team == game.get_kicking_team() \
            else game.config.offensive_formations
        self.aa = []

    def start(self):
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
            for a in actions:
                self.step(a)
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
                if action.player.position is not None:
                    self.game.pitch_to_reserves(action.player)
                else:
                    print("Ignoring PLACE_PLAYER action: Already in reserves")
            elif action.player.position is None:
                if action.position is not None:
                    self.game.reserves_to_pitch(action.player, action.position)
                else:
                    print("Ignoring PLACE_PLAYER action: Already in reserves")
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
                             self.team) + self.game.get_reserves(self.team),
                         positions=positions),
            ActionChoice(ActionType.END_SETUP, team=self.team)
        ]

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
            x = 2 - roll_direction.get_sum()
        elif self.ball.position.y == len(self.game.arena.board)-2:  # Below
            y = -1
            x = -2 + roll_direction.get_sum()
        elif self.ball.position.x == 1:  # Right
            x = 1
            y = -2 + roll_direction.get_sum()
        elif self.ball.position.x == len(self.game.arena.board[0])-2:  # Left
            x = -1
            y = 2 - roll_direction.get_sum()

        for i in range(roll_distance.get_sum()):
            self.ball.move(x, y)
            if self.game.is_out_of_bounds(self.ball.position):
                ThrowIn(self.game, self.ball, self.game.get_square(self.ball.position.x - x, self.ball.position.y - y))
                self.game.report(Outcome(OutcomeType.THROW_IN_OUT_OF_BOUNDS, position=self.ball.position,
                                         rolls=[roll_direction, roll_distance]))
                return True

        self.game.report(Outcome(OutcomeType.THROW_IN, position=self.ball.position, rolls=[roll_direction,  roll_distance]))

        # On player -> Catch
        catcher = self.game.get_catcher(self.ball.position)
        if catcher is not None:
            Catch(self.game, catcher, self.ball)
        else:
            Bounce(self.game, self.ball)

        return True


class Turnover(Procedure):

    def __init__(self, game):
        super().__init__(game)

    def step(self, action):
        self.game.report(Outcome(OutcomeType.TURNOVER, team=self.game.state.current_team))
        self.game.state.active_player = None
        EndTurn(self.game)
        return True


class Touchdown(Procedure):

    def __init__(self, game, player):
        super().__init__(game)
        self.player = player
        self.handle_bloodlust = False 
        self.eat_thrall = None 
        
    def start(self): 
        self.handle_bloodlust = self.player == self.game.get_active_player() and self.player.state.blood_lust 

    def step(self, action):
        if self.handle_bloodlust: 
                
            if self.eat_thrall is None: 
                self.eat_thrall = EatThrall(self.game, self.player)
                return None 
            
            if self.eat_thrall.failed: 
                return True 
        
        self.game.report(Outcome(OutcomeType.TOUCHDOWN, team=self.player.team, player=self.player))
        self.player.team.state.score += 1
        self.game.state.kicking_this_drive = self.player.team
        self.game.state.receiving_this_drive = self.game.get_opp_team(self.player.team)
        EndTurn(self.game, kickoff=True)

        return True


class TurnStunned(Procedure):

    def __init__(self, game, team):
        super().__init__(game)
        self.team = team

    def step(self, action):
        for player in self.team.players:
            if player.state.stunned:
                player.state.stunned = False
                player.state.used = True
        self.game.report(Outcome(OutcomeType.STUNNED_TURNED))
        return True


class EndTurn(Procedure):

    def __init__(self, game, kickoff=False):
        super().__init__(game)
        self.kickoff = kickoff

    def step(self, action):

        # Remove all procs in the current turn - including the current turn proc.
        # In rare cases there is no Turn proc yet - e.g. during a kickoff
        if self.game.get_current_turn_proc() is not None:
            self.game.purge_stack_until(Turn, inclusive=True)

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
                PreKickoff(self.game, team)

            ClearBoard(self.game)

        self.game.state.current_team = None
        self.game.remove_clocks()
        self.game.state.rerolled_procs.clear()
        self.game.state.player_action_type = None
        return True


class Turn(Procedure):

    def __init__(self, game, team, half, turn, blitz=False, quick_snap=False):
        super().__init__(game)
        self.team = team
        self.half = half
        self.turn = turn
        self.blitz = blitz
        self.quick_snap = quick_snap
        self.blitz_available = not quick_snap
        self.pass_available = not quick_snap
        self.handoff_available = not quick_snap
        self.foul_available = not quick_snap

    def start(self):
        self.game.state.current_team = self.team
        self.game.add_primary_clock(self.team)
        if self.blitz:
            self.game.report(Outcome(OutcomeType.BLITZ_START, team=self.team))
        elif self.quick_snap:
            self.game.report(Outcome(OutcomeType.QUICK_SNAP_START, team=self.team))
        else:
            self.game.report(Outcome(OutcomeType.TURN_START, team=self.team, n=self.turn))
            self.team.state.turn = self.turn
            TurnStunned(self.game, self.team)

    def step(self, action):

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

        action.player.state.hypnotized = False

        self.game.state.active_player = action.player
            
        # Start movement action
        if action.action_type == ActionType.START_MOVE:
            MoveAction(self.game, action.player)

        # Start blitz action
        if action.action_type == ActionType.START_BLITZ:
            BlitzAction(self.game, action.player)

        # Start foul action
        if action.action_type == ActionType.START_FOUL:
            FoulAction(self.game, action.player)

        # Start block action
        if action.action_type == ActionType.START_BLOCK:
            BlockAction(self.game, action.player)

        # Start pass action
        if action.action_type == ActionType.START_PASS:
            PassAction(self.game, action.player)

        # Start handoff action
        if action.action_type == ActionType.START_HANDOFF:
            HandoffAction(self.game, action.player)

        # Start handoff action
        if action.action_type == ActionType.START_THROW_BOMB:
            ThrowBombAction(self.game, action.player)

        if action.player.has_skill(Skill.BONE_HEAD):
            Bonehead(self.game, action.player)
        if action.player.has_skill(Skill.REALLY_STUPID):
            ReallyStupid(self.game, action.player)
        if action.player.has_skill(Skill.WILD_ANIMAL):
            WildAnimal(self.game, action.player, is_block_or_blitz=action.action_type == ActionType.START_BLOCK or action.action_type == ActionType.START_BLITZ)
        if action.player.has_skill(Skill.TAKE_ROOT) and not action.player.state.taken_root:
            TakeRoot(self.game, action.player)
        if action.player.has_skill(Skill.BLOOD_LUST):
            BloodLust(self.game, action.player, is_block=action.action_type == ActionType.START_BLOCK)

        return False

    def available_actions(self):
        move_players = []
        block_players = []
        blitz_players = []
        pass_players = []
        handoff_players = []
        foul_players = []
        bomb_players = []
        for player in self.game.get_players_on_pitch(self.team):
            if self.blitz:
                if self.game.num_tackle_zones_in(player) > 0:
                    continue
            if not player.state.used:
                if not player.state.taken_root:
                    move_players.append(player)
                if self.blitz_available and not player.state.taken_root:
                    blitz_players.append(player)
                if self.pass_available:
                    if not player.state.taken_root or self.game.get_ball_carrier() == player:
                        pass_players.append(player)
                if self.handoff_available:
                    if not player.state.taken_root or self.game.get_ball_carrier() == player:
                        handoff_players.append(player)
                if self.foul_available:
                    if not player.state.taken_root or self.game.get_adjacent_opponents(player, down=True, standing=False):
                        foul_players.append(player)
                if not self.quick_snap and not self.blitz and (player.state.up or player.has_skill(Skill.JUMP_UP)):
                    if self.game.get_adjacent_opponents(player, down=False):
                        block_players.append(player)
                if player.state.up and player.has_skill(Skill.BOMBARDIER):
                    bomb_players.append(player)

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
        if len(bomb_players) > 0:
            actions.append(ActionChoice(ActionType.START_THROW_BOMB, team=self.team, players=bomb_players))
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


class Negatrait(Procedure, metaclass=ABCMeta):
    def __init__(self, game, player, ends_turn=True):
        super().__init__(game)
        self.player = player
        self.waiting_reroll = False
        self.reroll_used = False
        self.rolled = False
        self.roll = None
        self.reroll = None

        self.roll_type = None
        self.skill = None
        self.success_outcome = None
        self.fail_outcome = None
        self.ends_turn = ends_turn

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
        self.player.state.failed_nega_trait_this_turn = True
        if self.ends_turn:
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
                # check reroll
                self.reroll = Reroll(self.game, self.player, self)
                return False

        # Is re-roll used?
        assert self.reroll is not None
        if self.reroll.use_reroll:
            # re-roll allowed
            self.rolled = False
            self.reroll = None
            return self.step(None)
        else:
            # no re-roll
            self.trigger_failure()
            return True

    def end_turn(self):
        # mark player turn as over
        EndPlayerTurn(self.game, self.player)


class Bonehead(Negatrait):
    def __init__(self, game, player=True):
        super().__init__(game, player)
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
    def __init__(self, game, player=True):
        super().__init__(game, player)
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
    def __init__(self, game, player=True, is_block_or_blitz=False):
        super().__init__(game, player)
        self.is_block_or_blitz = is_block_or_blitz
        self.roll_type = RollType.WILD_ANIMAL_ROLL
        self.skill = Skill.WILD_ANIMAL
        self.success_outcome = OutcomeType.SUCCESSFUL_WILD_ANIMAL
        self.fail_outcome = OutcomeType.FAILED_WILD_ANIMAL

    def get_target(self):
        target = 4
        if self.is_block_or_blitz:
            target = 2
        return target

    def apply_fail_state(self):
        self.player.state.wild_animal = True

    def remove_fail_state(self):
        self.player.state.wild_animal = False


class TakeRoot(Negatrait):
    def __init__(self, game, player=True):
        super().__init__(game, player, ends_turn=False)
        self.roll_type = RollType.TAKE_ROOT_ROLL
        self.skill = Skill.TAKE_ROOT
        self.success_outcome = OutcomeType.SUCCESSFUL_TAKE_ROOT
        self.fail_outcome = OutcomeType.FAILED_TAKE_ROOT

    def get_target(self):
        return 2

    def apply_fail_state(self):
        self.player.state.taken_root = True

    def remove_fail_state(self):
        pass  # taken_root is only reset upon end of drive


class BloodLustBlockOrMove(Procedure):

    def __init__(self, game, player):
        super().__init__(game)
        self.player = player

    def step(self, action):
        if action.action_type == ActionType.START_BLOCK:
            return True
        # Cancel block action
        self.game.purge_stack_until(Turn, inclusive=False)
        # Start move action
        MoveAction(self.game, self.player)
        self.game.report(Outcome(OutcomeType.MOVE_ACTION_STARTED, player=self.player))
        return True

    def available_actions(self):
        return [ActionChoice(ActionType.START_BLOCK, self.player.team, players=[self.player]),
                ActionChoice(ActionType.START_MOVE, self.player.team, players=[self.player])]


class BloodLust(Negatrait):
    def __init__(self, game, player=True, is_block=False):
        super().__init__(game, player, ends_turn=False)
        self.is_block = is_block
        self.roll_type = RollType.BLOOD_LUST_ROLL
        self.skill = Skill.BLOOD_LUST
        self.success_outcome = OutcomeType.SUCCESSFUL_BLOOD_LUST
        self.fail_outcome = OutcomeType.FAILED_BLOOD_LUST
        
    def get_target(self):
        return 2

    def apply_fail_state(self):
        #set_trace()
        self.player.state.blood_lust = True
        if self.is_block:
            BloodLustBlockOrMove(self.game, self.player)

    def remove_fail_state(self):
        pass  # blood lust is removed by EatThrall-procedure 
        

class Reroll(Procedure):

    def __init__(self, game, player, context):
        super().__init__(game, context=context)
        self.player = player
        self.use_reroll = None
        self.loner = None
        self.can_use_team_reroll = False
        self.can_use_pro = False
        self.pro = None
        self.secondary_clock = False
        self.skill = None
        self.block_actions = []
        self.block_action = None

    def start(self):

        # If procedure was already re-rolled, don't re-roll
        if self.context in self.game.state.rerolled_procs:
            self.can_use_pro = False
            self.can_use_team_reroll = False
            return

        # Collect block dice actions if context is block
        if type(self.context) == Block:
            self.block_actions.extend(self.context.available_actions())

        # Pro skill
        if self.player.can_use_skill(Skill.PRO):
            self.can_use_pro = True

        # Determine if team re-rolls are available
        self.can_use_team_reroll = self.game.can_use_reroll(self.player.team)

        # Check for automatic skill re-rolls
        self.skill = None
        if type(self.context) == Dodge and self.player.can_use_skill(Skill.DODGE):
            # Get players with tackle
            tacklers = self.game.get_adjacent_opponents(self.player, down=False, skill=Skill.TACKLE)
            if len(tacklers) == 0:
                self.player.use_skill(Skill.DODGE)
                self.use_reroll = True
                self.skill = Skill.DODGE
        if type(self.context) == PassAttempt and self.player.can_use_skill(Skill.PASS):
            self.use_reroll = True
            self.skill = Skill.PASS
        if type(self.context) == Intercept:
            # Make sure it is not a safe throw agility re-roll
            if self.context.safe_throw_reroll is None and self.player.can_use_skill(Skill.CATCH):
                self.use_reroll = True
                self.skill = Skill.CATCH
        if type(self.context) == Catch:
            if self.player.can_use_skill(Skill.CATCH):
                self.use_reroll = True
                self.skill = Skill.CATCH
        if type(self.context) == Pickup and self.player.can_use_skill(Skill.SURE_HANDS):
            self.use_reroll = True
            self.skill = Skill.SURE_HANDS
        if type(self.context) == GFI and self.player.can_use_skill(Skill.SURE_FEET):
            self.player.use_skill(Skill.SURE_FEET)
            self.use_reroll = True
            self.skill = Skill.SURE_FEET

        # Add secondary clock if waiting for opponent decision
        if self.can_use_pro and self.player.team != self.game.state.current_team:
            self.game.add_secondary_clock(self.player.team)
            self.secondary_clock = True

    def step(self, action):

        # If skill
        if self.skill is not None:
            self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.player, skill=self.skill))
            return True

        # If no re-rolls are available
        if not self.can_use_team_reroll and not self.can_use_pro:
            return True

        # Did use of Pro skill succeed?
        if self.pro is not None:
            self.use_reroll = self.pro.success
            # If Pro (with possible re-rolls) failed - team re-rolls are not allowed.
            return True

        # Use Pro skill?
        if self.can_use_pro:
            if action.action_type == ActionType.USE_SKILL:
                self.game.report(Outcome(OutcomeType.SKILL_USED, player=self.player, skill=Skill.PRO))
                self.pro = Pro(self.game, self.player, context=self)
                return False
            elif action.action_type == ActionType.DONT_USE_SKILL:
                self.can_use_pro = False

        # Did Loner roll succeed?
        if self.loner is not None:
            self.use_reroll = self.loner.success
            return True

        # If action is among parent actions
        if action is not None:
            for action_choice in self.block_actions:
                if action_choice.action_type == action.action_type:
                    self.block_action = action
                    return True

        # Team re-roll
        if self.can_use_team_reroll:
            if action.action_type == ActionType.USE_REROLL:
                self.use_reroll = True
                self.player.team.state.use_reroll()
                self.game.report(Outcome(OutcomeType.REROLL_USED, team=self.player.team))
                if self.player.has_skill(Skill.LONER):
                    self.loner = Loner(self.game, self.player, context=self)
                    return False
            elif action.action_type == ActionType.DONT_USE_REROLL:
                self.use_reroll = False
            return True

        # If neither Pro or team re-rolls can be used
        self.use_reroll = False
        return True

    def end(self):
        # Stop clock ff decision is opponent's
        if self.secondary_clock:
            self.game.remove_secondary_clocks()

        # Add context to set of re-rolled procs
        if self.use_reroll:
            self.game.state.rerolled_procs.add(self.context)

    def available_actions(self):
        """
        If bot, take actions in several steps: 1. Pro, 2. Reroll, 3. Go back to context
        If human, show all actions at once, but hide uneccessary don't use/reroll actions.
        """
        actions = []
        if self.skill is not None or self.loner is not None:
            return actions
        if not self.game.get_team_agent(self.player.team).human:  # Actor is non-human agent, action accuired from agent.act(game)
            if self.can_use_pro:
                actions = [
                    ActionChoice(ActionType.USE_SKILL, skill=Skill.PRO, team=self.game.state.current_team),
                    ActionChoice(ActionType.DONT_USE_SKILL, skill=Skill.PRO, team=self.game.state.current_team)
                ]
            elif self.can_use_team_reroll:
                actions = [
                    ActionChoice(ActionType.USE_REROLL, team=self.player.team),
                    ActionChoice(ActionType.DONT_USE_REROLL, team=self.player.team)
                ]
        else:  # actor is human agent. action supplied through game.step(action) 
            if len(self.block_actions) == 0 or self.context.favor != self.player.team: 
                if self.can_use_pro:
                    actions = [
                        ActionChoice(ActionType.USE_SKILL, skill=Skill.PRO, team=self.game.state.current_team)
                    ]
                    if not self.can_use_team_reroll:
                        actions.append(ActionChoice(ActionType.DONT_USE_SKILL, skill=Skill.PRO, team=self.game.state.current_team))
                if self.can_use_team_reroll:
                    actions = [
                        ActionChoice(ActionType.USE_REROLL, team=self.player.team),
                        ActionChoice(ActionType.DONT_USE_REROLL, team=self.player.team)
                    ]
            else: 
                if self.can_use_pro:
                    actions = [
                        ActionChoice(ActionType.USE_SKILL, skill=Skill.PRO, team=self.game.state.current_team)
                    ]
                if self.can_use_team_reroll:
                    actions = [
                        ActionChoice(ActionType.USE_REROLL, team=self.player.team)
                    ]

            if len(self.block_actions) > 0 and len(actions) > 0:
                actions.extend(self.block_actions)

        return actions


class Pro(Procedure):
    def __init__(self, game, player, context):
        super().__init__(game, context=context)
        self.success = False
        self.roll = None
        self.player = player
        self.reroll = None

    def start(self):
        assert self.player.can_use_skill(Skill.PRO)
        self.player.use_skill(Skill.PRO)

    def step(self, action):
        if self.roll is None:
            # Roll
            self.roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.PRO)
            self.roll.target = 4

            if self.roll.is_d6_success():
                # Success
                self.game.report(Outcome(OutcomeType.SUCCESSFUL_PRO, player=self.player, rolls=[self.roll]))
                self.success = True
                return True
            else:
                self.game.report(Outcome(OutcomeType.FAILED_PRO, player=self.player, rolls=[self.roll]))
                # Team re-rolls can be used to re-roll a failed Pro roll
                self.reroll = Reroll(self.game, self.player, context=self.context)
                return False

        # If re-roll used
        assert self.reroll is not None
        if self.reroll.use_reroll:
            # re-roll
            self.roll = None
            return self.step(None)
        else:
            # no re-roll - self.success is already False
            return True


class Loner(Procedure):
    def __init__(self, game, player, context):
        super().__init__(game, context=context)
        self.success = False
        self.roll = None
        self.player = player
        self.reroll = None

    def step(self, action):
        if self.roll is None:
            # Roll
            self.roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.LONER_ROLL)
            self.roll.target = 4

            if self.roll.is_d6_success():
                # Success
                self.game.report(Outcome(OutcomeType.SUCCESSFUL_LONER, player=self.player, rolls=[self.roll]))
                self.success = True
                return True
            else:
                self.game.report(Outcome(OutcomeType.FAILED_LONER, player=self.player, skill=Skill.LONER, rolls=[self.roll]))
                # Failed loner rolls can be re-rolled if the player has the Pro skill
                self.reroll = Reroll(self.game, self.player, context=self.context)
                return False

        # If re-roll used
        assert self.reroll is not None
        if self.reroll.use_reroll:
            # re-roll
            self.roll = None
            return self.step(None)
        else:
            # no re-roll
            self.result = False
            return True


class EatThrall(Procedure): 
    def __init__(self, game, player):
        super().__init__(game)
        self.player = player 
        self.victim = None 
        self.failed = None 
        
    def start(self): 
        self.victim_pos = self.game.get_adjacent_blood_lust_victims(self.player)
        
    def step(self, action): 
        if len(self.victim_pos)==0: 
            Turnover(self.game)
            
            ball = self.game.get_ball_at(self.player.position)
            if ball is not None: 
                Bounce(self.game, ball)
            
            self.game.pitch_to_reserves(self.player)
            self.game.report(Outcome(OutcomeType.EJECTED_BY_BLOOD_LUST, player=self.player))
            
            self.failed = True 
            
        else: 
            self.victim = self.game.get_player_at(action.position)  
            #set_trace() 
            self.game.report(Outcome(OutcomeType.EATEN_DURING_BLOOD_LUST, player=self.victim, opp_player=self.player))
            KnockDown(self.game, self.victim, armor_roll=False, injury_roll=True)
            self.failed = False 
            
        return True 
        
    def end(self): 
        self.player.state.blood_lust = False
        
    def available_actions(self): 
        if len(self.victim_pos) > 0:
            return [ActionChoice(ActionType.SELECT_PLAYER, positions=self.victim_pos, team=self.player.team)] 
        else: 
            return [] 


class HypnoticGaze(Procedure): 
    def __init__(self, game, player, target_player): 
        super().__init__(game)
        self.player = player
        self.target_player = target_player 
        self.roll = None
        self.reroll = None 
        
    def start(self): 
        pass 
    
    def step(self, action): 
        if self.roll is None: 
            #agility roll with tz modifiers except target_player
            self.roll = DiceRoll([D6(self.game.rnd)], roll_type=RollType.AGILITY_ROLL)
            self.roll.modifiers = self.game.get_hypno_modifier(self.player)
            self.roll.target = Rules.agility_table[self.player.get_ag()]
            
            if self.roll.is_d6_success(): 
                self.game.report(Outcome(OutcomeType.SUCCESSFUL_HYPNOTIC_GAZE, player=self.player, rolls=[self.roll]))
                self.target_player.state.hypnotized = True 
                return True 
            
            self.game.report(Outcome(OutcomeType.FAILED_HYPNOTIC_GAZE, player=self.player, rolls=[self.roll]))
            self.reroll = Reroll(self.game, self.player, self)
            return False
        
        assert self.reroll is not None 
        
        if self.reroll.use_reroll: 
            self.roll = None 
            self.reroll = None 
            return False 
        
        return True 

