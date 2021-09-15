# 2020 ruleset feature list

The purpose of this document gather a summary of the delta between the 2016 and 2020 rules, and showing the progress 
towards an implementation of the 2020 ruleset. The text below might not be perfect, check the rulebook before coding!    


## Terminology 
The 2020 rules coins a few new terms which are not rules but helps explain the rules.

| Term          | Meaning  |
|---------------|----------|
| Open player   | Player not in any opponent's tacklezone   |
| Marked player | Player in one or more of opponent's tacklezones  |
| Rush          | Rename of GFI |
| Deviate       | Moving the ball D6 squares in direction determined by a D8 (e.g. kickoff)   |
| Scatter       | Move ball three squares in direction determined by three subsequent D8 (e.g. inaccurate pass)|
| Bounce        | Move ball one square in direction determined by D8 (e.g. failed catch/pickup)  |
| Stalling      | Player can score with full certainty but chooses not to |



## Prayers to Nuffle
For every 50k in Team Value difference the underdog coach gets one roll (D16) on the Prayers to Nuffle table. This in addition to Petty Cash. This is also a kickoff result.   

| Prayer                        | Implemented   | Has Test(s) | 
|-------------------------------|---------------|-----------|
| 1 - Treacherou Trapdoor       |  :x:           |  No    |
| 2 - Friends with the Ref      |  :x:           |  No    |
| 3 - Stiletto                  |  :x:           |  No    |
| 4 - Iron Man                  |  :x:           |  No    |
| 5 - Knuckle Duster            |  :x:           |  No    |
| 6 - Bad Habits                |  No           |  No    |
| 7 - Greasy Cleats             |  No           |  No    |
| 8 - Blessed Statue of Nuffle  |  No           |  No    |
| 9 - Moles under the Pitch     |  No           |  No    |
| 10 - Perfect Passing          |  No           |  No    |
| 11 - Fan Interaction          |  No           |  No    |
| 12 - Necessary Violence       |  No           |  No    |
| 13 - Fouling Frenzy           |  No           |  No    |
| 14 - Throw a Rock             |  No           |  No    |
| 15 - Under Scrutiny           |  No           |  No    |
| 16 - Intensive Training       |  No           |  No    |




## Kickoff event table 

Fan Factor is used instead of Fame in some places. 

| Feature                | Implemented | Has Test(s) | Change  |
|------------------------|-------------|-----------|-------|
| 2 - Get the ref        | ✔️         | ❌        | Same |
| 3 - Time out           | No          | No       | (old Riot) Kicking team's turn marker on 6,7,8 then substract one turn, else add one.  |
| 4 - Solid defence      | No          | No       | (old Perfect Defence) Limited to D3+3 players  |
| 5 - High Kick          | N/A         | Yes        | Same |
| 6 - Cheering Fans      | No          | No       | D6+fan factor. Winner rolls a Player to Nuffle|
| 7 - Brilliant Coaching | N/A         | Yes        | Same |
| 8 - Changing Weather   | N/A         | ?        | If the new weather is Perfect Conditions the ball now Scatters (i.e. 3D8) |
| 9 - Quicksnap          | No          | No       | Limited to D3+3 open players|
| 10 - Blitz             | No          | No       | D3+3 open players  |
| 11 - Officious Ref     | No          | No       | D6+fan factor to determine coach. Randomly select player. D6: 1: Sent to dungeon. 2+ Stunned |
| 12 - Pitch invasion!   | No          | No       | D6+fan factor to determine coach. Randomly select D3 players. Stunned  |



## Skills 
'Jump' refers to the action of Jumping over Prone players which all players can do now.  

### Removed skills 
Skills that are completely removed are: Piling On, 

### Agility 
| Skill             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-------------|-------|
|Catch              | Yes         | No          | same |
|Diving Catch       | Yes         | No          | same |
|Diving Tackle      | No          | No          | works on leap and jump too |
|Dodge              | Yes         | No          | same |
|Defensive          | No          | No          | new: cancels adjescent opponent's Guard during Opponent's turn. |
|Jump Up            | Yes         | No          | same |
|Leap               | No          | No          | may Jump over any type of square, negative modifer reduce by 1, to minimum of -1 |
|Safe Pair of Hands | No          | No          | new |
|Sidestep           | Yes         | No          | same |
|Sneaky Git         | No          | No          | not sent of because of doubles on armor even if it breaks, may move after the foul |
|Sprint             | Yes         | No          | same |
|Sure Feet          | Yes         | No          | same |

### General
| Skill            | Implemented | Has Test(s) | Change  |
|------------------|-------------|-------------|-------|
|Block             | Yes         | Yes          | same |
|Dauntless         | Yes         | No           | same |
|Dirty player (+1) | Yes         | No           | same |
|Fend              | Yes         | No           | same |
|Frenzy            | Yes         | Yes          | same |
|Kick              | No          | No           | same |
|Pro               | No          | No           | +3 instead of +4. May only re-roll one dice |
|Shadowing         | No          | No           | success determined by D6 +own MA -opp MA > 5  |
|Strip Ball        | Yes         | Yes          | same |
|Sure Hands        | Yes         | Yes          | same |
|Tackle            | Yes         | No           | same |
|Wrestle           | Yes         | No           | same |

### Mutations
| Skill              | Implemented | Has Test(s) | Change  |
|--------------------|------------|-------------|-------|
|Big Hand            | Yes        | Yes         | same |
|Claws               | No         | No          | doesn't stack with Mightly blow |
|Disturbing Presence | Yes        | No          | same |
|Extra Arms          | Yes        | Yes         | same |
|Foul Appearance     | Yes        | No          | same |
|Horns               | Yes        | No          | same |
|Iron Hard Skin      | No         | No          | new: Immune to Claw |
|Monstrous Mouth     | No         | No          | new: Catch re-roll and immune to Strip Ball |
|Prehensile Tail     | No         | No          | works on Leap and Jump |
|Tentacles           | No         | No          | success determined as D6 +own St -opp ST > 5 |
|Two Heads           | Yes        | No          | same |
|Very Long Legs      | No         | No          | negative modifers for Jump and Leap (if player has skill) reduce by 1, to minimum of -1, Immune to Cloud Burster |

### Passing
| Skill          | Implemented | Has Test(s) | Change  |
|----------------|-------------|-------------|-------|
|Accurate        | No             | No          | Only quick pass and short pass |
|Connoneer       | No             | No          | as Accurate but on Long pass and Long Bomb |
|Cloud Burster   | No             | No          | New: choose if opposing coach shall re-roll a successful Interfere when throwing Long or Long Bomb  |
|Dump-Off        | Yes            | No          | same |
|Fumblerooskie   | No             | No          | new: leave ball in vacated square during movement. No dice involved |
|Hail Mary Pass  | No             | No          | tacklezones matter |
|Leader          | No             | No          | same |
|Nerves of Steel | Yes            | Yes         | same |
|On the Ball     | No             | No          | Kick off return and pass block combined |
|Pass            | Yes            | Yes         | same |
|Running Pass    | No             | No          | new: may continue moving after quick pass |
|Safe Pass       | No             | No          | Fumbled passes doesn't cause bounce nor turnover |


### Strength
| Skill             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-------------|-------|
|Arm Bar            | No          | No          | new |
|Brawler            | No          | No          | new |
|Break Tackle       | No          | No          | +2 on dodge if ST>4 else +1 once per turn  |
|Grab               | Yes         | No          | same |
|Guard              | No          | No          | works on fouls too |
|Juggernaut         | Yes         | No          | same |
|Mighty Blow +1     | No          | No          | doesn't work passively (e.g. attacker down), +X |
|Multiple Block     | No          | No          | same |
|Pile Driver        | No          | No          | As piling on but is evaluate as a foul |
|Stand Firm         | Yes         | No          | same |
|Strong Arm         | No          | No          | Only applicable for Throw Team-mate |
|Thick Skull        | Yes         | No          | same |

### Traits
| Skill             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-------------|-------|
|Animal Savagery    | No          | No          | new |
|Animosity          | No          | No          | same? |
|Always Hungry      | Yes         | Yes         | same |
|Ball & Chain       | No          | No          | ? |
|Bombardier         | No          | No          | ? |
|Bone Head          | Yes         | Yes         | same |
|Chainsaw           | No          | No          | ? |
|Decay              | No          | No          | ? |
|Hypnotic Gaze      | Yes         | Yes         | same |
|Kick Team-mate     | No          | No          | ? |
|Loner (+X)         | No          | No          | +X is new |
|No Hands           | Yes         | No          | same |
|Plague Ridden      | No          | No          | new |
|Pogo Stick         | No          | No          | ? |
|Projectile Vomit   | No          | No          | new  |
|Really Stupid      | Yes         | Yes         | same |
|Regeneration       | Yes         | Yes         | same |
|Right Stuff        | Yes         | Yes         | same? |
|Secret Weapon      | No          | No          | same |
|Stab               | Yes         | No          | same |
|Stunty             | No          | No          | No modifier for passing, but opponent gets +1 when interfering with pass from Stunty Player |
|Swarming           | No          | No          | new |
|Swoop              | No          | No          | ? |
|Take Root          | Yes         | Yes         | same |
|Titchy             | No          | No          | Same but also can never cause negative modifiers to opponent player's agility test (e.g. catching/throwing ball) |
|Throw Team-mate    | No          | No          | ? |
|Timmm-ber!         | Yes         | Yes         | same |
|Unchannelled Fury  | No          | No          | new |


## Passing

| Feature                | Implemented | Has Test(s) | Change  |
|------------------------|-------------|-------------|-------|
| Passing characteristic | No          | No          | AG does no longer determine passing ability |
| New distance modifiers | No          | No          | 0 for quick, ... , -3 for Long Bombs |
| Wildly inaccurate pass | No          | No          | Roll is 1 after modifiers. Deviates (like a kickoff) |
| Catch modifiers        | No          | No          | 0 for accurate |
| Pass Interference      | No          | No          | Replaces old interception rules |


## Other feature changes
Fame is removed and replaced with a similar 'Fan Factor' 


| Feature                 | Implemented | Has Test(s) | Change  |
|-------------------------|-------------|-------------|---------|
| Fan factor              | No          | No          | D3 + nbr of Dedicated Fans the team has |
| Team re-rolls           | No          | No          | Multiple team re-rolls can be used per turn |
| Player characteristic   | No          | No          | AG and AV is now e.g. 3+ instead of 4 |
| Passing characteristic  | No          | No          | new |
| Sweltering heat         | No          | No          | D3 players, randomly selected |
| Jump over prone players | No          | No          | New  |
| Niggling Injury         | No          | No          | +1 on Casualty roll instead |
| Casualty table          | No          | No          | D16 table  |
| Stunty Injury table     | No          | No          | 2-6 stunned, 7-8 KO, 9 Badly Hurt (without casualty roll), 10+ Casualty  |



## Races 

All races have changed. This table shows what is working. 

| Race             | Missing positions       | Missing skills                     | Have icons    |
|------------------|-------------------------|------------------------------------|----------------
| Amazon             |                         |                                    | YES    |
| Black Orc          |                         |                                    | NO   |
| Chaos Dwarf        |                         |                                    | YES    |
| Chaos Choosen      |                         |                                    | YES    |
| Chaos Renegades    |                         |                                    | Maybe? |
| Dark Elf           |                         |                                    | YES    |
| Dwarf              |                         |                                    | NO     |
| Elven Union        |                         |                                    | YES    |
| Goblin             |                         |                                    | Some?  |
| Halfling           |                         |                                    | YES    |
| High Elf           |                         |                                    | YES    |
| Human**            |                         |                                    | YES    |
| Imperial Nobility  |                         |                                    | NO     |
| Tomb Kings         |                         |                                    | YES    |
| Lizardmen          |                         |                                    | YES    |
| Necromantic        |                         |                                    | NO     |
| Norse              |                         |                                    | YES    |
| Nurgle*            |                         |                                    | NO     |
| Ogre               |                         |                                    | YES    |
| Old World Alliance |                         |                                    | NO    |
| Orc                |                         |                                    | YES    |
| Shambling Undead   |                         |                                    | YES    |
| Skaven             |                         |                                    | YES    |
| Snotling           |                         |                                    | No    |
| Vampire            |                         |                                    | YES    |
| Underworld Denizens|                         |                                    | Maybe? |         
| Wood Elf           |                         |                                    | YES   | 

* Nurgle's Rot needs to be implemented in the post-game sequence
