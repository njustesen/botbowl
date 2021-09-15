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
| 1 - Treacherou Trapdoor       |  :x:           |  :x:    |
| 2 - Friends with the Ref      |  :x:           |  :x:    |
| 3 - Stiletto                  |  :x:           |  :x:    |
| 4 - Iron Man                  |  :x:           |  :x:    |
| 5 - Knuckle Duster            |  :x:           |  :x:    |
| 6 - Bad Habits                |  :x:           |  :x:    |
| 7 - Greasy Cleats             |  :x:           |  :x:    |
| 8 - Blessed Statue of Nuffle  |  :x:           |  :x:    |
| 9 - Moles under the Pitch     |  :x:           |  :x:    |
| 10 - Perfect Passing          |  :x:           |  :x:    |
| 11 - Fan Interaction          |  :x:           |  :x:    |
| 12 - Necessary Violence       |  :x:           |  :x:    |
| 13 - Fouling Frenzy           |  :x:           |  :x:    |
| 14 - Throw a Rock             |  :x:           |  :x:    |
| 15 - Under Scrutiny           |  :x:           |  :x:    |
| 16 - Intensive Training       |  :x:           |  :x:    |




## Kickoff event table 

Fan Factor is used instead of Fame in some places. 

| Feature                | Implemented | Has Test(s) | Change  |
|------------------------|-------------|-----------|-------|
| 2 - Get the ref        | ✔️         | ❌        | Same |
| 3 - Time out           | :x:          | :x:       | (old Riot) Kicking team's turn marker on 6,7,8 then substract one turn, else add one.  |
| 4 - Solid defence      | :x:          | :x:       | (old Perfect Defence) Limited to D3+3 players  |
| 5 - High Kick          | ✔         | ✔        | Same |
| 6 - Cheering Fans      | :x:          | :x:       | D6+fan factor. Winner rolls a Player to Nuffle|
| 7 - Brilliant Coaching | ✔         | ✔        | Same |
| 8 - Changing Weather   | ✔         | ?        | If the new weather is Perfect Conditions the ball now Scatters (i.e. 3D8) |
| 9 - Quicksnap          | :x:          | :x:       | Limited to D3+3 open players|
| 10 - Blitz             | :x:          | :x:       | D3+3 open players  |
| 11 - Officious Ref     | :x:          | :x:       | D6+fan factor to determine coach. Randomly select player. D6: 1: Sent to dungeon. 2+ Stunned |
| 12 - Pitch invasion!   | :x:          | :x:       | D6+fan factor to determine coach. Randomly select D3 players. Stunned  |



## Skills 
'Jump' refers to the action of Jumping over Prone players which all players can do now.  

### Removed skills 
Skills that are completely removed are: Piling On, 

### Agility 
| Skill             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-------------|-------|
|Catch              | ✔         | :x:          | same |
|Diving Catch       | ✔         | :x:          | same |
|Diving Tackle      | :x:          | :x:          | works on leap and jump too |
|Dodge              | ✔         | :x:          | same |
|Defensive          | :x:          | :x:          | new: cancels adjescent opponent's Guard during Opponent's turn. |
|Jump Up            | ✔         | :x:          | same |
|Leap               | :x:          | :x:          | may Jump over any type of square, negative modifer reduce by 1, to minimum of -1 |
|Safe Pair of Hands | :x:          | :x:          | new |
|Sidestep           | ✔         | :x:          | same |
|Sneaky Git         | :x:          | :x:          | not sent of because of doubles on armor even if it breaks, may move after the foul |
|Sprint             | ✔         | :x:          | same |
|Sure Feet          | ✔         | :x:          | same |

### General
| Skill            | Implemented | Has Test(s) | Change  |
|------------------|-------------|-------------|-------|
|Block             | ✔         | ✔          | same |
|Dauntless         | ✔         | :x:           | same |
|Dirty player (+1) | ✔         | :x:           | same |
|Fend              | ✔         | :x:           | same |
|Frenzy            | ✔         | ✔          | same |
|Kick              | :x:          | :x:           | same |
|Pro               | :x:          | :x:           | +3 instead of +4. May only re-roll one dice |
|Shadowing         | :x:          | :x:           | success determined by D6 +own MA -opp MA > 5  |
|Strip Ball        | ✔         | ✔          | same |
|Sure Hands        | ✔         | ✔          | same |
|Tackle            | ✔         | :x:           | same |
|Wrestle           | ✔         | :x:           | same |

### Mutations
| Skill              | Implemented | Has Test(s) | Change  |
|--------------------|------------|-------------|-------|
|Big Hand            | ✔        | ✔         | same |
|Claws               | :x:         | :x:          | doesn't stack with Mightly blow |
|Disturbing Presence | ✔        | :x:          | same |
|Extra Arms          | ✔        | ✔         | same |
|Foul Appearance     | ✔        | :x:          | same |
|Horns               | ✔        | :x:          | same |
|Iron Hard Skin      | :x:         | :x:          | new: Immune to Claw |
|Monstrous Mouth     | :x:         | :x:          | new: Catch re-roll and immune to Strip Ball |
|Prehensile Tail     | :x:         | :x:          | works on Leap and Jump |
|Tentacles           | :x:         | :x:          | success determined as D6 +own St -opp ST > 5 |
|Two Heads           | ✔        | :x:          | same |
|Very Long Legs      | :x:         | :x:          | negative modifers for Jump and Leap (if player has skill) reduce by 1, to minimum of -1, Immune to Cloud Burster |

### Passing
| Skill          | Implemented | Has Test(s) | Change  |
|----------------|-------------|-------------|-------|
|Accurate        | :x:             | :x:          | Only quick pass and short pass |
|Connoneer       | :x:             | :x:          | as Accurate but on Long pass and Long Bomb |
|Cloud Burster   | :x:             | :x:          | New: choose if opposing coach shall re-roll a successful Interfere when throwing Long or Long Bomb  |
|Dump-Off        | ✔            | :x:          | same |
|Fumblerooskie   | :x:             | :x:          | new: leave ball in vacated square during movement. :x: dice involved |
|Hail Mary Pass  | :x:             | :x:          | tacklezones matter |
|Leader          | :x:             | :x:          | same |
|Nerves of Steel | ✔            | ✔         | same |
|On the Ball     | :x:             | :x:          | Kick off return and pass block combined |
|Pass            | ✔            | ✔         | same |
|Running Pass    | :x:             | :x:          | new: may continue moving after quick pass |
|Safe Pass       | :x:             | :x:          | Fumbled passes doesn't cause bounce nor turnover |


### Strength
| Skill             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-------------|-------|
|Arm Bar            | :x:          | :x:          | new |
|Brawler            | :x:          | :x:          | new |
|Break Tackle       | :x:          | :x:          | +2 on dodge if ST>4 else +1 once per turn  |
|Grab               | ✔         | :x:          | same |
|Guard              | :x:          | :x:          | works on fouls too |
|Juggernaut         | ✔         | :x:          | same |
|Mighty Blow +1     | :x:          | :x:          | doesn't work passively (e.g. attacker down), +X |
|Multiple Block     | :x:          | :x:          | same |
|Pile Driver        | :x:          | :x:          | As piling on but is evaluate as a foul |
|Stand Firm         | ✔         | :x:          | same |
|Strong Arm         | :x:          | :x:          | Only applicable for Throw Team-mate |
|Thick Skull        | ✔         | :x:          | same |

### Traits
| Skill             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-------------|-------|
|Animal Savagery    | :x:          | :x:          | new |
|Animosity          | :x:          | :x:          | same? |
|Always Hungry      | ✔         | ✔         | same |
|Ball & Chain       | :x:          | :x:          | ? |
|Bombardier         | :x:          | :x:          | ? |
|Bone Head          | ✔         | ✔         | same |
|Chainsaw           | :x:          | :x:          | ? |
|Decay              | :x:          | :x:          | ? |
|Hypnotic Gaze      | ✔         | ✔         | same |
|Kick Team-mate     | :x:          | :x:          | ? |
|Loner (+X)         | :x:          | :x:          | +X is new |
|No Hands           | ✔         | :x:          | same |
|Plague Ridden      | :x:          | :x:          | new |
|Pogo Stick         | :x:          | :x:          | ? |
|Projectile Vomit   | :x:          | :x:          | new  |
|Really Stupid      | ✔         | ✔         | same |
|Regeneration       | ✔         | ✔         | same |
|Right Stuff        | ✔         | ✔         | same? |
|Secret Weapon      | :x:          | :x:          | same |
|Stab               | ✔         | :x:          | same |
|Stunty             | :x:          | :x:          | :x: modifier for passing, but opponent gets +1 when interfering with pass from Stunty Player |
|Swarming           | :x:          | :x:          | new |
|Swoop              | :x:          | :x:          | ? |
|Take Root          | ✔         | ✔         | same |
|Titchy             | :x:          | :x:          | Same but also can never cause negative modifiers to opponent player's agility test (e.g. catching/throwing ball) |
|Throw Team-mate    | :x:          | :x:          | ? |
|Timmm-ber!         | ✔         | ✔         | same |
|Unchannelled Fury  | :x:          | :x:          | new |


## Passing

| Feature                | Implemented | Has Test(s) | Change  |
|------------------------|-------------|-------------|-------|
| Passing characteristic | :x:          | :x:          | AG does :x: longer determine passing ability |
| New distance modifiers | :x:          | :x:          | 0 for quick, ... , -3 for Long Bombs |
| Wildly inaccurate pass | :x:          | :x:          | Roll is 1 after modifiers. Deviates (like a kickoff) |
| Catch modifiers        | :x:          | :x:          | 0 for accurate |
| Pass Interference      | :x:          | :x:          | Replaces old interception rules |


## Other feature changes
Fame is removed and replaced with a similar 'Fan Factor' 


| Feature                 | Implemented | Has Test(s) | Change  |
|-------------------------|-------------|-------------|---------|
| Fan factor              | :x:          | :x:          | D3 + nbr of Dedicated Fans the team has |
| Team re-rolls           | :x:          | :x:          | Multiple team re-rolls can be used per turn |
| Player characteristic   | :x:          | :x:          | AG and AV is now e.g. 3+ instead of 4 |
| Passing characteristic  | :x:          | :x:          | new |
| Sweltering heat         | :x:          | :x:          | D3 players, randomly selected |
| Jump over prone players | :x:          | :x:          | New  |
| Niggling Injury         | :x:          | :x:          | +1 on Casualty roll instead |
| Casualty table          | :x:          | :x:          | D16 table  |
| Stunty Injury table     | :x:          | :x:          | 2-6 stunned, 7-8 KO, 9 Badly Hurt (without casualty roll), 10+ Casualty  |



## Races 

All races have changed. This table shows what is working. 

| Race             | Missing positions       | Missing skills                     | Have icons    |
|------------------|-------------------------|------------------------------------|----------------
| Amazon             |                         |                                    | ✔    |
| Black Orc          |                         |                                    | :x:   |
| Chaos Dwarf        |                         |                                    | ✔    |
| Chaos Choosen      |                         |                                    | ✔    |
| Chaos Renegades    |                         |                                    | Maybe? |
| Dark Elf           |                         |                                    | ✔    |
| Dwarf              |                         |                                    | :x:     |
| Elven Union        |                         |                                    | ✔    |
| Goblin             |                         |                                    | Some?  |
| Halfling           |                         |                                    | ✔    |
| High Elf           |                         |                                    | ✔    |
| Human**            |                         |                                    | ✔    |
| Imperial Nobility  |                         |                                    | :x:     |
| Tomb Kings         |                         |                                    | ✔    |
| Lizardmen          |                         |                                    | ✔    |
| Necromantic        |                         |                                    | :x:     |
| Norse              |                         |                                    | ✔    |
| Nurgle*            |                         |                                    | :x:     |
| Ogre               |                         |                                    | ✔    |
| Old World Alliance |                         |                                    | :x:    |
| Orc                |                         |                                    | ✔    |
| Shambling Undead   |                         |                                    | ✔    |
| Skaven             |                         |                                    | ✔    |
| Snotling           |                         |                                    | :x:    |
| Vampire            |                         |                                    | ✔    |
| Underworld Denizens|                         |                                    | Maybe? |         
| Wood Elf           |                         |                                    | ✔   | 

* Nurgle's Rot needs to be implemented in the post-game sequence
