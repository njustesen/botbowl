# 2020 ruleset feature list

The purpose of this document gather the delta between the 2016 and 2020 rules, and showing the progress towards an implementation of the 2020 ruleset.  

## Terminology 
The 2020 rules coins a few new terms which are not rules but helps explain the rules.

| Term          | Meaning  |
|---------------|----------|
| Open player   | Player not in any opponent's tacklezone   |
| Marked player | Player in one or more of opponent's tacklezones  |
| Rush          | Rename of GFI |
| Deviate       | Moving the ball D6 squares in direction determined by a D8 (e.g. kickoff)   |
| Scatter       | Move ball three squares in direction determined by three subsequent D8 (e.g. inaccurate pass)|
| Bounce        | Move ball one square in direction determined by D8 (e.g. failing catch/pickup)  |
| Stalling      | Player can score with full certainty but chooses not to |



## Pregame 

| Thing             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-----------|-------|
| Fan factor        | No          | No       |  |
| Prayers to Nuffle | No          | No       | All new, see implementation status of each prayer below |


## Kickoff event table 
2020 rules introduces the term _marked_/_open_ for a player that is/isn't  in an opponent's tacklezone. The terms are used below   

| Feature                | Implemented | Has Test(s) | Change  |
|------------------------|-------------|-----------|-------|
| 2 - Get the ref        | N/A         | No       | Same |
| 3 - Time out           | No          | No       | Kicking team's turn marker on 6,7,8 then substract one turn, else add one. |
| 4 - Solid defence      | No          | No       | Limited to D3+3 players  |
| 5 - High Kick          | N/A         | ?        | Same |
| 6 - Cheering Fans      | No          | No       | D6+fan factor. Winner rolls a Player to Nuffle|
| 7 - Brilliant Coaching | N/A         | ?        | Same |
| 8 - Changing Weather   | N/A         | ?        | Same |
| 9 - Quicksnap          | No          | No       | Limited to D3+3 open players|
| 10 - Blitz             | No          | No       | D3+3 open players  |
| 11 - Officious Ref     | No          | No       | D6+fan factor to determine coach. Randomly select player. D6: 1: Sent to dungeon. 2+ Stunned |
| 12 - Pitch invasion!   | No          | No       | D6+fan factor to determine coach. Randomly select D3 players. Stunned  |



## Skills 


### Removed skills 
Skills that are completely removed are: Piling On, 

### Agility 
| Skill             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-------------|-------|
|Catch              | No          | No          | same |
|Diving Catch       | No          | No          | same |
|Diving Tackle      | No          | No          | works on leap and jump too |
|Dodge              | No          | No          | same |
|Defensive          | No          | No          | new |
|Jump Up            | No          | No          | same |
|Leap               | No          | No          | may Jump over any type of square, negative modifer reduce by 1, to minimum of -1 |
|Safe Pair of Hands | No          | No          | new |
|Sidestep           | No          | No          | same |
|Sneaky Git         | No          | No          | not sent of because of doubles on armor even if it breaks, may move after the foul |
|Sprint             | No          | No          | same |
|Sure Feet          | No          | No          | same |

### General
| Skill             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-------------|-------|
|Block             | No          | No           | same |
|Dauntless         | No          | No           | same |
|Dirty player (+1) | No          | No           | same |
|Fend              | No          | No           | same |
|Frenzy            | No          | No           | same |
|Kick              | No          | No           | same |
|Pro               | No          | No           | +3 instead of +4. May only re-roll one dice |
|Shadowing         | No          | No           | success determined by D6 +own MA -opp MA > 5  |
|Strip Ball        | No          | No           | same |
|Sure Hands        | No          | No           | same |
|Tackle            | No          | No           | same |
|Wrestle           | No          | No           | same |

### Mutations
| Skill             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-------------|-------|
|Big Hand            | No         | No          | same |
|Claws               | No         | No          | doesn't stack with Mightly blow |
|Disturbing Presence | No         | No          | same |
|Extra Arms          | No         | No          | same |
|Foul Appearance     | No         | No          | same |
|Horns               | No         | No          | same |
|Iron Hard Skin      | No         | No          | new: Immune to Claw |
|Monstrous Mouth     | No         | No          | new: Catch re-roll and immune to Strip Ball |
|Prehensile Tail     | No         | No          | works on Leap and Jump |
|Tentacles           | No         | No          | success determined as D6 +own St -opp ST > 5 |
|Two Heads           | No         | No          | same |
|Very Long Legs      | No         | No          | negative modifers for Jump and Leap (if player has skill) reduce by 1, to minimum of -1, Immune to Cloud Burster |

### Passing
| Skill             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-------------|-------|
|Accurate        | No             | No          | Only quick pass and short pass |
|Connoneer       | No             | No          | as Accurate but on Long pass and Long Bomb |
|Cloud Burster   | No             | No          | New: choose if opposing coach shall re-roll a successful Interfere when throwing Long or Long Bomb  |
|Dump-Off        | No             | No          | same |
|Fumblerooskie   | No             | No          | new: leave ball in vacated square during movement. No dice involved |
|Hail Mary Pass  | No             | No          | tacklezones matter |
|Leader          | No             | No          | same |
|Nerves of Steel | No             | No          | same |
|On the Ball     | No             | No          | Kick off return and pass block combined |
|Pass            | No             | No          | same |
|Running Pass    | No             | No          | new: may continue moving after quick pass |
|Safe Pass       | No             | No          | Fumbled passes doesn't cause bounce nor turnover |


### Strength
| Skill             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-----------|-------|
|Arm Bar            | No          | No        | new |
|Brawler            | No          | No        | new |
|Break Tackle       | No          | No        | +2 on dodge if ST>4 else +1 once per turn  |
|Grab               | No          | No        | same |
|Guard              | No          | No        | works on fouls too |
|Juggernaut         | No          | No        | same |
|Mighty Blow +1     | No          | No        | doesn't work passively (e.g. attacker down), +X |
|Multiple Block     | No          | No        | same |
|Pile Driver        | No          | No        | As piling on but is evaluate as a foul |
|Stand Firm         | No          | No        | same |
|Strong Arm         | No          | No        | Only applicable for Throw Team-mate |
|Thick Skull        | No          | No        | same |

### Traits
| Skill             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-------------|-------|
|Animal Savagery    | No          | No          |  |
|Animosity          | No          | No          |  |
|Always Hungry      | No          | No          |  |
|Ball & Chain       | No          | No          |  |
|Bombardier         | No          | No          |  |
|Bone Head          | No          | No          |  |
|Chainsaw           | No          | No          |  |
|Decay              | No          | No          |  |
|Hypnotic Gaze      | No          | No          |  |
|Kick Team-mate     | No          | No          |  |
|Loner (+X)         | No          | No          |  |
|No Hands           | No          | No          |  |
|Plague Ridden      | No          | No          |  |
|Pogo Stick         | No          | No          |  |
|Projectile Vomit   | No          | No          |  |
|Really Stupid      | No          | No          |  |
|Regeneration       | No          | No          |  |
|Right Stuff        | No          | No          |  |
|Secret Weapon      | No          | No          |  |
|Stab               | No          | No          |  |
|Stunty             | No          | No          |  |
|Swarming           | No          | No          |  |
|Swoop              | No          | No          |  |
|Take Root          | No          | No          |  |
|Titchy             | No          | No          |  |
|Throw Team-mate    | No          | No          |  |
|Timmm-ber!         | No          | No          |  |
|Unchannelled Fury  | No          | No          |  |


## Passing
Passing has changed, a lot! 

| Feature                | Implemented | Has Test(s) | Change  |
|------------------------|-------------|-------------|-------|
| Passing ability        | No          | No          | AG does no longer determine passing ability |
| New distance modifiers | No          | No          | 0 for quick, ... , -3 for Long Bombs |
| Wildlt inaccurate pass | No          | No          | Roll is 1 after modifiers. Deviates (like a kickoff) |
| Catch modifiers        | No          | No          | 0 for accurate |
| Pass Interference      | No          | No          | Replaces old interception rules |


## Other feature changes

| Feature                 | Implemented | Has Test(s) | Change  |
|-------------------------|-------------|-------------|-------|
|Team re-rolls            | No          | No          | Multiple team re-rolls can be used per turn |
|Player characteristic    | No          | No          | AG and AV is now e.g. 3+ instead of 4 |
|Passing characteristic   | No          | No          | new |
|Sweltering heat          | No          | No          | D3 players, randomly selected |
|Jump over prone players  | No          | No          | New  |




## Races (TODO!!)

All races have changed. This table shows what is working. 

| Race                   | Missing positions       | Missing skills                     | Have icons    |
|------------------------|-------------------------|------------------------------------|----------------
| **Amazon**        |                         |                                    | YES           |
| **Bretonnia**     |                         |                                    | YES           |
| **Chaos**         |                         |                                    | YES           |
| **Chaos Dwarf**   |                         |                                    | YES           |
| **Dark Elf**      |                         |                                    | YES           |
| Dwarf Slayer      |                         |                                    | NO            |
| **Elven Union**   |                         |                                    | YES           |
| Goblin            | Pogoer, Fanatic, Looney, Bomma, Doom Diver, 'Ooligan        | Bombardier, Secret Weapon, Chainsaw, Ball & Chain, Swoop, Fan Favorite | SOME?      |
| Halfling          | Treeman                 | Timmm-ber! | YES     |
| **High Elf**      |                         |                                    | YES              |
| **Human**         |                         |                                    | YES           |
| Human Nobility    |                         |                                    | NO            |
| **Khemri**        |                         |                                    | YES           |
| Khorne            |                         |                                    | NO           |  
| **Lizardmen**     |                         |                                    | YES           |
| Necromantic       |                         |                                    | NO           |
| **Norse**         |                         |                                    | YES        |
| **Nurgle***       |                         |                                    | NO            |
| **Orc**           |                         |                                    | YES           |
| Ogre              | Snotling, Ogre          | Titchy,                            | YES           |
| **Skaven**        |                         |                                    | YES           |
| Savage Orc        |                         |                                    | NO            |
| Skaven: Pestilent Vermin | Novitiates, Pox-flingers, Poison-keepers | Bombardier, Secret Weapon  | NO        |
| **Undead**        |                         |                                    | YES
| **Vampire**       |                         |                                    | YES
| **Wood Elf**      |                         |                                    | YES              |
| Chaos Renegades   | Goblin Renegade, Orc Renegade, Skaven Renegade, Dark Elf Renegade, Chaos Troll, Chaos Ogre | Animosity | MAYBE?
| Slann             |                         |                                    | NO              |         
| Underworld        | Warpstone Troll, Underworld Goblins, Skaven Linemen, Skaven Throwers, Skaven Blitzers | Animosity | MAYBE?              |         

* Nurgle's Rot needs to be implemented in the post-game sequence