# 2020 ruleset feature list

The purpose of this document gather the delta between the 2016 and 2020 rules, and showing the progress towards an implementation of the 2020 ruleset.  

## Terminology 
The 2020 rules coins a few new terms which are not rules but helps explain the rules.
| Term             | Meaning  |
|-------------------|-----|
| Open player         |  |
| Marked player  |  |
| Rush  |  |
| Deviate  |  |
| Bounce  |  |
| Scatter  |  |



## Pregame 

| Skill             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-----------|-------|
| Fan factor           | No          | No       |  |
| Prayers to Nuffle           | No          | No       | All new, see implementation status of each prayer below |
|             | No          | No       |  |
|          | No          | No       |  |


## Kickoff event table 
2020 rules introduces the term _marked_/_open_ for a player that is/isn't  in an opponent's tacklezone. The terms are used below   

| Feature             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-----------|-------|
| 2 - Get the ref      | N/A          | No       | Same |
| 3 - Time out         | No          | No       | Kicking team's turn marker on 6,7,8 then substract one turn, else add one. |
| 4 - Solid defence    | No          | No       | Limited to D3+3 players  |
| 5 - High Kick        | N/A          |        | Same |
| 6 - Cheering Fans     | No         | No       | D6+fan factor. Winner rolls a Player to Nuffle|
| 7 - Brilliant Coaching | N/A        | No       | Same |
| 8 - Changing Weather  | N/A         | No       | Same |
| 9 - Quicksnap        | No          | No       | Limited to D3+3 open players|
| 10 - Blitz           | No          | No       | D3+3 open players  |
| 11 - Officious Ref   | No          | No       | D6+fan factor to determine coach. Randomly select player. D6: 1: Sent to dungeon. 2+ Stunned |
| 12 - Pitch invasion! | No          | No       | D6+fan factor to determine coach. Randomly select D3 players. Stunned  |



## Skills 
This is a table of new or modified skills

| Skill             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-----------|-------|
|           | No          | No       |  |



## Passing
Passing has changed, a lot! 

| Feature             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-----------|-------|
| Passing ability          | No          | No       | AG does no longer determine passing ability |
| New distance modifiers          | No          | No       | 0 for quick, ... , -3 for Long Bombs |
| Wildlt inaccurate pass           | No          | No       | Roll is 1 after modifiers. Deviates (like a kickoff) |
| Catch modifiers          | No          | No       | 0 for accurate |
| Pass Interference          | No          | No       |  Replaces old interception rules |
|           | No          | No       |  |
|           | No          | No       |  |



## Other feature changes

| Feature             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-----------|-------|
|Team re-rolls           | No         | No       | Multiple team re-rolls can be used per turn |
|Player characteristic           | No         | No       | AG and AV is now e.g. 3+ instead of 4 |
|Passing ability  | No         | No       |  new |
|            | No          | No       |  |
| Sweltering heat          | No         | No       | D3 randomly selected |


| Jump over prone players         | No         | No       | New  |




## Races

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

## Skills

| Skill             | Implemented | Has Test(s) |
|-------------------|-------------|-----------|
| **Accurate**      | YES         | YES       |
| **Always Hungry** | YES         | YES       |
| Animosity         | NO          | NO        |
| Ball & Chain      | PARTIALLY   | NO        |
| **Big Hand**      | YES         | YES       |
| **Block**         | YES         | YES       |
| **Blood Lust**    | YES         | YES       |
| Bombardier        | YES         | NO        |
| **Bone-head**     | YES         | YES       |
| **Break Tackle**  | YES         | YES       |
| **Catch**         | YES         | YES       |
| Chainsaw          | NO          | NO        |
| Claws             | YES         | NO        |
| Dauntless         | YES         | NO        |
| **Decay**         | YES         | YES       |
| Dirty Player      | YES         | NO        |
| Disturbing Presence | YES       | NO        |
| **Diving Catch**  | YES         | YES       |
| **Diving Tackle** | YES         | YES       |
| **Dodge**         | YES         | YES       |
| Dump-Off          | YES         | NO        |
| **Extra Arms**    | YES         | YES       |
| Fan Favorite      | NO          | NO        |
| Fend              | YES         | NO        |
| Filthy Rich       | NO          | NO        |
| Foul Appearance   | YES         | NO        |
| **Frenzy**        | YES         | YES       |
| Grab              | YES         | NO        |
| Guard             | YES         | NO        |
| Hail Mary Pass    | YES         | NO        |
| Horns             | YES         | NO        |
| **Hypnotic Gaze** | YES         | YES       |
| Juggernaut        | YES         | NO        |
| Jump Up           | YES         | NO        |
| Kick              | NO          | NO        |
| Kick Team-Mate    | NO          | NO        |
| Kick-Off Return   | NO          | NO        |
| Leader            | NO          | NO        |
| **Leap**          | YES         | YES       |
| Loner             | YES         | NO        |
| Mighty Blow       | YES         | NO        |
| Multiple Block    | NO          | NO        |
| Monstrous Mouth   | NO          | NO        |
| **Nerves of Steel**| YES         | YES       |
| No Hands          | YES         | NO        |
| Nurgle's Rot      | NO          | NO        |
| **Pass**          | YES         | YES       |
| Pass Block        | NO          | NO        |
| Piling On         | NO          | NO        |
| Prehensile Tail   | YES         | NO        |
| Pro               | YES         | NO        |
| **Really Stupid** | YES         | YES       |
| **Regeneration**  | YES         | YES       |
| **Right Stuff**   | YES         | YES       |
| **Safe Throw**    | YES         | YES       |
| Secret Weapon     | NO          | NO        |
| Shadowing         | YES         | NO        |
| Side Step         | YES         | NO        |
| Sneaky Git        | YES         | NO        |
| Sprint            | YES         | NO        |
| Stab              | YES         | NO        |
| Stakes            | YES         | NO        |
| Stand Firm        | YES         | NO        |
| **Strip Ball**    | YES         | YES       |
| **Strong Arm**    | YES         | YES       |
| Stunty            | YES         | NO        |
| Sure Feet         | YES         | NO        |
| **Sure Hands**    | YES         | YES       |
| Swoop             | NO          | NO        |
| Tackle            | YES         | NO        |
| **Take Root**     | YES         | YES       |
| Tentacles         | YES         | NO        |
| Thick Skull       | YES         | NO        |
| Throw Team-Mate   | NO          | NO        |
| **Timmm-ber!**    | YES         | YES       |
| Titchy            | PARTIALLY   | NO        |
| Two Heads         | YES         | NO        |
| **Very Long Legs**| YES         | YES       |
| Weeping Dagger    | NO          | NO        |
| **Wild Animals**  | YES         | YES       |
| Wrestle           | YES         | NO        |

## Pre-match Sequence

| Step                    | Implemented | Has Test(s) |
|-------------------------|-------------|-------------|
| **1. Weather Table**    | YES         | YES         |
| 2. Stadium              | NO          | NO          |
| 3. Referee              | NO          | NO          |
| 4. Inducements          | NO          | NO          |
| 5. Special Play Cards   | NO          | NO          |
| 6. Referee              | NO          | NO          |
| **7. Coin Toss**        | YES         | YES         |
| **8. Fans and Fame**    | YES         | YES         |

## Kick-off Table

| Kick-off Event            | Implemented | Has Test(s) |
|---------------------------|-------------|-------------|
| **1. Weather Table**      | YES         | YES         |
| 2. Stadium                | NO          | NO          |
| 3. Referee                | NO          | NO          |
| 4. Inducements            | NO          | NO          |
| 5. Special Play Cards     | NO          | NO          |
| 6. Referee                | NO          | NO          |
| **7. Coin Toss**          | YES         | YES         |

## Post-match Sequence

| Step                      | Implemented | Has Test(s) |
|---------------------------|-------------|-------------|
| 1. Improvement Rolls      | NO          | NO          |
| 2. Report Game Result     | NO          | NO          |
| 3. Fortune and FAME       | NO          | NO          |
| 4. Hire and Fire          | NO          | NO          |
| 5. Special Play Cards     | NO          | NO          |
| 6. Prepare for Next Match | NO          | NO          |

Some of these steps should be done in a league manager rather than in FFAI if humans are playing.

## Coaching Staff
| Staff                     | Implemented | Has Test(s) |
|---------------------------|-------------|-------------|
| 1. Head Coach (Argue the call) | NO     | NO          |
| 2. Assistant Coaches      | YES         | YES         |
| 3. Cheerleaders           | YES         | YES         |
| 4. Apothecary             | YES         | NO          |

## Star Players
Star players are not supported - mostly because the pre-match sequence isn't implemented yet.
