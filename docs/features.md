# Features

botbowl currently supports the following features.
The purpose of this document is to have an overview of the road map for the project and to easily allow developers to 
identify missing features. If you want to help developing botbowl, the easiest way to get started is to implement or write 
a test for a skill.

## Game Modes

| Mode               | Implemented | Has Test(s) |
|--------------------|-------------|-------------|
| **AI vs. AI**      | YES         | YES         |
| **Human vs. AI**   | YES         | N/A         |
| **Hotseat**        | YES         | N/A         |
| **Online play**    | YES         | N/A         |
| **OpenAI Gym**     | YES         | YES         |
| **Headless**       | YES         | YES         |
| **AI Competition** | Yes         | NO          |

## Integrations

| System             | Implemented | Has Test(s) |
|--------------------|-------------|-------------|
| **OBBLM**          | NO          | NO          |
| **FUMBBL**         | NO          | NO          |

We are actively working towards and integration between botbowl and OBBLM.

## Races

A race is supported if all positions can be used and all their starting skills are supported.

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
| **Always Hungry** | YES         | YES       |
| Animosity         | NO          | NO        |
| Ball & Chain      | PARTIALLY   | NO        |
| **Big Hand**      | YES         | YES       |
| **Block**         | YES         | YES       |
| **Blood Lust**    | YES         | YES       |
| Bombardier        | YES         | NO        |
| **Bone-head**     | YES         | YES       |
| **Break Tackle**  | YES         | YES       |
| **Catch**         | YES         | YES       |
| Chainsaw          | NO          | NO        |
| Claws             | YES         | NO        |
| Dauntless         | YES         | NO        |
| **Decay**         | YES         | YES       |
| Dirty Player      | YES         | NO        |
| Disturbing Presence | YES       | NO        |
| **Diving Catch**  | YES         | YES       |
| **Diving Tackle** | YES         | YES       |
| **Dodge**         | YES         | YES       |
| Dump-Off          | YES         | NO        |
| **Extra Arms**    | YES         | YES       |
| Fan Favorite      | NO          | NO        |
| Fend              | YES         | NO        |
| Filthy Rich       | NO          | NO        |
| Foul Appearance   | YES         | NO        |
| **Frenzy**        | YES         | YES       |
| Grab              | YES         | NO        |
| Guard             | YES         | NO        |
| Hail Mary Pass    | YES         | NO        |
| Horns             | YES         | NO        |
| **Hypnotic Gaze** | YES         | YES       |
| Juggernaut        | YES         | NO        |
| Jump Up           | YES         | NO        |
| Kick              | NO          | NO        |
| Kick Team-Mate    | NO          | NO        |
| Kick-Off Return   | NO          | NO        |
| Leader            | NO          | NO        |
| **Leap**          | YES         | YES       |
| Loner             | YES         | NO        |
| Mighty Blow       | YES         | NO        |
| Multiple Block    | NO          | NO        |
| Monstrous Mouth   | NO          | NO        |
| **Nerves of Steel**| YES         | YES       |
| No Hands          | YES         | NO        |
| Nurgle's Rot      | NO          | NO        |
| **Pass**          | YES         | YES       |
| Pass Block        | NO          | NO        |
| Piling On         | NO          | NO        |
| Prehensile Tail   | YES         | NO        |
| Pro               | YES         | NO        |
| **Really Stupid** | YES         | YES       |
| **Regeneration**  | YES         | YES       |
| **Right Stuff**   | YES         | YES       |
| **Safe Throw**    | YES         | YES       |
| Secret Weapon     | NO          | NO        |
| Shadowing         | YES         | NO        |
| Side Step         | YES         | NO        |
| Sneaky Git        | YES         | NO        |
| Sprint            | YES         | NO        |
| **Stab**          | YES         | YES       |
| Stakes            | YES         | NO        |
| Stand Firm        | YES         | NO        |
| **Strip Ball**    | YES         | YES       |
| **Strong Arm**    | YES         | YES       |
| Stunty            | YES         | NO        |
| Sure Feet         | YES         | NO        |
| **Sure Hands**    | YES         | YES       |
| Swoop             | NO          | NO        |
| Tackle            | YES         | NO        |
| **Take Root**     | YES         | YES       |
| Tentacles         | YES         | NO        |
| Thick Skull       | YES         | NO        |
| Throw Team-Mate   | NO          | NO        |
| **Timmm-ber!**    | YES         | YES       |
| Titchy            | PARTIALLY   | NO        |
| Two Heads         | YES         | NO        |
| **Very Long Legs**| YES         | YES       |
| Weeping Dagger    | NO          | NO        |
| **Wild Animals**  | YES         | YES       |
| Wrestle           | YES         | NO        |

## Pre-match Sequence

| Step                    | Implemented | Has Test(s) |
|-------------------------|-------------|-------------|
| **1. Weather Table**    | YES         | YES         |
| 2. Stadium              | NO          | NO          |
| 3. Referee              | NO          | NO          |
| 4. Inducements          | NO          | NO          |
| 5. Special Play Cards   | NO          | NO          |
| 6. Referee              | NO          | NO          |
| **7. Coin Toss**        | YES         | YES         |
| **8. Fans and Fame**    | YES         | YES         |

## Kick-off Table

| Kick-off Event            | Implemented | Has Test(s) |
|---------------------------|-------------|-------------|
| **1. Weather Table**      | YES         | YES         |
| 2. Stadium                | NO          | NO          |
| 3. Referee                | NO          | NO          |
| 4. Inducements            | NO          | NO          |
| 5. Special Play Cards     | NO          | NO          |
| 6. Referee                | NO          | NO          |
| **7. Coin Toss**          | YES         | YES         |

## Post-match Sequence

| Step                      | Implemented | Has Test(s) |
|---------------------------|-------------|-------------|
| 1. Improvement Rolls      | NO          | NO          |
| 2. Report Game Result     | NO          | NO          |
| 3. Fortune and FAME       | NO          | NO          |
| 4. Hire and Fire          | NO          | NO          |
| 5. Special Play Cards     | NO          | NO          |
| 6. Prepare for Next Match | NO          | NO          |

Some of these steps should be done in a league manager rather than in botbowl if humans are playing.

## Coaching Staff
| Staff                     | Implemented | Has Test(s) |
|---------------------------|-------------|-------------|
| 1. Head Coach (Argue the call) | NO     | NO          |
| 2. Assistant Coaches      | YES         | YES         |
| 3. Cheerleaders           | YES         | YES         |
| 4. Apothecary             | YES         | NO          |

## Star Players
Star players are not supported - mostly because the pre-match sequence isn't implemented yet.
