# Sampling actions

To correctly sample actions for a tree searching bot we have to wrap the 
framework with some scripted behavior. We'll here develop a function that 
takes a game object and returns a list of actions. By return actions (instead of 
action choices) we can sample directly from the list and sent to the environment. 

We're gonna need our function to have an internal state, we make this explicit by 
making it a class. 
 

```python
import botbowl 
from botbowl import Action, ActionType
class ActionSampler: 
    """ 
    Example usage: 
    > get_actions = ActionSampler() 
    > actions = get_actions(game) 
    > action = random.choice(actions)
    > game.step(action)
    """
    
    def __init__(self):
        self.end_setup = False 
    def __call__(self, game: botbowl.Game) -> List[botbowl.Action]:
        actions = self.get_list_of_actions(game) 
        assert len(actions) > 0 
        return actions      
 
    def get_list_of_actions(self, game: botbowl.Game) -> List[botbowl.Action]:
        # main logic here
        ... 
```


First let's start with the basic functionality and some extra steps to handle the setup. 

 ```python

def get_list_of_actions(self, game: botbowl.Game) -> List[botbowl.Action]:
    action_types = {ac.action_type for ac in game.get_available_actions()}
    proc = game.get_procedure()
    actions = [] 
    if self.end_setup: 
        self.end_setup = False 
        return [Action(ActionType.END_SETUP)] 
    
    if ActionType.END_SETUP is action_types: 
        # perfect defence
        if isinstance(proc, botbowl.core.procedures.Setup) and proc.reorganise:
            # For now, we don't do anything in perfect defence 
            return [Action(ActionType.END_SETUP)] 

        self.end_setup = True 
        
        for ac in game.get_available_actions(): 
            if ac.action_type not in {ActionType.END_SETUP, ActionType.PLACE_PLAYER}}:
                actions.append(Action(ac.action_type)) 
        
        return actions
        
    # handle all other actions
    for ac in game.get_available_actions(): 
        if len(ac.positions) > 0: 
            for pos in ac.positions: 
                actions.append(Action(ac.action_type, position=pos))
        elif len(ac.players) > 0: 
            for player in ac.players: 
                actions.append(Action(ac.action_type, player=player))
        else: 
            actions.append(Action(ac.action_type))
    
    return actions 

```



