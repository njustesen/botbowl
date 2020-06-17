# Submit Your Bot to Blood Bowl

## Bots as Servers

We will host your bot somewhere in the cloud on its very own instance. This is done by applying a wrapper around your bot that 
is responsible for the communication with the competition server. You don't need to making any changes to you bot since it won't 
be aware of any this. However, you must use this [submission template](https://github.com/njustesen/bot-bowl-submission) and 
modify the run.py script so that it imports your bot. Simply replace the line ```import mybot``` with ```import <your-bot-script>```.

In your bot script, make sure your bot is registered with a suitable name for your bot:
```python
# Register the bot to the framework
ffai.register_bot('my-random-bot', MyRandomBot)
```

In Bot Bowl II, every bot plays ten games against each other bot. During each series of ten games, the two competing bots will 
be instantiated just once, then play the ten games of the series, and finally shut down. Before each new game in the series, the ```new_game()``` method is called and at the end of each game, ```end_game()``` is called. This allows you to easily adapt 
your strategy based on the game result. It is, however, not possible to save data between the series but this is hardly useful anyways. 

## Submission link

Coming soon
