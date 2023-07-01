# Submit Your Bot to Bot Bowl IV

## Bots as Servers

You must use the [submission template](https://github.com/njustesen/bot-bowl-submission) and 
modify the run.py script so that it imports your own bot. Simply replace the line ```import mybot``` with ```import <your-bot-script>```.

In your bot script, make sure your bot is registered with a suitable name for your bot:
```python
# Register the bot to the framework
botbowl.register_bot('my-random-bot', MyRandomBot)
```

In Bot Bowl IV, every bot plays a number of games against each other bot. During each series of ten games, the two competing bots will 
be instantiated just once, play the ten games of the series, and finally shut down. Before each new game in the series, the ```new_game()``` method is called and at the end of each game, ```end_game()``` is called. 
This allows you to easily adapt your strategy based on the game result without having to save and load files but it also means that if you bot crashes, it will stay unresponsive for the rest of the series. You are free to implement your own error handling if you believe your bot has a chance to crash.

If you want to understand how the competition server works in more detail, take a look at the [competition example](https://github.com/njustesen/botbowl/blob/main/examples/competition_example.py). I would recommend running your bot in this competition setting before submitting.
Note, that the competition code may change in the coming months.

## Submission link

Submit your bot [here](https://forms.gle/t5YQseRFMP76YVGBA). You can edit your submission up until the deadline if you want to make any changes.

## Verification
Within one week after the submission deadline, we will test your bot against the baseline random and contact you to verify that the results are as expected. 
In case we did not manage to set up your bot in the correct way, we will contact you for assistance. 
