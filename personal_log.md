## Timed Take-home description
Task: On a single h100, get the lowest fineweb validation loss possible in 5 mins of training time. No changes to the data allowed. No looking at the future versions of the repo readme or other nanogpt speedrun records beyond the current commit (commit 7), but you can use everything else in the ML literature.

Part of the interview will involve a presentation on your work and answering questiongs about the code in your solution. So you must make sure to understand your code well, even if you are using AI.

Time: 2 hours

Hardware: 4xH100 (so you can run up to 4 experiments in parallel)


## Personal Log
9:15 am: Start. I have 2 hours until 11:15 am. I am doing a timed takehome where I have 2 hours to solve the problem.

9:21 am: Coding tools setup and basic environment setup. Let's plan things out a bit first.

So my goal is to get the lowest loss I can in a 5 minute single gpu run.

Basic infra I might want:
- I probably want the main run to have some kind of queing system since I have 4 gpus and may want to run larger hyperparameter searh or kick things off and stuff like that
- For the queing system, i want to be able to override, delete, etc. How should we represent hte queue? Maybe just as a file?
- In order to do orchestration, one idea is that every time I kick off a run, it saves the code for hte file. Let's keep everything single file for now to keep it simple. This sounds great for now. So the main thing I'll need is the queing system. And I would love for ongoing runs to also be figure outable, like we want to know for ongoing runs which logfile they are currently writing to.

9:33 am: I'm 15 minutes in. I am still figuring out infra. I can't spend more than 15 minutes more on this. I think I'm actually going to go to a 4 gpu parallel per run strategy so I can iterate faster, ideally every 1.5 minutes. I am also having claude code implement a queing system. The queing system is the first part I don't fully understand yet, so let's make sure I do. Added the teach skill to help here.

9:45 am: Unfortunately, I previously started the wrong runpod instance and it only had 1 gpu. Just finished redoing setup on the 4 gpu machine.

10:00 am: Set learning rate schedule to be a funciton of time. Time is 5 min / number of gpus. Decreased train and vlaidation dataset size since short runs and dont want dataloading to take forever.

Currently waiting for CC to finish somei infrastructure, but then will do a baseline run.Then I will do sme LR tuning.

10:23 am: I spent a few minutes talking to Aayush about strategy. Now back to it, kicked off a run with higher learning rate. First goal is to spend some time tuning that.

10:36 am: Interestingly, I am finding that the higher learning rate is not doing better so far. I will do some binary search to find the best one.

10:54 am: Checked in with Aayush. I am now getting some benefits to the learning rate tuning and found 0.003 seemed to work better than what I had.

11:02 am: Looks like the smalelr batch size is definitely helping. We can continue to do binary search on batch size for a while.