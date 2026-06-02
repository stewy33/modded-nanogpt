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

9:33 am: I'm 15 minutes in. I am still figuring out infra. I can't spend more than 15 minutes more on this. I think I'm actually going to go to a 4 gpu parallel per run strategy so I can iterate faster, ideally every 1.5 minutes. I am also having claude code implement a queing system.G The queing system is the first part I don't fully understand yet, so let's make sure I do. Added the teach skill to help here.

9:45 am: Unfortunately, I previously started the wrong runpod instance and it only had 1 gpu. Just finished redoing setup on the 4 gpu machine.

10:00 am: Set learning rate schedule to be a funciton of time. Time is 5 min / number of gpus. Decreased train and vlaidation dataset size since short runs and dont want dataloading to take forever.

Currently waiting for CC to finish somei infrastructure, but then will do a baseline run.Then I will do sme LR tuning.

10:23 am: I spent a few minutes talking to Aayush about strategy. Now back to it, kicked off a run with higher learning rate. First goal is to spend some time tuning that.

10:36 am: Interestingly, I am finding that the higher learning rate is not doing better so far. I will do some binary search to find the best one.

10:54 am: Checked in with Aayush. I am now getting some benefits to the learning rate tuning and found 0.003 seemed to work better than what I had.

11:02 am: Looks like the smalelr batch size is definitely helping. We can continue to do binary search on batch size for a while. Batch size 128 does even better than 256 or 512, bu tonly marginally better than 256. Trying 64 now, robably won't work well here. Tuned lr by sqrt of batch size. Will do additional tunin gafter.

Did more experiments and wrote up a short report of what I did.

===== 2.5 hour mark =====

4:22 pm: Took a break. Now I will do another 2.5 hours. I will build on top of this, so the task continues the same but I have an additional 2.5 hour allocation. I can now get to more researchy stuff. However, it may still be worthwhile to do additional tuning, not sure.

Overall, I'm seeing agents are good at autonomously doing hyperparameter tuning. So I will let them run their separate single gpu rusn to do that. I'm giving claude code 2 gpus to do batch size and LR tuning and I'll use the other 2 for more creative things.

Claude code thinks we should consider tuning:
1. Optimizer — cheapest, highest-confidence wins, and you're already here.
  - The Muon:AdamW LR ratio is hardcoded at 0.1*lr (line 436). This is your "V1 vs RMW" — almost certainly worth decoupling
  into two independent LRs and sweeping. The 0.1 is inherited, not tuned for your batch size.
  - Muon momentum=0.95, nesterov=True, backend_steps=5. Fewer NS steps = faster/step but worse orthogonalization — a direct
  throughput↔quality knob.
  - AdamW betas (0.9, 0.95). In a ~few-hundred-step run, beta2 warmup matters a lot — second moment barely converges.
  - weight_decay=0 — correct for this regime, low priority.
Overall, I think this could be good to keep doing in the first window that's doing HP tuning. I think the LR ratio thing is worth trying. Then omentum and adamw betas? I guess I'd want to see if our steps stay too small at first in some way. we could log athat as a statistic as well. But do that aft erh tebatch size and LR tuning happens.

4:37 pm: Starting with some profiling now. There is a wait and warmup period for the profiler over the first few steps so we don't get the wrong kind of timing.