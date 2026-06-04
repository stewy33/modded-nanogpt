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

4:50 pm: Now using flex attention with document boundaries. This was a no-brainer. Also doing a run with perfetto traces with pytorch profiler logging. I notice not that much of the GPU's memory is being used. Everything is so small here this might just be how it is. A larger batch size doesn't help. But I do wonder if we have extra memory if there are ways we can trade off that fact for more speed.


5:10 pm: I'm 7 minutes short of the hour. My current priorities are to maybe tune the bs 64 a bit more on one gpu, figure out why flexattention isn't helping.

And then kick off one expeirment based off of the performance study I saw.

5:19 pm: I have a little less than an hour left. Let me think about hwat I'm doing. There is more tuning going on, but I need to move to adam vs muon tuning soon. There is also flex attention that I need to resolve. And ns steps tuning (easy win).

The next big thing to do is the unembedding matrix. Hw do we address that? I think we could avoid materializing it in some way?

5:40 pm: I have 35 minutes left. I see flex attention helped once I compiled the block mask thing to prevent the slowdown. It seems like increasing neuton schulz iterations also helps a bit? Concerned this might interact with other hps badly but maybe okay for now? Very small improvements but it is around 0.005. Flex attention did like 0.01.

I am next using a faster kernel for hte lm head, which should help a lot. Holding off fancier tricks to deal with it. Looking into any other addition clear kernsl or dhitngs i shoudl be using.

I thought the kernel was faster but CCE (Cut Cross Entropy) actually just saves memory by avoiding materializing some large thing, by later recomputing it. So it saves memory but makes things slower, which is not what we want at such low batch size. So the last thing I'm running is the fp8 head.

Finally, I'm taking 5 minutes with the /teach skill to try to understand the code. I think I will be over 2 hours though, but I lost the timer.

Things I would do if I had more time
- Understand the code better
- See if there are any no-brainer kernel improvements or data loading improvements that don't change the logic of the code, just make it faster. In particular, we're in a weird regime where we have lots of free memory and just want to avoid any extra computations that we could, so maybe there are special kernels for this?
- Get fp8 working properly (I suspect there might be instability if I just do it straight on the llm head). I would also want to do fp8 training to the mlps or the entire network. If it was unstable I suspect I'd need to 1) keep activations and certain weights in bf16 and keep only some in fp8, but I don't know the convention here 2) consuider using softcap or something similar to prevent magnitudes from growing too large, especially in the LM head

Takeaways to understand better in the future
- How pytorch DDP works and alternatives, i.e. does every process run the exact same python code just with a different environment variable? How does `synchronize` work and are there other primitives I should know about?
- How and why does Muon work? And intuitively, why is it not good for the lm head?
- What is the right way to set the sequence length here for training? Why don't we instead just traing with batch size of 1, and have a max cutoff length?
- How sensitive is the lm head and why is it so much more sensitive than other parameters? And how to deal with this if we want to quantize?
- During quantization, how can I tell if I'm having issues with precision or dynamic range? Is there a principled approach besides retraining?
- Meta-level
  - Think clearly about infra for running lots of small parallel experiments. I think there are reusable patterns here that work across lots of problems with that same shape. Then there are other problems with a different shape, where you don't get many iterations. I should practice these as well to come up with a good workflow.
  - Just having multiple Claude Code sessions in different terminal tabs is not that bad. But I can probably optimize further.
  - Asking the AI, "given my constraints and objective, is this a good idea?" would've caught the CCE issue and saved 10 minutes. Similarly if I asked this about fp8 head, it probably could've done a better yolo run for numerical stability than what I did.
  - It was easy for the AI to write code without bugs in this case for the most part. On problems where that is not the case, I probably would need to trust the AI much less.
  - The /teach skill is not that amazing yet. I think I kind of need to go through the code manually and have the AI teach me for now.