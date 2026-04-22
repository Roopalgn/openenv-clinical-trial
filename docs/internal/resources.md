Good. Um, okay. Well, uh, yes. Okay. So, can someone wake them up? Okay, we're
0:06
awake. We're awake. We're here. Um, so yeah. Uh, welcome everyone. This Oh, wow. There's like 127 people that are
0:12
here already, but very cool. Okay. Um, so so welcome everyone. Like today is a is a very special day. It's like I mean
0:18
it's sort of a disservice to call this a talk because it ended up turning into more of a like a mini uh conference. Um
0:26
so we have six speakers today uh that are all like some of the people I admire the most like doing very interesting
0:31
work in open source. Uh we have like Zach Zachw from PyTorch, Davidi also from PyTorch, Sonia Bhutani also
0:38
PyTorch, Lewis Tunstall from Hugging Face, Ben Burtonshaw from hugging face and Daniel from Unsllo. Um so I think I
0:45
counted uh basically I did this earlier. So there are seven basically many talks today. So Daniel and Lewis will be
0:52
talking like multiple times. Um, I expect this whole thing to run for about like two and a half hours to three
0:59
hours. So, you know, stick around, get your lunch, like come hang out. But yeah, like without further ado, I think
1:05
our first speaker is going to be uh Daniel. Daniel, please take it from here. Yeah, thank you. Thank you. Um, you
1:10
know, thanks everyone for joining. Um, yeah. So, like you know, our program, we can hear you.
1:16
Wait, can you hear me? Oh, we hear you now. Yes. Oh, wait. Can you hear me now?
1:22
Yes. Okay. Yeah. Okay, cool. Well, in terms of the program, um, so you know, there
1:27
is a lot today. Um, but yes, um, as Mark said, there will be all mini, um, you know, mini talks for everyone. Um, so
1:34
like, you know, it's not going to be boring. It's going to be very, very fun. Um, but like at the first step, you
1:39
know, I'm going to be talking more about like, you know, introduction and intuition for reinforcement learning. Um, so yeah, hello, I'm Daniel. Um, and
1:46
yeah, we allow you to like, you know, the what what is Unslaf? We allow you to train and run language models locally.
1:52
Definitely check us out on GitHub. Um yeah, but like today I'm going to talk about
1:57
reinforcement learning quick takes. Um mainly like you know I always have like questions what is reinforcement
2:03
learning? What can reinforcement learning do and what it can what it cannot do. Um so I broke it out I broke
2:08
it down into like different sections. Um and I'll go to like you know I'll just go through each one separately. Um and
2:14
each of them are related to each other. So they're not like um you know they're not like disparate. They're all
2:20
correlated with each other. So the first one is like you know reinforcement learning is essentially
2:25
efficient in context learning. Um so for example um you know because you guys are from GPU mode um you know like for
2:31
example we ask a language model to create a Python function to do fast
2:37
matrix multiplication. Um so like if you ask you know GBD5 or Gemini or you know
2:42
Claude to do this it should create some sort of function to generate you know some sort of like Python function to do
2:47
matrix multiplication. Um alo obviously you should not use you know you shouldn't probably do it in Python you
2:53
know obviously do CUDA or you know even like you know C++ for like CPU um CPU
2:58
type workloads um but for now let's just assume it's just a Python function um
3:03
and you know if you call GPT um you should probably give it more you know more constraints for example tell it to
3:10
place that function between triple back ticks like for example on the left um and yeah like after you call it like you
3:17
know for example if you dse one For example, you have like a think token, you know, I need to create a fast matrix
3:22
multiplication, blah blah blah blah blah, right? And it does output some sort of algorithm that does this matrix
3:27
multiplication. Um, yeah. And so you don't just call the language model once, you call it many,
3:33
many, many times to generate, you know, different algorithms to do matrix multiplication, right? You get your
3:38
first one, the second one, the third one, you get many, many, many different ones. And the goal um is to call the
3:45
language model so many times until you get the best one. Um that's that's kind of like how you do um how you get the
3:51
best matrix multiplication algorithm. And then after you generate all of these
3:57
examples, you you put this all of these algorithms through a RO environment,
4:02
some sort of verifier, some sort of tester to test if these implementations are good or bad. Um so you can't just
4:09
like you know take the language models generated code you know and just use it. You have to test if it actually works
4:15
you know how fast is it um is it like effective um you know are there bugs in
4:20
the implementation and then if you put this through a you know environment you the goal is you
4:27
want to like time you know is this slow is this fast um what is the error rate between the true matrix multiplication
4:34
algorithm between you know matrix A and B. Um so you need to like calculate all of this and then in the end you output
4:40
some sort of number right so like for example the first one is pretty good right torch matt mole like okay that
4:45
makes sense plus 10 um the second one is like a numpy product um which which also
4:51
works right a numpy dot um but the problem is it did not follow the instructions right so it didn't follow
4:56
the instructions we need to put put it between three back ticks right there is no back ticks over here so we minus five
5:02
for this right it doesn't actually follow the style and then the last one is definitely wrong. Um so we give a
5:07
score of minus 100. The reason is because you know the output is totally wrong plus you know 0.001 something like
5:14
this. Um so this is kind of how we evaluate these you know implementations um by just like shoving some sort of
5:21
like you know shoving through a RO environment um or a verifier or a tester
5:26
um and then you get out some sort of number and so like you know the way to score
5:32
this right after you score all of these you then do the prompt again in the
5:38
language model and shove you know place in you know the examples into the actual
5:43
prompt right so this is the Second prompt, you're going to ask the language model again. Um, and so previously, you
5:48
know, we said, create a Python function to do fast matrix multiplication. Um, and then you add something to the
5:54
prompt. You say, I created some for reference with these scores from minus 100, you know, the worst to 100, right?
6:00
And you put the good one, you put a bad one, and then you say, you know, your job is to make a better one and place it
6:05
between the triple back ticks. Um, and that's what you do to ask the language model.
6:12
And you would assume like you know this process will go on forever right you will call many many many GPUs you know
6:17
you you continuously call the lang language model multiple you know infinite times until your score
6:23
continuously goes up um and so that's why I like to call like you know reinforcement learning it's kind of like efficient um it's so this is an example
6:31
of in context learning where you call the language model multiple times to generate examples
6:36
um so you might be wondering okay this can go on forever um but you should employ like some sort pruning method.
6:43
Um, some sort of like, you know, should we don't like, you know, don't put every single example inside of here. Um, you
6:48
know, maybe remove some of the bad ones. Um, you know, you know, will this go on forever? Um, you know, can you make this
6:55
a very very very long prompt? That's the main question. Um and there is like some sort of you know long context benchmark
7:02
um called um I think long arena or long context benchmarks which show that generally models do better and better
7:08
over time um for long context but the accuracy is not you know it it does degrade over time. Um so even Gemini for
7:15
example you know has 1 million context lengths. Um at hard mode you know if you want to find eight needles it does
7:21
decrease accuracy to like 20%. Um and so yes over time these models will get better and better and better. Um but
7:27
remember the goal is we want the model um not to utilize all of this long context just to like solve one small
7:33
little problem like this algorithm. Um and so with reinforcement learning the trick is um reinforcement learning is
7:40
essentially efficient in context learning. You do the same approach as before. You call the model many many
7:46
times to generate the fast matrix multiplication algorithm. So you do the same as before. um you have an RO
7:52
environment or a tester to get like a you know reward for each of these um implementations. So these are the
7:58
numbers but then the trick is you reward every single line or every single token with
8:05
the same number that you find. Um for example the first example was plus 10. Um and so we reward every single token
8:12
or every single line with plus 10. Um and so that's the trick of reinforcement learning. You don't you don't just like
8:19
reward the whole thing like the whole implementation is plus 10. You reward every single token with plus 10. Um and
8:27
then every single token that was bad, you know, the bad example um you know the plus 0.00001
8:33
um you just reward everything, every single token, every single line as minus 100. Um and then the trick for
8:39
reinforcement learning is now we update all the weights with back propagation by using these numbers. Right? So this is a
8:45
good one, this is a bad one. Um and then we just update the weight of back propagation. And then the trick of this
8:50
is there is no more long context needed. You do not need to like shove in you know all of the examples into the
8:56
context. You just ask one question. You know how do we create a fast matrix
9:01
multiplication algorithm and you just let back propagation and gradient descent do the work. Um and so reinforcement learning essentially is a
9:07
more token efficient approach than in context learning. And you know normally
9:13
how I like to frame it is reinforcement learning is just patience is all you need. You just need to wait you know
9:18
just wait and wait and wait and wait and you will get a good result at the very end. So Daniel I had two questions on this.
9:25
So like I mean so so there is a simpler baseline right like the simpler baseline is you just have a base model and you do
9:33
like pass at K where K is like a very large number. Um this seems simpler. It
9:38
seems less efficient because like K is hard but like could you speak a bit more like what you mean when you say RL is more efficient because most people would
9:45
say it's not efficient because you only get one bit of information for a very long roll out. So could you sort of like
9:50
help me understand this intuition of efficiency better? Yes. So reinforcement learning is less
9:56
efficient in general than like for example than supervised fine tuning. Um because like for example over here right we just allocate everything as minus
10:02
100. It's probably not a good idea to like you know make everything as minus 100. For example, like you you get a
10:09
final reward and you just make every single token as minus 100. For example, this line, right? I need to create a
10:14
fast matrix multiplication, you know, algorithm or something. This probably is good, right? You should not reward this
10:20
as minus 100. It should be like plus, you know, 50 or something, right? Or like, you know, return a mat, you know,
10:26
a you know, the the symbol for matrix multiplication b that's that's correct, right? But the only problem line is this
10:32
line, right? The plus 0.001. Um, okay, put extra dot. Um but yes, that's the
10:37
problem line and that should be like minus 100. Um so in general reinforcement learning um it's not
10:43
efficient not because it's not efficient, it's because at the very end we reward every single token with the
10:49
same number. Um and so we could actually make this process much more efficient. Um I'll actually talk I'll talk about
10:55
this. Um it's called process supervision. Um yeah. Yeah. So so this is interesting. So like
11:00
I mean this is this is like u but this could work for SFT as well too, right?
11:06
Like if basically you could reformulate I don't know like because you could view this as a translation task from English
11:11
to a fast kernel code or you could view it as like well let me reward individual tokens and then basically this is like a
11:18
richer signal and I see okay yes correct yes you could pose reinforcement learning as just
11:24
supervised fine tuning. So for example, instead of like just predicting the next token, you can like change the change
11:30
the loss I guess um to like just I guess it's kind of like DPI for example, like you know what is good, what is bad um
11:36
but it's more like a dynamic version. Um so reinforceable learning you don't need it's like more like regression I guess like you don't you can p you're trying
11:42
to predict a number like reward um and then you're just updating the weights. Um but I will talk about process
11:48
supervision I think like in the next few slides. Um yeah. All right. So I I think someone's
11:53
already asking about GRPO, but I I assume you'll be talking about that. Yes, I will be talking. I'll let you keep going. Yes. Yes.
11:58
Okay. But yeah, like essentially I like to call like you know patience is all you need or like luck is all you need. Um at
12:05
the very very beginning reinforcement learning will get zero reward. It won't work. Um but then you just have to wait
12:11
and wait and wait and wait and wait um until until it will like you will get some reward. Um and that's the trick of
12:18
reinforcement learning. You just have to wait. Um and the next topic is like you know but
12:23
but you know you don't just want to wait forever right and so the goal is reinforceable learning is is inefficient
12:29
yes it's not very efficient because of this problem right you just allocate minus 100 to everything however it's not
12:35
done um so for example at the very beginning of training um assume all the probabilities are like equal probability
12:42
right so like you know one divided by whatever the token size is um so 100k for example right every single token is
12:48
equal probability to be shown. When you do reinforcement learning, you will get bad answers. And so the goal is you want
12:55
to penalize these bad answers. And so the probability of getting these answers becomes less. Um and then you want to
13:02
like increase the probability of the other answers. Um so the bad ones we reduce the probability. So this dent um
13:08
and then we increase the probability of good answers or like you know the answers we have not seen yet.
13:15
And so over time if you keep doing this process you will keep generating bad and bad and bad answers. And so the whole
13:21
landscape of the probability distribution will like essentially shift and change um with this approach and
13:27
with reinforcement learning. And then once you get a very good answer there is like maybe somewhere in this high dimensional probability distribution you
13:34
will increase that probability dramatically um and so the rest will like decrease. Um so this is kind of
13:40
this is kind of like how you visualize reinforcement learning. So essentially it's not like just you know you're like forcing the algorithm to like learn oh
13:47
okay you know everything is bad um it will just like forcibly not produce that one answer it essentially tries to like
13:53
change the loss it changes the probab probability distribution um for every single other answer um so that's why
13:59
like normally how I like to say is like reinforcement learning is not it's not dumb um it's just not very efficient um
14:06
but it but it works um it definitely works so the next point is like you know for
14:13
reinforcement learning to work the probability of a good answer must be greater than zero. Um if your answers
14:20
are always zero, right? So for example, if you keep waiting forever um but you
14:25
never get a reward, reinforcement learning will not work. Um so this is the most important factor for
14:30
reinforcement learning. It must have a probability of a good answer to be more than zero. Um it's this is the most I
14:37
would say this is like a law of reinforcement learning. Um and so like for example, there are many many problems that reinforcement learning can
14:43
um encounter that will you you might al always get zero reward. For example, your formatting might be wrong. Um, you
14:49
might have to do like some priming or warm up um to like, you know, force a reinforcement learning algorithm to generate the format. Um, you might have
14:57
to do some supervised finetuning. You might have to do some pre-training. Um, or the problem is just way too out of
15:03
distribution. So, all of these problems can come up. Um, and you just must you
15:08
just have to like I guess you have to pray that there is some probability of the good answer and it must be more than
15:15
zero. Um if it's zero then you just you would just be wasting compute and just waste you know just sitting there and
15:21
then reinforcement learning just does not work. Um okay so so Daniel I had a question for you then which is okay perfect no go to
15:26
the next slide it's actually oh okay yeah um okay so something I found very unpredictable is like let's say you're
15:33
like a middle class researcher you have like a bunch of GPUs maybe a couple of nodes uh and you're trying to figure out
15:40
like you have a task that you care about and you're trying to figure out like what's the best way to allocate your compute these days I found it kind of
15:46
nonobvious like how many GPUs should be reserved for synthetic data generation how much should you do for fine-tuning
15:52
how much should be for RL. It's not like entirely clear to me how to go about
15:58
this. So, do you have some intuitions that have helped you do this? Yes, that's a great question. Um, so I I
16:03
guess this plot kind of makes a bit like you know somewhat kind of like sums up your question like you know for example in the olden days you would do like
16:09
pre-training then you know SFG supervised finetuning then some post-training phase. This is like you
16:15
know the general workflow of everyone. So your question is more like you know what is the bucket size like you know
16:20
how do we allocate resources to which bucket? Um so I guess this one right so like you
16:25
could for example theoretically just start from a f random initialization of the model um and somehow like train the
16:32
model just doing reinforceable learning or some you know doing something to go to the final model. Um and so the
16:37
question is how do we actually allocate you know how do we allocate resources during this process. Um.
16:44
Mhm. And so in the olden days what you would do is you would do some pre-training um to you know go to this space. You would
16:50
do some supervised finetuning to go to over here. You do some preference finetuning and then you use like reinforcement learning to get your final
16:56
model. Um so this is kind of the process. Um and so the the question is like you know how much percentage do we
17:03
spend in each of these stages? Um and you know my point is like you know you should spend the majority of your time
17:09
in pre-training. um you should do fine-tuning, you know, some sort of like supervised supervised fine-tuning step
17:15
and then you should spend some resource of reinforcement learning. But in general, um the all of these steps are
17:20
very important. Um and then the go the reason why they're all very important is because you want to make the process
17:25
efficient um and not you know waste resources. Um for example, you could for
17:31
example bypass pre-training and just go through the RL model, right? So you don't need to do super supervised
17:37
fine-tuning, you don't need to do preference fine tuning, just skip everything. Um and for example, deep
17:42
speed zeros, something like this. Um and I actually do not suggest people to do this because this is actually very this
17:48
is very wasteful. Um and the goal for artificial intelligence, the goal for deep learning is you want to you know
17:54
efficiently allocate resources and and in my view all of machine learning is just efficiency, right? How do we make
18:00
how do we make you know everything more efficient? And so I would suggest people to like you know do all of these phases.
18:06
Um maybe like allocate 10% of computer RL 10% to super bias fine tuning and
18:12
then the rest for pre-training. Um so 80% 10 10. Um yeah I'm not sure if that kind of answers the question. Um
18:18
no no it does and I'm thank you for giving actual numbers. I think that like you're putting your mouth your money
18:23
where your mouth is there. So I think that's very helpful. Yes. Thank you. to be honest, it's more like 99% should be
18:29
pre-training [laughter] and then 1%, you know, 0.5 0.5. Um, however, you know, if
18:34
pre-training does the loss doesn't seem to be decreasing, then obviously you need to like, you know, then I guess you have to allocate more time. Um, but the
18:40
trick is like, you know, for example, when you do pre-training first, you can always expand, you know, increase the
18:45
time you want to do for reinforcement learning. Um, so it's more like you just have to do good pre-training and then you can spend more time for RL or SFT.
18:53
Um, yeah. Yeah. So like the next one is like yes
18:58
as you know Mark was talking about process supervision for example like you know RL is not efficient. The main problem is when you do reinforcement
19:05
learning you assign the reward minus 100 to every single token or every single line exactly the same. Um and that is a
19:12
problem for reinforcement learning. Um there is a method called process supervision where you you know manually
19:19
check you know which line is good which line is bad and then you allocate um different numbers for each of the lines
19:26
um for example you know these tokens the think token are whatever it's not that important so I just allocate you know
19:32
reward of zero um you know this in the prompt during the generation during inference phase um during the inference
19:38
step it does say I need to create a fast matrix multiplication algorithm so this is actually not that bad um and it does
19:44
follow the format. Um, and so we pass a few points, but then we do allocate minus 100 because it did a wrong, you
19:50
know, this this step was wrong. Um, and so like you could, this is called process supervision where you're like,
19:56
you know, look at the, you know, you you don't just allocate minus 100 to every single line or every single token. You
20:02
actually look at, you know, which token is good, which token is bad. Um, and so
20:08
how would you do this? Unfortunately, what people do now is you have to have like you know you will ask many many
20:13
many people to like you know look at the outputs you know manually you know humans have to manually label this um
20:19
and so there is a way you know instead of doing this instead of asking humans to label it why don't
20:26
you ask a language model to label it um so this is called like you know llmage or something like this where you just
20:32
call the language model you know to label this for you right so the prompt will be like you know to to like Chip or
20:39
Gemini or Claude, you know, you know, is this line good? Is this line good? Is this line good? Or, you know, is this
20:45
line good and relevant to the original question and you let you let the language model label these numbers. Um,
20:52
and so this is like, you know, this could work. Um, you just call the language model many many many times and
20:57
it will label this for you. But then we encounter a big problem. Um,
21:03
which is essentially you will you might fall into the trap of reward hacking. Um and so one of the biggest problems is if
21:09
you do this process supervision approach um you know you ask the language model many many times to label it um you might
21:15
also encounter the language model um not labeling correctly. Um for example um it
21:21
might actually cheat for example it might delete the timer in the you know in the function. It might edit A to be zero. It might edit B to be zero. You
21:28
know it might do many many weird things. Um and the language model might still label this as good. Um you actually
21:34
don't know what it will do. Um and so this is the one of the biggest problems of reinforcement learning and this is
21:40
actually this keeps most researchers awake at night like you know will will RL just go into the reward hacking trap.
21:47
Um and you know essentially RL will not work. Um and so this is you know this is a still an open question. Um another one
21:55
for example is corruption. Um where it will actually try to corrupt your computer for example it might you know
22:01
do rm-rf or something I don't know something like this. Um, and so one of the biggest problems is it might
22:06
actually edit, you know, your computer to try to, you know, cheat its way out.
22:12
Um, it might even say in the prompt like, you know, some weird formatting like, you know, something else inside the prompt. Um, and so this is one of
22:19
the problems for reinforcement learning if you want to automate it. Um, you know, it might reward wrong signals. Um
22:25
yeah but in terms of like different flavors of reinforcement learning you know at
22:30
the very very beginning stage we had PO which was you know I think this was invented by open AI researchers um and
22:37
so like you had you had many different types of models for PO yes sorry yeah yes yeah so sorry but before you go
22:43
there let's let's talk about hacking for a second I think the hacking point like has a sort of simple answer at small
22:49
scale which is like look at your data you're like oh like you're doing CUDA kernel codegen how many important
22:54
kernels are there like you know a few hundred look at them but like I really don't buy that like you know like something like clot code for instance
23:00
like for me it was very challenging to imagine that like you know Boris is like looking at every code snippet being
23:06
generated so h how do you end up making these things like reliable at scale is it kind
23:11
of like what you said just LLM is judge and you audit the LM as judge like or or kind of what do you suspect is the
23:17
secret sauce for a lot of these largecale audits of reward hacks? Yeah. So, LM as a judge can can be useful. So,
23:24
you can like ask a language model, you can ask a language model, you know, does this implementation look correct? Um,
23:31
you know, are there any bugs in this implementation? You can like, you know, essentially you can call a language model many many times and give like some
23:37
sort of number, right? So, for example, um you know this if you give this language model like you know does this
23:42
look correct? It will probably say no because it does something else. Um and so rewarded like minus 100. So yes, you
23:48
shouldn't you're correct in the normal days like you know if you have like few data you can you should manually inspect
23:54
it um but then you know because of reinforcement learning it automatically generates everything you don't want to
24:01
like you know it will be a very tiring job to inspect every single step you know every single data point um normally
24:06
how I would like to do it is like you know maybe do a sampling you know sample 1% every single 10 turns to see the
24:13
generation of your model um and then if it looks wrong I think you should just terminate the run um and see what
24:19
happened. Um so generally that's how people do it. You just like sample you look at the output every single like 10
24:24
turns. Um look at like look at look at like 10 examples. Um yeah. Uh so so like reward hacking is more of
24:31
a property of the model being dumb, not of the model being very clever would kind of be your thesis here, right?
24:37
I would Yeah. So reward hacking I wouldn't say the model's being smart.
24:43
I would say the model's just being maybe your instruct I would say actually the human who wrote the instruction is
24:50
probably incorrect. Um your instruction has to be more clear. Um for example like you know you you should like
24:55
mention you know do not do this or do not do that. Um but then I guess if you keep doing this
25:01
your system problem might be very long. Um that is also a problem. Um and so like I guess it's more like you know the human has to like guide the model the RL
25:08
process. Um yeah I mean you could imagine for example you could imagine but you know But but it's a positive
25:14
sign that we can detect this early is really your thesis. Like it's not something that's sort of emergent as the
25:20
model becomes superhuman near the end of a very expensive RL training run which would be very annoying. Okay, I
25:26
see. Yeah, definitely like you should most training runs you need to inspect it like you know every single 10 turns look
25:32
at the generation look at the inference output. Um don't just let it run forever. That's probably not a good idea. Um and then you know every single
25:39
10 turns you know does it look correct? um you know does it like you know I I
25:44
guess we still need humans to like inspect if the model actually is correct or wrong. Um and then if it looks wrong
25:50
just terminate it or like you know go back to the previous checkpoint um and you maybe edit it somehow or like delete
25:56
maybe another way is to like delete the bad responses like you know maybe add some sort of filter to like you know
26:01
guide the model's reinforcement learning process. Um yeah right let's keep going. Yeah thank you.
26:08
Yeah, but like in terms of PO like you know there are many different parts of PO. Um but then we had GRPO and
26:15
essentially GRPO is just a more efficient version of PO. Um essentially we got rid of the value model right
26:21
delete it. Um and so GRPO essentially just made it much more efficient to do
26:26
training. Um and then with reinforcement learning with verifiable rewards RLVR
26:31
instead of having a reward model. So the reward model actually had to predict, you know, what is the reward for this
26:37
specific turn. Um instead of that, we just had some sort of environment to
26:42
check, you know, to tell you if it's good or bad, right? So it's like some like tester, some sort of verifier. Um
26:48
and this is um you know, Gio with RLVR. Um Daniel, sorry interrupt you, but like
26:53
your other co-speakers have things they'd like to say, so I'm adding David and Sonia to the chat.
26:58
Oh, okay. Yeah. Oh, hi. Hello. [laughter]
27:07
Now you can go first. Huh? You can go first, please.
27:12
Um, well, I don't know. We kind of moved on from the topic. I was um maybe I could add some color on the reward
27:18
hacking part. Oh, yes. I Yeah, we can go back to that. Yeah. So, basically, like another way of
27:24
seeing this uh sometimes reward hacking kind of surprises you. Um like for
27:29
example uh when we when we work on llama uh we had
27:35
a model that was a very good model um that surprisingly was very laconic like
27:41
uh it didn't seem like it wanted to talk to you very much um but it tended to be quite accurate in benchmarks and uh we
27:47
ended up not not shipping that model although it's you know its benchmark score were a bit higher than the ones we ended up choosing um but so in general I
27:55
think that one way of seeing this what's happening here is that um unlike supervised fine tuning in reinforcement
28:01
learning you're exploring right so you have your exploration and then almost think of this is like water percolating
28:07
in a way right it's going to find different paths and whatnot um and then it's basically just going to score them
28:14
based on you know what you said. So one you know one way of thinking about this
28:19
is that the search algorithm is going to give you exactly what you asked for which may or may not be what you wanted.
28:25
That's kind of up to you. Um, and so anything that you're not specifying is
28:31
fair game. Um, one example, this happens constantly. Um, and it's just a byproduct of search. Uh, cool example
28:38
that you can find on YouTube. I'll see if I can link it. Um, so even a simple
28:43
agent found a long-standing bug in Super Mario 1 that had been there for like 25
28:49
years where when you jump, if you turn back while after you jump, you have like
28:54
one frame in which you're invulnerable. Um obviously nobody had was able to exploit it but certainly an agent can.
29:01
Um and so if you like you know we training agents normally to kind of play
29:06
like humans that's kind of underspecified but we never say it in the reward function. The reward function is clear the levels as quickly as
29:12
possible with a higher score as possible. And so the model will just exploit the out of it because the
29:18
sooner you clear a level the more bonus points you get after you get to the flag, right? Um and so that's how you
29:23
get to these like um you know non uh nonhumanl looking uh non-human looking
29:28
models. Um you this is this honestly happens even with humans actually
29:34
there's a great example on Wikipedia on this um which is perverse incentives. So
29:41
this example was uh from India in the 1800s where they had too many cobras in
29:46
Delhi and so the British protectorate had the great idea of saying okay I want fewer cobras so what am I going to do
29:53
let's just have a bounty right like bring me a dead cobra and I give you some money right so that's what you ask
30:00
for to the process and what you get is people bringing cobras uh so by the end
30:06
they gave up on it and therefore people just freed up the cobras which are now worth Um so in the end Delhi had more cobras
30:12
than in the beginning of the process. Uh and so this happens like constantly. So you're kind of moving towards a
30:20
like almost dictating a policy in a way like uh not a policy in the RL sense in a policy like a policy maker like you're
30:26
writing a law right this literally happens all the time with laws people find loopholes and so the search process
30:32
will kind of do that. Um and so yeah you have to monitor and you have to iterate. Uh it's true that you cannot possibly
30:39
read everything that the model produces. Um you sample you can do some like aided
30:44
search for example you get like vibe checks and then you're like oh like for example this is what we did with
30:49
llama like huh these answers are kind of short. All right so let me compute average length of a thousand's
30:55
responses. I'm not reading a thousand responses but if I get the average length it's lower. It's kind of confirming kind of my vibe. Right? So
31:02
you start with a vibe and then you use filters to kind of see if that's the case. Um it's still kind of more art
31:09
than science. I think this is also an area where an LLM itself can do this eventually. Um but hopefully you get the
31:15
gist. Yeah, I was going to say like I actually agree like you know it's more like you're trying to say the environment is
31:22
the problem. like you know there are like there are some sort of like bug um there's like some sort of like you know
31:27
as you said the Mario example um you know there's like you know some hacks you know you could like yeah the I guess
31:33
the environment itself has like you know specific things you can like you know um utilize um and yeah like you know even
31:40
humans have this problem like you know we can like hack our way to do something um yeah I guess I agree with that point
31:46
um yeah I'm so sorry to the audience for
31:52
interrupting Daniel's energy, but no, maybe it's a question or a comment. I always had the conception that for some
31:59
reason reinforcement learning is like this idea where you don't have to be involved in data prep. But the more I've
32:05
played with it and like to uh to David's comment, it feels like this is like more involved in the sense once you like
32:12
achieve these rewards possibly successfully as well, then you have to figure out how to swap these questions for harder questions. And sometimes when
32:19
you're starting reinforcement learning, you give like very tough task to the model which it like you say luck is all
32:26
you need or like patience is all you need but sometimes it's like the task is so hard it never gets it right. So do
32:32
you see like there's also like more data prep involved as you like train the model you give it easier questions and
32:38
you give it harder questions because I've seen this come up in some people. Yeah, I think that's called curriculum
32:43
learning. I think like you know essentially you can like guide the process. You do like some easy questions and then over time you give it harder
32:50
and harder questions. Um yeah, that's actually pretty smart. I think like you know all of Yeah, I I think that's
32:55
actually what people should do. Um you know like don't don't give like hard questions at the very beginning. I think
33:01
like one of the things I said like you know if the probability of the probability of a good answer was zero for example if you ask the model a hard
33:07
question the language the language model would just never learn. Um, so you should, as you said, like, you know,
33:12
maybe do some easy questions first and then move over, you know, increase the difficulty as time goes on. Um, I think
33:18
that's actually I think that's I think that's what people do. I'm not sure. Um, but I if people are not doing that, you should do that. Um, yeah.
33:24
So, you probably don't want probability of zero and probability of one. You want it to be like somewhere in between of that.
33:29
Yes, correct. Yeah. Yeah. Definitely not probability of one. Then I guess the then the model will just output the same response over and over again and it
33:36
won't be very useful. Um, yeah. Okay. Thanks. Thanks.
33:42
Yes. Okay. Um but yes like in terms of the you know the you for example this
33:48
RLVR process um so essentially previously like you know you had a reward model. So this is actually a
33:55
model um to predict the reward for a specific um you know sentence or a
34:00
specific token. Um instead now with RLVR it's an environment. Um so like you
34:05
essentially it's a tester a verifier. Um this could be for example a formatting check you know does this code execute or
34:12
not um a reg regular expression check or the lm as a judge approach um and or
34:19
another way is if for environments you can also use open environments for the actual environment generation. So for
34:25
example this step you know this environment step you can use open environments which has a collection of
34:30
many many many pre-built environments. Um for example in this notebook that we have um you can use open environments
34:36
2048 um game um to actually play the 2048 game with reinforcement learning
34:42
right so at the very beginning of very beginning it doesn't actually you know you the goal is to generate an algorithm
34:48
to solve 2048 at the very beginning it doesn't do very well um you might get timeout issues um you get like
34:55
exceptions it just doesn't do very well and then over time it might actually generate some sort of strategy to play
35:02
the 2048 game. Um so it will get better over time and so open environments is a collection of many many many
35:08
environments which you can use for um you know for reinforcement learning.
35:15
In terms of like other updates like you know from the onsoft side of things we are going to be releasing ultra long context reinforcement learning which is
35:21
coming this week soon. So this will make so this is directly compatible with open environments. This will make training
35:26
even much more efficient and much more better for reinforcement learning. Um and another m you know another important
35:32
point from reinforcement learning is you need to make inference extremely fast. Um you know previously if you want to
35:38
you know normally what happens is as overtime goes on as time goes on inference will take 99% of the training
35:45
run and training will only take 1% or less. Um and so the goal is if you want
35:50
to make reinforcement learning faster you need to make inference faster. Um and so you know we spend some time
35:56
trying to make inference faster. Um so you know llama's faster, jupiters is faster um by using um unsluff for
36:02
reinforcement learning. Um and yeah also ultra long context as models get more
36:07
and more capable you need to increase the context length. Um so you know as the problems get more and more harder
36:13
you need to fit larger and larger context lengths inside the window. Um and so you need to focus a lot of time on trying to make this possible. Um and
36:21
you know finally you know my view is you know most of the large labs you know their goal is reinforcement learning
36:27
will automate reinforcement learning. Um so essentially the goal is you know to create so many different types of
36:33
environments you know to to do weather prediction to play games to do you know stock trading I guess to solve mass
36:40
equations but also to you know do training faster right you can essentially make an environment for all
36:46
of these characteristics and this is what you call the intelligence explosion right so some people think AI can
36:52
automate AI research um and once you get to that point um then you reinforceable
36:57
learning can scale to like AGI for example. Um so this is kind of the viewpoint of the large labs for example.
37:03
Um and you know some people think this might happen, some people might not think this will happen. Um and so like
37:08
you know this is more like a wait and see approach. Um you know can we allocate more resources resources
37:13
through reinforcement learning to make you know to reach this intelligence explosion phase. Um yeah but like you
37:20
know I guess thanks from my side you know that's kind of the main points I wanted to say. Um definitely check us
37:25
out on GitHub. Um and I also have like a conglomeration of stickers um of random
37:32
unsolved stickers. Um but yeah, thanks for this part. Um yeah,
37:39
wonderful. Uh thank you Daniel. Uh I guess like Daniel will be coming back for like a second talk like in the
37:45
[laughter] in this like current workshop. So if you enjoyed this, you'll enjoy next. I think the next speaker was
37:50
remind was it Lewis? Let me see. I think it was Lewis. Yes.
37:57
Yes. All right. You can see the screen.
38:03
Yes. Very clear. Awesome. Cool. So, yeah, thanks a lot for uh
38:08
having us here, Mark, and uh also thanks to D for giving this awesome introduction to reinforcement learning.
38:15
Um, so Daniel gave like a really nice overview of like how reinforcement learning works and also how it's become
38:23
like a kind of important part of like post training toolkit. And um I think
38:29
almost every model now that is released has some degree of reinforcement learning uh in the pipeline. And there's
38:35
a really nice um picture here from the ALMO 3 [clears throat] uh paper uh from Alan and I where they
38:42
kind of show like the the various stages that go into like uh you know post training. So typically you have some SFT
38:49
then you might do some preference alignment DPO and then typically some VR and there's like this whole process that
38:55
Daniel kind of alluded to where you kind of have to filter your data to get like good prompts that are kind of learnable
39:00
that they have the right foot1 and all that kind of stuff. Um and this works really well like we we we know from many
39:07
uh open recipes that you can significantly improve the performance on various capabilities like math and code
39:15
um by doing a um but there is like a kind of common problem and the common
39:20
problem is that at some point um your reward will typically plateau and what this means is that you reach a point
39:26
where [clears throat] the kind of set of prompts uh that you've got and perhaps just the capacity of the model um can't
39:33
really go beyond the ceiling and so then your kind of choice is like okay I could
39:38
like take a take checkpoint of a model and then find some more data and like keep doing more RL um but in general
39:45
this kind of idea of just doing RL data sets is a bit limited
39:50
and on top of that like the the kind of let's say the real world like the real world applications um of language models
39:58
um are far more complex than just you know solving like a math model. Um, in
40:04
reality, you know, the kind of models that we we use today, they have lots of interactions with different systems. So,
40:10
if you're a programmer, um, you've probably use cord code and codecs. And what you can see is that these kind of
40:17
like systems are like interacting with like files with the debugger, uh, maybe,
40:22
you know, interfacing to APIs, all that kind of stuff. And so there's like a whole range of different signals um that
40:28
now the model has to deal with. And if you're only training on like kind of static data sets, it's pretty hard to
40:34
kind of teach the model how to kind of generalize um to these like you know different domains. And so the sort of
40:42
common paradigm um of like going beyond doing RL like static prompts is to
40:48
introduce this concept of an environment. And there's a really nice picture here from Meta's code model. Um
40:55
this was uh a really cool paper where they showed how to basically do RL at
41:00
scale uh with like thousands and thousands of environments for coding. And the basic picture is that you have
41:06
some prompt, you give it to your your agent or your model. Um it might do some
41:11
reasoning uh like Daniel showed with the matrix modification to figure out how to make things fast. It will generate some
41:18
output or an action and then you feed this into the environment where [clears throat] the environment might be
41:23
you know like a compiler or it could be you know some unit tests and then that environment will then give you uh a
41:30
reward and some observation and some other information and then you kind of loop this over and now the agent then
41:37
based on that kind of information now has to take another reasoning step another action and then so on so forth.
41:43
So basically you have this paradigm of like taking actions uh observing the
41:48
output from the environment and then you know learning back problem um to improve
41:54
your ability to take those actions. So these environments as I said they
42:00
could be things like you know unit tests um but you know nowadays right we want these things to be kind useful. So they
42:06
can be for example web browsers and there's a a really cool um open source
42:12
environment called browser gym uh which basically you know creates a whole bunch of different tasks around you know how
42:18
to get like a language model to basically um navigate through the web
42:23
and you know this is the thing that I think many people kind of are like hopeful that you know eventually we'll have agents that you know can sort of
42:29
take actions for us like you know booking you know Airbnbs, flights, all that kind of stuff um or interfacing you
42:36
know with Excel on the web all that kind of stuff. So this is like a really good example of a real world environment.
42:42
Um the one that is also very like popular for like software engineers um is to have something like you know MCP
42:50
um or the ability to say use MTPs or skills things like this. And um a very
42:55
kind of interesting direction uh from mostly the Chinese labs has to basically create kind of simulators of these MCP
43:03
servers um where essentially an LLM is kind of pretending to basically be an
43:09
API and then the agent is interacting with these like simulated APIs um to
43:15
then you know figure out how to get better at using these servers. So those
43:20
are a few examples and one of the things that is like kind of alluded to in the
43:25
discussion with Daniel was about like curriculum learning. So if you're using like a static kind of data set the kind
43:33
of the difficulty of the problems is to some extent you know fixed like a priority. Um, and what you really want
43:40
to be able to do is like have the ability to kind of give the model like on demand the kind of most learnable
43:47
promise. And so environments for RL give you a kind of natural way to kind of
43:53
provide the model kind of you know every batch or every training step uh kind of most appropriate problems. Um and
44:00
there's a very nice paper uh that came out like just the end of last year called RLB where instead of doing
44:05
reinforcement learning with verifiable rewards, you do reinforcement learning with verifiable environments. And they
44:11
showed that for example here if you take something uh like an existing reasoning model and you try to kind of squeeze out
44:18
a bit more performance uh [clears throat] in math using like standard RLVR, you kind of really are
44:25
kind of stuck at like half a percent. But if you have this like diversity of environments and and problem difficulty
44:31
then you can get you know some significant improvement which which is nice. So these kind of three things of
44:38
like you know having real world environments the ability to kind of naturally have a curriculum makes this
44:44
paradigm quite compelling and obviously like you know open AAI and anthropic and
44:49
you know the other kind of [clears throat] frontier labs that they were the ones to sort of pioneer this with things like 01 um and later 3. Um
44:58
but you know since then many of the sort of you know big Chinese labs have been releasing not just the models but like
45:04
kind of detailed tech reports um into how they post train them and a really good example of you know using a lot of
45:11
environments comes from the deepse 3.2 two paper where they mentioned that they
45:16
basically created like you know roughly 2,000 like different environments and over 85,000 different tasks and then
45:24
they use this uh to basically you know improve the capabilities of deepc3.2
45:31
two across like a wide range of different agent kind of benchmarks and
45:36
you can see here um in this uh kind of graph from the the [clears throat] tech report that you've got this kind of like
45:43
you know performance that comes from SFT and then this blue curve is like kind of showing like through the course of
45:49
training how one of these benchmarks called tailbench um you know is improving through this kind of process
45:56
coupled with these environments. Um since then, uh you know, the labs
46:02
have kind of been trying to outdo each other. So there's Miniax. Um they have a really nice tech report or smaller blog
46:09
post where they basically uh show um that if you scale this like really really fast. So now it's not just like
46:15
2,000 environments, they have like over 100,000 environments um which they kind of created from GitHub repositories and
46:22
then basically the repository itself is then an environment. So it's kind of like imagine that you've got like an ID.
46:28
So you can like you know you have a task like fix this bug and then the [clears throat] the agent or the policy
46:34
has to then you know navigate the environ navigate the repo fix the problem make sure the test pass and you
46:41
know at the time of the release of of this one model um you know you can see
46:46
that across all these different like eval uh this process does does work and
46:52
in particular it sort of mentioned that they observed this kind of like you know positive correlation between like
46:57
scaling the number environments um but that they're far from reaching convergence. And so this kind of
47:03
suggests that like you know we're sort of buried on this whole topic of of environments and how far you can really
47:08
push um you know the performance of models across like various you know domains that we care about by you know
47:15
scaling compute but also by scaling the diversity of environments they're exposed to.
47:22
So, you know, where do we get these environments? Um, today in open source, um, they're usually scattered on GitHub.
47:29
Um, and this reminds me a lot of like the early days of the transformers. Um
47:35
you some of you may be too young to remember this but there was a time when um the kind of only way to get access to
47:41
mob weights was to go to GitHub, find like a URL to some Google Drive, try to
47:48
download the weights, try to run the code, but it never really worked. And that was like a big motivation, you
47:53
know, at home phase to create the transformers library to sort of unify all these like disparit uh ways of of
48:00
like, you know, training transformers and also sharing them in a kind of standard way. And there are like other
48:06
projects in the community uh trying to kind of unify the environment um ecosystem. Um but we've got like our own
48:14
kind of take on that. And it's sort of like part of the motivation is like how do we make you know environments kind of
48:19
like the first class citizens that you know we already have like data sets and models. Um but you know something like
48:25
you know the hugging face hub how do we kind of integrate this in in a straightforward way.
48:30
So this is like the motivation behind open end um which is a community project
48:36
uh that was kickstarted by meta face and um the basic idea here is to take the
48:43
sort of you know wellestablished like kind of paradigm for environments in RL
48:48
that was you know pioneered through things like opening a gym and gymnasium and in this kind of like sort of setting
48:54
you basically have like three things that you need to uh kind of keep in mind. you have obviously the agent and
49:00
the environment. Um but then you just want to keep track of like you know the actions the agent makes the observations
49:05
that you get from the environment and then the rewards of course. And so your kind of like loop uh for these
49:12
environments is pretty simple. You basically um [clears throat] have a kind of environment you're interested in. So
49:19
something like text arena which is like a bunch of text environments. You can basically pull these from the honeyface
49:24
hub and then this will [clears throat] like you know either instantiate itself as like a docker image or docker
49:30
container and then you can like inter interact with [clears throat] it. Um or as we'll see later you can also just
49:35
directly you know send HTTP requests to the hyping face hub and get your requests directly from uh from from
49:42
there. And so then once you've got your environment, you just have a basically a loop which is like you know the agent
49:48
takes some action. You feed that action into the environment you get the result and then from that you track the
49:54
[clears throat] rewards and then the rest of the loop that Daniel mentioned um you know for RL follows
50:00
straightforwardly. So, an example that I'll talk about a
50:05
bit later, more hands-on. Um it's kind of like a simple one, you know, using Wordle uh the game that I think most
50:12
people here know about where you start off with like a kind of hidden word uh that you have to guess and you can kind
50:17
of select uh uh you know some candidates and at each step you're given some like
50:24
partial feedback as to whether you know the the letters are at the right place um or you know are you completely wrong
50:32
together. Um, and the way it works like in kind of an IRL loop, um, is that
50:37
you're going to basically, you know, instantiate your environment. Um, you generate your rollouts which are going to be basically your your candidate
50:44
solutions and then you extract the the actions from those rollouts and then you
50:49
feed them into this game and then you get the rewards and then that's it. So
50:54
again the the kind of interface uh between open end and like you know uh training frameworks like TRL uh is
51:00
pretty straightforward. Um and it's been designed in a way that is you know kind of maximally agnostic. So it doesn't
51:07
make any sort of strong assumptions about you know what the training framework is actually going to do with those rewards and how it implements you
51:13
know different algorithms. Um and we have you know examples in in a variety
51:18
of different training framework you know unsloth and agent enforcement trainer um which is like again one of the goals of
51:24
the project to sort of show that um you know as a community you can build these kind of environments together and then
51:30
they can kind of be you know maximally shared across all the different initiatives to to train.
51:38
So I talked about this idea that you know you can pull uh environments uh open end from the honeypace hub. Um and
51:46
the basic kind of infrastructure behind this is something called hugging face spaces. Um if you haven't heard of
51:52
spaces I I won't try to sort of you know do a big intro but basically um these
51:57
are like kind of applications which run on the hugging face hub. They have their own hardware so it can be CPU or GPU. Um
52:05
and [clears throat] these inter these kind of applications um you know conventionally have been used for building like machine learning demos. So
52:12
the things like if you want to like have for example a chat interface for your model or if you want to share um some
52:17
sort of like cool way that your model can you know generate outputs. Um but this whole infrastructure can be
52:23
basically repurposed uh to create environments. And so the way it generally works is that you sort of pit
52:29
install the space um as like a uh as a as a kind of a module like git repo. Um
52:36
and then you can like pull this uh environment from the hub um and then you know do all the things I mentioned
52:42
before. And the thing that like I find you know particularly cool um is the idea that you can use these environments
52:48
uh kind of remotely. So, um, you know, if you want to like, you know, run lots of different environments, it's a bit of
52:55
a drag to have to kind of like, you know, install everything or set these things up kind of locally. Um, but having the ability to kind of quickly,
53:02
you know, sort of, you know, interface with these things remotely, um, I think is like a very cool feature. And Ben
53:07
later, we'll talk about some benchmarks that he's run, you know, comparing kind of like, you know, what's the kind of
53:13
throughput you can get, um, both locally and remotely.
53:18
So I mentioned that these environments are spaces. Um that means that they're essentially deployed uh on the hub and
53:26
they have like a user interface and an API. Um but because they're a special
53:31
class of space called a docker space, they also then come with like a registry. Uh they're git repos. So you
53:37
can version them like anything else similar to how we do with models. Um and they also come with you know built-in
53:43
things like you know authentication and so on. So one of the kind of cool things here is that now the same sort of things
53:48
that you know we're used to using in the face hub around like models and data sets can now be linked very closely um
53:56
with with environments and I think will talk after me showing
54:03
you how you you build environments. Um but in open end it's quite straightforward. You basically have a
54:09
CLI you can in it and it will create template of like you know all the files that are needed to build these like face
54:15
spaces and then you can just push them to hang and then share them you know with your collaborators.
54:22
So that's me for now. Um I don't know Mark if there are any questions at this point.
54:28
Uh yeah absolutely. I I I think I had like sort of one main question like I think this is in reference to the thing
54:33
you mentioned on TC creating like 18,000 environments like presumably machines created the majority of those
54:39
environments. Uh and so like in this setting like how do you think about like like the quality
54:45
of these things like the sort of their propensity for reward hacking uh
54:50
measuring the the utility of an individual environment as far as with respect to the entire postraining run?
54:58
Yeah, just broad thoughts there would be great. Yeah, it's a really good question. Um,
55:03
so indeed I I think from what is like at least you know publicly disclosed in in
55:09
these like tech reports um it seems that uh most of the emphasis has been on
55:16
environments that are relatively straightforward to automate uh the
55:21
creation of. And so as a result that tends to be things like code related like software engineering related or uh
55:28
tasks uh perhaps where you have like um a final answer that can be verified
55:34
easily. So you know things like in mathematics right uh you have an easy way to kind of say verify the
55:40
correctness. Um so what I've seen in a lot of these papers is they will um
55:46
typically create uh some you know real environments um like they'll take maybe
55:52
some MCP service from GitHub and then they will install them set them up as as Docker containers um and then this gives
55:59
them a template to then basically use another agent to kind of create variants of that with different tasks and then
56:06
you can kind of then run this at scale. um when it comes to more like real world
56:13
tasks um from what I understand this is still a very like uh artal uh you know
56:20
enterprise so you know there's there's discussions like in the rumor mills that
56:25
like you know openai and athropic pay you know hundreds of thousands of dollars for people to create things like
56:31
you know a clone of slack or a clone of salesforce uh because then you know if you agent in this like high fidelity,
56:39
you know, replication, you have a good shot that then, you know, people can then sort of integrate this into Slack without having to, you know, gather lots
56:46
of data at infant. Um, so the short answer is I think so far in open source,
56:53
um, we're still, I would say, kind of like constrained to to these environments where it's easy to kind of
57:00
verify things, um, and it's easy to to sort of set things up around GitHub
57:06
repos basically. Um but going beyond that I think will require some some big
57:12
dedication uh from from some groups or individuals. [clears throat] Um I have one more question. I think
57:18
Sonia also had a question. Um so I mean I'm old enough to like remember the like
57:23
the Dota bots like I think for me that was like a big field AGI moment because I'm a big player myself and I played
57:29
against these bots and they just destroyed me. And so there was sort of this meme at the time that like well all we need to do is collect more and more
57:36
games and eventually like this would sort of like give us a path towards AGI but like that turned out to be sort of a
57:42
research dead end. And so what would you say is different basically this time where like you're describing something similar where people are creating all
57:49
sorts of simulators but this time people feel more convinced that this will actually work than we we did 10 years
57:55
ago. So why why do you think that's the case? Yeah, I guess the obvious difference is
58:01
like in the Dota context, it wasn't an LLM controlling uh the games and and even like later,
58:08
you know, deep mind they had this um uh gate like the alpha star like the the
58:13
Starcraft one and the go one. Yeah. Yeah, they also they also had a nice one. I think it was um gau where it
58:21
was like uh essentially trained across like you know many many different um [clears throat]
58:26
uh games but also like different uh kind of like robotic environments and stuff. Um and indeed it also didn't lead to
58:34
anything beyond that. Um my suspicion is it's probably because that you know the
58:39
LMS weren't in the game. So you you didn't have the flexibility to kind of benefit from all the kind of world
58:45
knowledge that the LM has ingested to then hope that you can kind of generalize um you know a bit outside the
58:52
the narrow confines of a game. Um I think um I think here like the the sort
58:59
of big push on environments um is partly related to something that's discussed a lot around this like you know jagged
59:05
intelligence like the idea that like most of the models that we interact with
59:10
they have like kind of spiky capability where like you know they're very good perhaps [clears throat] on domains that
59:17
you know the labs have targeted uh to optimize but then if you kind of deviate
59:22
out of this they they tend fail in this like spectacularly silly way. Um, and part of the hope I think here with
59:29
environments is that if you scale these in diversity, um, then there's there's a there's a better chance that you kind of
59:35
smoothing out this like jagged intelligence. Um, whether that turns out to be true, I think is still an open
59:41
question, but I think that's the way it's going. Okay. So, just doing a time check here.
59:48
I think the San had a question and then I guess San was also the next speaker so you can introduce yourself if you like.
59:54
Um no thanks thanks Louis this is like really insightful. Um I I had a question which I think David might have answers
1:00:01
to but there's like so so much thoughts on and sorry to put
1:00:06
David you on the spot but there's so much thoughts on okay you should like train this model in pre-training then
1:00:11
you go to mid- training then RL when you like think about environments and all these tasks we throw at the model is
1:00:18
there like some method to the madness that you probably wanted to like solve certain list of tasks first or certain
1:00:24
environments and then you graduate it Because I imagine like if you have 18,000 environments or whatever, you
1:00:31
can't just like throw everything at the model at the same time. Yeah. Um, okay. The picture tends to be
1:00:39
a little bit more messy than this in practice. Like for example, uh, this is something that Quen started
1:00:45
doing a while back, but I think now it's standard practice. You actually want to inject even some of these post-raining
1:00:51
traces back into pre-train. Um, and that'll help downstream. So the whole picture is a little bit like it's
1:00:57
whatever works. Um I think in practice so there's there's kind of in theory yes
1:01:05
the idea of doing curriculum learning makes sense right you start from the easy things first then you nail them
1:01:10
then you move to the medium ones and so on and so forth. So in learning right
1:01:16
you should always be aware of the bitter lesson right and so the more kind of human biases
1:01:23
I'm not saying biases in a bad term like inductive bias you're injecting into the process on the one hand you're making
1:01:29
learning bootstrapping easier uh on the other you are kind of constraining where search can go so in practice you have a
1:01:36
lower um so it depends I think this is something yeah we have tried uh even in
1:01:43
the lama And uh like sometimes it works, sometimes it doesn't work. You normally
1:01:48
want to kind of have some like whenever you're undecided, it's normally good to have some sort of like a random schedule
1:01:55
to it. So for example, you can kind of sample easy examples with higher probability and then you kneel this
1:02:02
probability to go down as you kind of continue training. Um or you can just say it. And uh like the it
1:02:10
approach like kind of works actually like basically what it ends up being is you
1:02:16
know you start from a hard example you're like I can't solve it. So basically you just wasted some compute there but you haven't constrained the
1:02:23
model. So you know depending on how much compute you have and uh depending like
1:02:29
there isn't a kind of clear-cutting rule I would say but um yeah I would say that like starting from a well start from
1:02:36
nothing because you always want a basement and then like having some sort of like um randomized schedule like that
1:02:44
um you know see if it helps. I actually have a question for you David. I haven't seen this discussed
1:02:50
much in literature but um you know there's the whole thing about like difficulty and curriculum but I wonder
1:02:58
in your experience um how much of a role is like the time determination of like
1:03:04
the the sort of trajectory important so like for example you know there's these
1:03:09
like tasks that are like you know popular on Twitter with like okay Claude used like I don't know 100 hours to to
1:03:16
do something obviously that in an RL loop would be horrible, right? In principle because you don't want to be
1:03:22
waiting like 100 hours uh to get your reward. And so do you do you have like
1:03:28
some scheduling on like the kind of length of of these tasks? Um or is that something that is also in the
1:03:35
bucket? So I'm going to speculate because we I haven't done something of that sort with
1:03:40
like such a long time. I think that the problem also becomes that um like in a
1:03:46
real ultimately you get a reward again you can tweak these things but like one of the things you got to be careful is
1:03:51
that if you get the same reward for any trajectory then you're going to be overweight on the short trajectories. So
1:03:57
learning the long ones you probably need like much higher rewards or something. Um I would suspect you probably would uh
1:04:06
break this up into phases. Um, not to contradict what I just said, but I think
1:04:11
this is just about being efficient towards your compute. And so it's probably okay to tolerate some
1:04:16
inefficiency, but once you're going to like these extremely long horizon tasks,
1:04:22
um, my suspicion is that you probably have to start from SFT again and so you have to synthesize like a good trace.
1:04:30
And I would synthesize it first, I think, at least to kind of get the model going, right? and then start the kind of
1:04:38
brute force reinforcement learning. Like the problem is if you start the brute force approach right away, if the task
1:04:44
is too hard or too complex and you never get to something that gives you clear rewards, then you're stuck forever. One
1:04:50
example of this was Starcraft. Like um if you for example, if you're learning chess, you can totally start from a
1:04:57
random model, right? Uh the random model plays itself. they both suck and so eventually one of them is gonna win and
1:05:04
that that kind of gives you rewards and you can kind of get the machine going, right? Not so for Starcraft because in
1:05:09
Starcraft there's so much going on, right? Then the sequence of action necessary to either win or lose a game
1:05:16
is actually quite hard. And so what ends up happening is you have like the beginning of the game the a random
1:05:22
policy will basically just kind of jiggle workers back and forth. Nobody wins, nobody loses, and you're stuck there forever. And so you have to notch
1:05:29
the process. But with either SFT data or reward shaping so that you can for
1:05:34
example say killing units is good. Do that or you know mining resources is
1:05:40
good, creating units is good and therefore the model will kind of do start doing this and then you know this
1:05:45
kind of is conducive to one of them eventually winning or just SFD data. Just copy what this guy did and then
1:05:51
just kind of get it going. So long sequences like this are going to be more similar to to a game of Starcraft I
1:05:56
would say. Yeah, that's a good point. Something I didn't mention just to end on this Mark
1:06:02
is that um the environments are not just useful like in an RL, right? Like if you
1:06:07
have a very good environment, you can actually use it as like a mechanism synthetic data generation. Um which is
1:06:13
again I think the value of having like you know these high fidelity you know clones of like you know valuable
1:06:19
applications. Exactly. [clears throat] Right. So in RL basically the price you pay is that each
1:06:24
trace like each episode is less informative than um an SFT example right
1:06:30
that you mine from people however you kind of shift towards like we can as long as I have compute I can generate an
1:06:37
infinity of these and so in the SFT regime I'm constrained by how much data
1:06:43
I have and this data is produced by people and sometimes it's very expensive um in the RL regime I'm only constrained
1:06:49
by compute Right. Um and so this is bitter lesson on compliant because the compute curve is on an exponential
1:06:56
curve. Right? Every you know two to three years Nvidia gets us like 5x the performance right. I don't get 5x the
1:07:03
data every two years. Far from it. The cost of data is always the same. It goes higher. So right so which where do you
1:07:09
want to be? And so that's that's really what it is. It's inefficient but you know I I have a lot of firepower.
1:07:18
I guess met has low GPS. Yeah, I mean the Yeah, I will say I
1:07:23
don't know if I'm as optimistic as you. I think the 5x performance every year is also sort of like becoming a bit sus
1:07:29
from Nvidia's case. Uh you know like black reliability blah blah blah but may maybe the subject for like another talk.
1:07:37
Um all right we have the one and only Sanam please take it from here. I will share your screen.
1:07:44
Awesome. Thank you so much. Uh I apologize in advance for my sound. I gave my mic to my family who I'm pretty
1:07:49
sure have never used it. They probably sold it on eBay. So, if everyone stars Unslaught, TRL, and um Open End, I
1:07:57
promise I'll buy a couch. Okay, that's my promise to you. Uh I'm I'm one of the folks who works on Open. In the intro,
1:08:03
Mark said all of the speakers are people he respects. Uh I'm not one of them. He let me on this uh stream because he
1:08:10
likes my joke. I would quote after maybe one or two beers. So, that's why I'm here. Uh what I'm here to talk about is
1:08:18
just give you all a walkthrough of how open sort of brings things together and
1:08:23
how you can bring your own environments. What's the pattern to building these environments and I promise I won't stay
1:08:30
between the great speakers. So I'll keep my talk quite short but please keep the chat spamming. I already see a lot of
1:08:36
messages coming up. I want to make like a suggestion here. Um, one thought I have is about LMS and
1:08:44
models. People always assume that they want a specialized model. So, so just just to double check, you
1:08:50
just have uh your screen was just click to exit full screen. Was that intentional? I was in slideshow mode. Let me see if I
1:08:57
should share it. Just just click it again. Maybe is the slideshow on?
1:09:03
No, it's just showing click to exit full screen. Okay, let me share my entire screen. Give me one second. Sorry. Mhm.
1:09:08
See guys, I clearly don't know what I'm doing. I'm sorry about that. Please, please, please complain in chat. All right. I'm bad at this. Share my screen.
1:09:15
Share my entire screen. All right. Now you'll get a mirror effect with mark in it. That's okay.
1:09:23
An infinite mark. Okay. Yeah. Okay. This works. Perfect. Awesome. Awesome. So the claim I was about to make is people assume or it's
1:09:30
kind of known but just to call it out that you assume that you want a specialized model but what you really
1:09:36
care about is a specialized LLM because you don't just want a tab completion
1:09:41
model right like that is sort of helpful when we're at the stage of cloud code where you want these models to be really
1:09:46
good and from the previous two speakers it sort of alluded to the fact that
1:09:51
reinforcement learning can really help with this. So that's where like reinforcement learning becomes this pill
1:09:58
uh where you want to inject it into LMS so they can possibly solve any task which you can put in an environment and
1:10:05
Zach actually has like a really exciting talk where he'll talk about sandboxes but the challenge with this is
1:10:13
presumably you have 18,000 environments like in the previous case
1:10:18
how do you like standardize these in a sense that we all are programmers here right? So you have like five different
1:10:24
environments forget 18,000 and their APIs are so different
1:10:29
how the heck does the model like efficiently train on this you probably want the like compute to be utilized
1:10:35
rather than writing all this infra right so that's the challenge that open is
1:10:40
trying to address how do we create the standardized spec which makes it easy for the entire community uh which I
1:10:48
think is quite similar to the spirit of GPU mode and we want to keep everything open source where you can bring your own
1:10:54
environments. They use the same spec and we're like very well integrated with the best uh trainers like TRL and Onslaught.
1:11:02
So everyone's life is easy, everyone is happy and the end goal would look like something like this where you just
1:11:08
change the line of import top where you change the Docker environment or the
1:11:14
Docker file where you're importing your environment from and your model keeps learning. That's the goal that we're
1:11:20
trying to march towards uh for the better of the community and also for better of research if I may.
1:11:27
So we wanted to test this theory and funny enough we did this like two days after launch. It was quite an
1:11:32
experience. Um Daniel and myself we did a hackathon with the wonderful folks at AMD and the models I'm quoting here or
1:11:40
the examples I'm quoting here are from a hackathon trained on a single MI300X
1:11:45
which really impressed us. I think Daniel was really happy with the examples. Folks came up with these environments
1:11:51
themselves. Um, most of them are in our GitHub repo and they trained their agents as well using just one GPU.
1:11:58
So from all over the place, you can see some folks called Mario. No, they didn't exploit the bug that David was talking
1:12:06
about, but that was still awesome to see. Uh, another cool example was improvements on code writing. And if you
1:12:13
go to the next uh slide, you'll see that my favorite one is on the bottom left, which was project Pikachu, which came
1:12:19
out of our discord. Um, no, I don't care about trading with RL. Who cares about money where can you when you can solve
1:12:26
Pikachu, right? So, in theory, we did test our approach and it did work quite
1:12:32
well. So, now I want to walk you through how you can actually build these environments and how do you bring them
1:12:37
to life because hopefully you're sold that this approach sort of works.
1:12:42
But to get there, I'll do a quick recap of reinforcement learning. So to
1:12:48
simplify things a lot, um you usually have an agent uh which observes the
1:12:54
world around it, chooses its action and gets some feedback based on which hopefully it improves and the
1:13:00
environment is the world around the agent which uh reacts to its behavior
1:13:06
and can give it rewards. In simple code, uh you can assume an
1:13:13
example of teaching a dog how to sit down. And in code, this would look like something like this where you ask the
1:13:20
dog to sit. And right now, okay, I'll just say it's probably my dog, right? It doesn't understand the word sit. So, it
1:13:26
just starts barking or sping in a spinning in a circle. And that makes me upset. So, I scold it, which is a
1:13:32
negative reward. And hopefully over time, it learns that it'll get a treat every time it says or it obeys sit. So
1:13:39
that is a very simple example of reinforcement learning but the idea is there's like a few simple things
1:13:45
involved every time and these are the concepts that we've built upon uh for the open environment spec
1:13:54
are uh you reset an episode of learning and that begins a fresh training
1:14:00
session. An observation is what the agent sees in the world.
1:14:06
An action is chosen by the agent and it's what it does. And one step in an
1:14:13
episode of learning is just either one action or one observation being returned
1:14:19
from the environment. So those are the four simple things uh which have been around in reinforcement learning. I'm
1:14:24
sorry to everyone who already knew this um but we kind of settle on this pick.
1:14:33
So to build your environment, Lewis actually showed you the two commands. So this is kind of ridiculous that I have
1:14:38
to show this, but when you do open and init your name of environment, it
1:14:43
creates a skeleton of files. Uh which actually simplifies a lot of things, but I just want to explain them as I go.
1:14:51
It's five files that you need to define where you define these behaviors and models. Uh so you define its action, its
1:14:58
observation and state data classes and inside the environment you expose the
1:15:03
behaviors which were the three reset step and state methods. Now we connect these using HTTP. So the
1:15:12
environment talks to the client. Uh we have to define that and then finally we
1:15:17
wrap everything in a fast API server. So you never actually see HTTP. Uh it's
1:15:22
ridiculous. We we don't like debugging tensor mismatch. So we definitely should not be debugging that. Then finally to
1:15:29
make this scalable uh and Zach will really like go in deep with this topic we wrap everything in a docker file. So
1:15:35
you can actually just import a docker file and you have a new environment. Of course uh you import that from hugging
1:15:42
face and we all love hugging face. It's really easy to use uh anything and everything on the hub.
1:15:48
So the universal interface for any environment would sort of look like this. You define an environment and you
1:15:54
expose these three methods reset, step and state. Now I've been
1:16:00
going on and on about this. So let me like put a concrete example to this
1:16:06
and of course u I forgot this slide but you have to also define these data structures so that things are
1:16:12
standardized. Uh everything is natively Python. There's no external dependencies, but you usually define
1:16:18
your action data class, your observation data class, and state data class. Um, action and observation are sort of
1:16:25
straightforward. In state, you have to be careful about what metadata you're exposing. And I'll also call out that in
1:16:30
state, you can cause a lot of reward hacking. So, that's like some place you probably want to carefully look at.
1:16:40
Again, this is the goal we're sort of marching towards, which is having a standardized training interface. So,
1:16:46
life is easy. But to put a concrete example, u you can
1:16:53
imagine the game of connect. Now, I had a very sad childhood. So, I actually never won a game. But for those of you
1:16:59
who won this game, you were supposed to drop chips uh in a vertical column. And
1:17:04
you can connect them horizontally or diagonally. And the first person who does that in this twoperson game wins.
1:17:11
So an action would be connect for action which uh decides what column you're dropping the chip in. Observation would
1:17:18
keep a track of what's your board state, what legal actions are allowed and is
1:17:24
the game finished or not. State would just keep a track of the board state. Right? And you can imagine
1:17:32
in the docker there's not a lot of dependencies. So you just have a numpy array and you just wrap your fast API
1:17:37
server to expose it. In simple logic, the reset for connect 4
1:17:44
would look like all right, empty your board, uh remove all the chips, set it to player one, uh make sure all
1:17:52
columns are allowed, and in any step a user drops a chip in a column, checks if
1:17:59
they've won. If not, the game continues.
1:18:05
And this is what a connect for environment code would look like. I just pulled this out of our GitHub. Uh it's
1:18:10
not formatted well. Sorry about that. But that's the simple logic for defining an environment. That's all it takes in
1:18:17
some sense. So I would say a lot of effort goes into curating your rewards
1:18:22
and also the questions or the prompts you're sending to your model to solve this. And the challenge that openend is
1:18:29
trying to solve uh is standardizing on this approach and hopefully making everyone's life easy.
1:18:36
So I wanted to put another inspired example for GPU mode. I'm very underqualified for this. So please
1:18:42
excuse me if this is not right but you can imagine a kernel sandbox conceptually where you want it to
1:18:49
improve kernel writing would have the following methods. In reset, it just
1:18:55
resets everything, records a baseline, probably keeps a track of observations.
1:19:02
In a step, it receives the kernel code, maybe compiles it with certain flags. I
1:19:07
just compile everything with default. I never like even look at the flags, but I know you all are experts, so you probably would do that. It runs the
1:19:13
benchmark, and maybe based on like how better the benchmark improvements are, you give your model a certain rewards.
1:19:22
In code that would look something like this. So your models would have kernel action kernel observation. Your
1:19:28
environment would be a kernel sandbox and your first API server would look
1:19:33
something like that. So that's all I had to present to today.
1:19:38
Um I also want to call out that we're also think about how to add tools uh to
1:19:44
these environments and David can talk a lot more about this. I would also request you all to look at our RFC's.
1:19:51
they have a lot more thought on all of this but we're thinking about how to
1:19:56
expose tools both in real time and also in simulation um as you train these
1:20:02
models one last call out I have for you all is please go to our GitHub uh consider
1:20:09
starting it consider joining our community or honestly the best thing you can do for us is just u check out the
1:20:16
project tell us what you think if you don't like things please please please complain to us. We'd love to like make
1:20:23
this project the best use case for you if you were training RL models, which hopefully you are.
1:20:29
Sweet. So, thank you, Sonia. I I guess like um do you have a good sense of like what
1:20:35
kinds of environments like would you be particularly excited to see? Like what do you think are some things that people are underexploring that you'd like to
1:20:42
see more of? I think I'm at a like different mind point now because like I'm curious to
1:20:47
like learn the methods on how people are using this. Um for me like the cool
1:20:53
games are like always cool environments, right? Because for all of us who write code, there's this like experience we
1:20:58
had with robotics, right? Where you write code and it's like holy crap like the the arm is moving or whatever. It's
1:21:04
the same for like RL in the sense you have this like cute little model and now it's able to play your game and
1:21:10
hopefully like play it better than you. So for me it's like that experience, but I think everyone has their own takes.
1:21:17
Zach has a lot of opinions on this. So Zach, I do. I in fact I I was up late working
1:21:22
on what amounted to just a few slides. Uh but I was going to put in a request for environments at the end. Um but like
1:21:30
one thing that would be really great is we we're all using GPTO OSS uh in some
1:21:36
form or another. It comes with some well it doesn't come with there's some missing tools there that would be
1:21:42
useful. There's a browser tool and like a coding container tool that they didn't
1:21:47
open source with the model and we could squeeze out a lot better performance if we had that same tool um to reproduce
1:21:55
results with and then develop further. So I would love to see someone I' I'd love us to do a hackathon even on the
1:22:02
quest for the missing tools from GPTOSS. Um, that that actually sounds pretty
1:22:08
fun. I'd be I'd be down to join. Um, maybe we should talk more about this. Uh,
1:22:14
I I guess maybe Ben, do you have comments on this or do you want to just get started yours your talk?
1:22:19
No comments. It sounds like a really cool hackathon. Yeah, I definitely join in as well, but I'm pretty much ready to
1:22:25
go with a tutorial. I I'll just say one more time for everyone asking, we will post all the
1:22:30
materials. Don't worry about it. Just like stay on GP mode. That's where the recording is. Go to OpenM Rebbo. That's
1:22:35
where we publish everything. Uh just just enjoy the lecture. Stop worrying about like materials. They'll be there. [laughter]
1:22:41
Yeah. I mean this lecture is already recorded by the way. Like it's just it's a YouTube link so it's chill. All right.
1:22:47
Uh Ben, please take it from here. Cool. So what I've prepared for the for
1:22:52
this session is like a tutorial and it's all contained within this single GitHub
1:22:58
repo that's called hugging phase GPU mode open. And maybe in the future I'll
1:23:04
fork it into like something else but but right now I made this specifically for for this presentation but I've actually
1:23:12
pulled some things from from other places in the open end repo. So the the
1:23:17
first um like markdown file is on environment and that's basically pulled
1:23:23
from Sanam's presentation on how to run ends which is basically what Sanam just
1:23:28
presented. So, I'm going to skip that, but afterwards you can come here and you can get a markdown version of that. Then
1:23:34
I've got a markdown page uh deployment which I'm going to present and that's on how to deploy your M's target base
1:23:41
spaces. And then another one on on scaling which is um just a tutorial on
1:23:47
how you can scale open environments for free. And then uh a final one page four
1:23:56
on training GPO with TRL and Wordle which is a kind of tutorial on what Lewis is going to present next. So so
1:24:03
you can start off from here and kind of get all of the resources. Um and that should be pretty handy.
1:24:10
Um let me go to this uh page two to start off. Okay cool. So in in this first part
1:24:18
of the tutorial, I'm going to take you through how to deploy an openm environment on the hub. We already have
1:24:26
a pretty good idea of what an openm environment is. Now Lewis presented it first. Sonia presented again uh again
1:24:33
and in simple terms, it's a fast API web application. There's nothing really
1:24:38
fancy to it. Uh I'm going to go into the details and show you what that looks like, but it's a Python package managed
1:24:45
with a a nice toml and and a fast API application. It's a very um familiarly
1:24:50
structured web application. And to deploy that application, we use
1:24:57
hugging face spaces as the infrastructure layer for this, right? So
1:25:03
a lot of people kind of associate spaces as like you know this great kind of demo tool and and you can push your grad
1:25:10
applications there or or any kind of application and you use it for demos like image generation or text generation
1:25:16
or whatever. But actually under the hood spaces have
1:25:21
three main attributes which are uh it's a server right like it's a web
1:25:27
application it's a git repository like models and data sets so you can do
1:25:32
version control and it's also a container registry so you can say like
1:25:38
docker pull and use the URL of the space a specific URL this one here um and say
1:25:46
okay let's pull the the cont the image of that that space and I'll run it as a container. So within OpenM, we can use
1:25:55
that um infrastructure without needing to build anything, right? This isn't anything specific to RN environments,
1:26:01
but it's um really advantageous. And then once we've got that uh once
1:26:08
we've got that container running, we can then interact with it in three ways. And
1:26:13
so I'm going to work through that first, how we um use these three points to access a space. So the first one is
1:26:21
this. We just treat the space itself, the server, the application running on
1:26:26
the space as our RL environment. So we say to our uh our class, our environment
1:26:32
class, use the base URL um of the space URL and let's uh reset that environment
1:26:40
and then let's step through that environment. And in fact, we can train
1:26:46
using this kind of interface. And I've done some benchmarking, which I'll come to in the next uh tutorial. And we can
1:26:52
train up to reasonable batch sizes with a hugging face space like this. And we
1:26:58
don't really need to do anything. We don't need to pay um to to deploy it on a upgraded system. This is just like the
1:27:05
free tier space and it will work on most examples. So
1:27:11
what kind of endpoints does this application have? It has a websocket endpoint which we recently implemented
1:27:17
and I'll come to why we implemented that and and how that's useful for the community. But it also has HTTP
1:27:24
endpoints which you can use to reset and step through the environment.
1:27:30
Okay. The other point that we can interact with a space is that it's a it's a repository, right? So we can pip
1:27:38
install the code from that repository which in the context of an environment
1:27:43
is very useful because we can install the client code of our environment in
1:27:49
our client site. So we can say like um pip install space url and get client
1:27:54
code and we can say pip install um sorry we can say docker run and then use
1:28:01
the container image. We've got the same thing uh from two places.
1:28:07
And then the last the last point is this uh docker registry which I already kind of came over where we can say okay let's
1:28:14
pull this space as a as a container and then let's run it or we can just do like
1:28:20
this and so what that means if you're building environments is that you only have to manage one location right and
1:28:28
here's all of that kind of summarized so you could do as I said you could
1:28:34
interact with it as a space. You could run the container like in a in
1:28:41
another terminal and then interact with that running container locally or like a
1:28:46
kind of just an abstraction of that. You can just say from hub and and use the repo ID and it will pull that container
1:28:52
and then and then run it and you can interact with it or in fact you can say I don't have docker available. Um, so
1:28:59
pull that Python code and run it using Unicorn and run the application locally
1:29:06
and then let me interact with it like that. And so we've got these four um interfaces to the same space which we
1:29:13
can use to run in different ways. And I'm going to work through those uh now.
1:29:19
So the first one is um just to use the Python code itself for the fast API application kind of the most obvious. So
1:29:27
we can clone the repo uh and then sync the the Python package and then run the
1:29:33
server. this sake uses a script a script sorry but we could also just use the
1:29:38
unicorn command and then um we can um yeah or we could
1:29:48
refer to the project itself and run the script like that and then we can um interact with our
1:29:55
Uicorn server and like define the host and the the ports and run the server
1:30:02
or moving on we can run the docker container and use the same example. We
1:30:08
can um pull the container and then oh sorry we can clone the repo and then
1:30:13
make changes to it and then build the docker container as well. And inside the openm cli we've got like a convenience
1:30:20
tool openm build that that does that for us but it's essentially just wrapping
1:30:26
docker build. So, so we can also do the same there and then we can run these containers and
1:30:32
as I said before connect with them uh from Python in the same kind of way.
1:30:39
So what does it look like from the CLI? So as we showed before, you can do this
1:30:45
open end in it and you can create the skeleton and then you can push that straight to the hub which I'll come back
1:30:51
to in a second uh as a little demo and then you can configure that
1:30:57
environment um mainly with environment variables to say like how many workers do you want to use uh and um whether you
1:31:04
want to use concurrent environments which I'll come to in the next section.
1:31:10
So feel free to kind of drop in any interruptions if you like. Uh I'm just
1:31:16
gonna set this up so it's a little bit easier to see. Yes. I I had a couple of questions like like basically um so I know you
1:31:22
mentioned at some point like oh if you don't have like Docker you can use UV but like how does this work? Like I mean presumably most environments aren't
1:31:29
written in Python like if you're a Pokemon you're written in like whatever that was written in. Then you have like some shim in Python. So could you speak
1:31:36
to how like basically native code gets deployed in environments on the hub?
1:31:42
Yeah. So when when the environment is outside of Python, you need to use a
1:31:48
container to define the code outside of of Python and do dependency management basically. So if you say like if you
1:31:55
want to run the Uicorn application, sorry the Python application, uh yeah, you can't you'll need to manage the
1:32:01
dependencies outside of Python and we'll do that with Docker. That's basically Yeah.
1:32:06
I see just and I know this may be a pedantic question but just because you have cost there. So the cost basically
1:32:12
it's like is it like paid for by the people hosting the environment or is it paid for by the people that are pulling
1:32:18
the environment? Uh how does this work if people are having like multiple rollouts where they're doing multiple
1:32:23
parallel evals? Yeah. So in the example I showed you where like this kind of example where
1:32:31
you interact with the like the space itself, right? Like the spa the
1:32:36
environment host would would pay for that compute. Um, and so in most cases,
1:32:42
like if you're publishing an environment, you would probably want to do that free and just let people try it out and kind of understand the
1:32:48
environment that you use to maybe learn from it and then they could pull that environment and and run it for free. Um,
1:32:55
because of how spaces works like uh let me just maybe show you like you you can
1:33:02
just duplicate spaces so they're kind of like super easy to just pull over to your Oh, I see. like like running locally is
1:33:08
something you want people to do like you want discover things and then like I see
1:33:13
totally so yeah I expect most people will like come here and like try it out and learn from it maybe duplicate it if
1:33:20
they want to make changes and share it but in most cases we expect people to kind of um to run this this locally.
1:33:28
Got it. Okay, cool. Um so yeah. Okay, so I I shared
1:33:34
this little demo. Um, we we've kind of seen this a little bit already, but let me just work through it. So, you
1:33:39
basically just do open M in it, which looks like this. And it's um a simple
1:33:45
command line. It just needs to know what the name of your environment is going to be. So, let's call it GPU mode. Um, and
1:33:53
so we basically
1:33:59
I'm in the right place. Okay. Sorry.
1:34:05
Sorry, I I just had a quick add to what uh Ben answered about Mark's question.
1:34:10
Um for like most of these game environments and things like that, you can imagine you sort of building adapters or rappers just like add on to
1:34:17
the spec. But our goal for like hopefully this year is we'll bring like environments from the ground up. So
1:34:24
we'll see them like being less clunky. And the other thing I want to call out is like most of these are for
1:34:29
reinforcement learning training. So that will go in a lot more depth, but the CPU
1:34:34
cost of running an environment compared to like the GPU involved in training is like really fractional. So you can
1:34:40
imagine like you probably would not mind hosting these on your server.
1:34:47
Sorry to interrupt your flow. No worries. It was actually extremely helpful.
1:34:52
Gave me a chance to get this back in line. So yeah. Okay. Um, as I was
1:34:57
showing you, this is just how you initialize an environment. So, and I just want to kind of work through that.
1:35:02
So, let's say I'm going to create a new environment called GPU mode. So, I just say open in GPU mode, and it creates
1:35:10
that for me. And it tells me that there's this new file called openend mode. Let's go and have a look at that.
1:35:16
And if you look over here in like the file explorer, this is what it looks like. And this is a skeleton. So, it
1:35:22
comes out the box. So if you're looking you're looking on the right pane here uh and it comes out the box with this
1:35:27
readme that has some metadata which makes it discoverable on the hub basically uh but also gives you like the
1:35:34
templates and you'll notice that we've like said uh the entire skeleton so that
1:35:39
it's already preconstructed for you with the name that you chose which in my case was GPU mode and it gives you all of
1:35:45
these examples of how you do it. So out of the box you have like a self-contained fully documented end that
1:35:53
is also interoperable and and standardized and uses open m. It comes
1:35:58
with this pi project toml which installs um all the like the base dependencies
1:36:04
that you need and then gives some like recommendations and gets that running.
1:36:09
Uh and then it has the the client code itself which Sanam already kind of
1:36:14
unpacked and went into detail on.
1:36:19
And that's uh that's all there. Right. So then um let's just uh change into
1:36:26
that. Probably don't want to go too far into this because we want to go into the um some other things after that. But we
1:36:34
can then basically just like run this end like this locally. Uhhuh. H sorry
1:36:39
that port's already in use. So yeah. Okay, that's it running locally
1:36:45
now. And then if we go to our like Python interpreter, I think I have that
1:36:50
here. So okay, if I did that, I think it wouldn't work. That's not deployed. But
1:36:59
if I copy the URL out of here, then I can just say, okay, run on here.
1:37:07
Uh yeah. Okay. And then it's like step through and that's the reward and then another reward. Uh and that's this um
1:37:14
openend environment running here. Uh and then let me um just now go back to my
1:37:21
terminal. Close that down. Uh and do open m
1:37:26
push. Open them I'm in the right directory. I was. Yeah. Cool. So then that's now
1:37:33
pushed it to the hub. And if I go over here, I open a browser, you see that
1:37:38
I've got this uh space running here, open M. And then I can uh just like test
1:37:46
out the end here. And this is just a skeleton end, but just like echoes back whatever you you ask it. So it's just
1:37:53
echo back. Hello. Um, but if the end was like more complicated or let's say it
1:37:58
was like Sudoku or Text Arena or even browser control, you could try that out and kind of get a sense of of how that
1:38:05
environment works and what the model was was doing um for this use case like in the space and then you would be able to
1:38:14
as I show you go here and kind of duplicate it embed this space but you could also run locally and then you'll
1:38:21
see this docker run command which is the same docker run command from the docs. I
1:38:27
think this should just drop in. Yeah. And so then now what you'll see it's doing is that it's running the same
1:38:35
container image from the space uh inside like locally but this time via docker
1:38:41
instead of the um instead of the uh python server the fast application.
1:38:49
So that's it. Uh let me see if I can also just interact with that like that just
1:38:56
to kind of complete the loop. I think that one should work. Yeah. So
1:39:01
now you've got like the same application running in the space which you which you can also use. Okay. So that's the
1:39:07
deployment side of it. The next little tutorial I've got uh is how you scale uh
1:39:13
these environments. So I'll just move over to that. Any questions while I'm moving across?
1:39:19
Uh yeah, I had two. Uh the first one was like what's the like UI framework for
1:39:24
rendering stuff? Is this like a gratio like based inspiration or is it like like when you're when you're in the
1:39:30
browser? This one here it isn't actually. This is a very simple just index.html with some
1:39:35
plain I see. Okay. um we considered getting Gradio involved but in most cases a lot of people were
1:39:41
kind of hacking these interfaces for their various environments and so we didn't see the benefit in standardizing
1:39:47
the UI too soon because um yeah a lot of there's a lot of experimentation going
1:39:52
on there like for example around like a connect for game or a wordle game like how do you visualize those games so you
1:39:59
can understand what's going on when I was running training loops and like um debugging processes I found this UI like
1:40:07
really powerful because I could create on the side like a little rendering of Wordle for example to get a sense of
1:40:13
what the model was actually working through in those like first few steps. So so so speaking of rendering like this
1:40:19
would work for like real-time games as well like I mean yeah I experimented with Pong and I
1:40:26
don't think this UI would would work in in real time games right that's where I think we would want to move on to a
1:40:31
fullyfledged UI. I see. Okay, got it. Cool. Okay. Uh so right scaling. So I'm
1:40:40
just going to give you uh a kind of community idea of how you would scale OpenMP and Zach is going to come back
1:40:46
with a kind of big boy scaling and this is sort of like hacker scaling and how
1:40:51
we get to like greater than one containers and these kind of things so that you can get like um you know sort
1:40:58
of um humble uh humble origins training runs and try to understand RL in the in
1:41:06
the early days as it were. So there are three main um ways to scale. Uh the
1:41:14
provider which is an integrated class and openn but I'll explain the websockets which are a way of having uh
1:41:20
sessions like multiple sessions on a single container running concurrently
1:41:26
and then um scaling the the services themselves right horizontal scaling. And
1:41:31
I took through these through a series of experiments to like basically see how far I could push each type of
1:41:38
infrastructure. So starting from hugging face spaces like free tier to paid servers on hugging face to um compute
1:41:46
instances and then multiple compute instances with low balancing and I kind of worked up those those first few steps
1:41:53
like up to let's say GPU middle class and how you would scale.
1:41:58
So the the first way and and the easiest way which most people have come to straight away is to use a provider and
1:42:06
at the moment there are um three providers implemented in Uicorn and
1:42:11
there's a Kubernetes one on the way. Uh but we could add more providers uh for
1:42:17
various different container runtimes. Right? So the first is a Docker provider
1:42:23
which uses your local Docker runtime. is the default one and and it's the the most straightforward. The second is a a
1:42:30
Uicorn provider which combines UV and uicorn to run Python only code uh inside
1:42:37
um for your environment. The third is a docker swarm provider which uses the docker swarm runtime so that you can
1:42:44
have concurrent environments for the for your uh concurrent containers for your environment. And so with these you can
1:42:52
run containers on your laptop uh pretty easily and get them to concurrent
1:42:57
environments. And so if you're doing um let's say like evaluation or inference workloads using an an environment you
1:43:05
can get pretty far with the with these swarm providers. And if you were going
1:43:10
to a single instance with multiple GPUs you could use the swarm provider and
1:43:16
scale pretty far there. Um but in fact we can already get quite far on on one
1:43:23
container uh with websockets. So this was actually um pretty much a purely
1:43:29
community implemented feature. So where we use as well as HTTP endpoints we use
1:43:35
a websockets endpoint. And if you're not familiar with with websockets they're a
1:43:42
persistent longunning connection from client to server. So a HTTP request will
1:43:48
send a message from client to server and back and uh the client will track the
1:43:54
the state and the server may remain um stateless and in that environment your
1:44:02
in that situation your environment is has a single your container has a single
1:44:08
environment that maintains the state. That's kind of ideal for situations like a coding environment or a file
1:44:15
manipulation environment where you want to um you want that to be sandboxed,
1:44:21
right? But in a situation where you don't necessarily want it to be sandboxed, you want to get concurrent
1:44:27
sessions running on a single container, then websockets become really advantageous. And that's where the
1:44:33
community kind of came in to implement this was because a lot of people were using environments like Wordle, Sudok,
1:44:41
Sudoku 2048, these kind of things and they were deploying single containers in
1:44:47
order to do it which became really resource intensive quite quickly. So
1:44:53
with websockets we can have say um parallel s sessions running inside a
1:44:58
single container and and basically it looks like this where we instantiate uh a new end on a
1:45:06
single container and every single time we instantiate an end we get a new session and so if that session becomes
1:45:14
idle as let's say the GPU as the GPU performs inference then it doesn't lock
1:45:20
up the container and it doesn't continue to consume any memory. So by taking this
1:45:26
uh and just to show you like the implementation of that that's what it this is what it looks like that every single time we start a new
1:45:34
session we just create a new environment so you have this connection between sessions and environments
1:45:40
and the takeaway from this is that instead of having a single container for every running end like HTTP we have uh
1:45:48
as many sessions as we need for the memory and compute resources of the the
1:45:54
process within a single container.
1:46:00
Okay. Uh so okay, I'm going to come back to the the experiments on that. Feel free to ask me any questions again. How
1:46:07
do we scale a single container? Well, that we can do with workers. So we can use Unicorn to create um to to run
1:46:15
multiple workers and to have the application running across processes. We can use asynchronous clients as well so
1:46:22
that we don't lock up the the server when we're um using the GPU for example
1:46:29
and we can do the same um and pass those to the um to docker. So the other
1:46:35
element to say is that uh which I've just kind of skipped over was that as we um as I said some environments like a
1:46:43
coding environment they want to be sandboxed and their ideal to be sandboxed and so we can pass a an
1:46:49
environment variable to those environments that says max concurrent ends is one. We say we don't want to run
1:46:56
multiple sessions in a single container. And actually when we build the the
1:47:02
environment we can define that within our skeleton and say um maxing current
1:47:07
ends is never greater than one. This is a sandboxed environment. Okay. So that's how you deploy it and
1:47:14
and that's how we kind of squeeze the most out of uh these single containers. We tried it on um Hugging Face and we
1:47:22
get up to concurrent batch sizes with something like Text Arena which is Wordle or Sudoku and and other textbased
1:47:29
games. We can get up to concurrent batch size of 128. If we use a paid space, we
1:47:34
can get up to concurrent batch sizes of 512 which is way beyond a kind of
1:47:40
comparably budgeted GPU. So um there should be a lot of room for people to try out spaces and let's say like just
1:47:47
set up quite simple experiments and even the CPU upgrade space is 3 cents a hour.
1:47:55
So we're not really talking about high investments here. And so what I experimented with was how
1:48:03
far we could push these ends to kind of to find these numbers basically.
1:48:09
And what we found, this is the the takeaway kind of image. What we found
1:48:14
was that at this number of around 512, we got to the capacity of a single 48
1:48:19
core CPU. Um, and then by using a load balancer, we could take that up um to
1:48:26
16,000 concurrent requests. And Zach will show you examples going kind of like even beyond that. And we did that
1:48:32
using a load balancer. So I I set it up with um slurm and a mvoy load balancer
1:48:39
but actually you you could deploy this on cloud providers using something like uh GCP's cloud run or or any kind of
1:48:47
managed load balancing service and you'd be able to deploy these environments quite easily
1:48:52
and the experiments that we run um
1:48:59
let me just zoom in a bit they um we found that these local Docker containers
1:49:07
were able to get uh we were able to get quite a lot of concurrent requests out of them. We could get up to 2,000
1:49:12
concurrent requests running on eight cores and with two nodes we could get up to 16,000 concurrent requests.
1:49:20
So that was our kind of I'll just work through the findings now of this little experiment. And uh the first finding was
1:49:27
that you can have like really high concurrent requests running on quite small devices because we're using the
1:49:35
websockets and concurrent sessions. So if you're doing workloads like evaluation or inference, these can scale
1:49:41
pretty far and you don't necessarily need to use a different environment tool for um local or training.
1:49:51
We found that the kind of the sweet spot number for hugging face spaces was around 128 concurrent requests and we
1:49:59
can go a little bit beyond that but then um we hit the hugging face hub rate limit in fact so it's not necessarily
1:50:05
the compute um we can get beyond that rate limit with um like premium hub
1:50:11
subscriptions like pro we can get a a greater rate limit um but in most cases
1:50:16
we probably don't need that the other finding was that yeah the um
1:50:22
multi-node scaling works and is pretty easy to set up and get to these high um
1:50:27
concurrent requests which are really informed by the the resource consumption of our environment
1:50:34
and this is what uh how they scale at each point. So the first one is the the spaces where around 128 they collapse
1:50:41
and that's just because they're getting rate limited essentially and then at these numbers of of 512 the single core
1:50:48
starts to drop away as the it times out basically and that's where we use the envoy load balancer to go to greater
1:50:55
than one. So, um, I'd recommend that you could take a look at this this experiment, but if you want to try out
1:51:02
on your own workload, uh, you can run these experiments. It's in a a project called OpenM scaling and you can test
1:51:10
run these tests on your own environment and kind of see to what scales you can get. But in most cases, the takeaway is
1:51:16
that OpenM can kind of is an ideal tool for like community users, but also um,
1:51:22
people that want to scale beyond that. So, that's the takeaway. Cool. Are there any questions?
1:51:30
Um, no. Yeah, I I I think this was quite clear. Uh, thank you so much, man. I really appreciate it. Like these
1:51:36
experiments. These are quite nice. Um, I guess like then like the sort of interesting stuff might also be like
1:51:42
sort of endto-end profiling of like bottlenecks like basically where do the bottlenecks show up like between like
1:51:49
just like scaling up the well maybe not because like I feel like this all like very
1:51:54
embarrassingly parallel. So it's like quite easy to scale and understand the performance characteristic of like
1:52:00
deploying large scale environments, right? So it's not something that I expect would have too many performance
1:52:06
footguns when doing RL training. No, the technology is is um familiar,
1:52:13
right? their web they're fast API web applications and you're using horizontal scaling and so cloud providers are kind
1:52:21
of full of like services and tools to scale these environments and so everyone will have a will have a preference. What
1:52:27
I think is really distinctive to um to an RL environment compared to like a web
1:52:33
application is the the reproducibility and like the the relationship that the
1:52:39
the environment has with the data set and and the model itself. And so that
1:52:44
that's kind of the next steps for for environments really of like how we connect a model that's published on the
1:52:51
hub to its environment in the way that we already connect it to its data set so that people can understand that
1:52:57
environment, read the paper and then kind of take away those findings. [clears throat]
1:53:02
Very cool. Well, uh thank you Ben. Um I guess we have three more talks everyone. I think the next speaker is going to be
1:53:08
Lewis. Uh so let me bring him on.
1:53:16
All right, cool. Do you want to share your screen?
1:53:22
Yeah. Can you see this VS code?
1:53:28
Yes. And uh I apologize for the audio. I am in a dingy office. Um so I'll try my
1:53:36
best. Um so what I thought would be good to
1:53:42
kind of build on top of what Ben showed is like okay we've seen now how
1:53:48
um these environments are like deployed on base hub and how you can interact with them but I thought it'd be useful
1:53:55
to sort of show like just one example how you put this in the RL um and I
1:54:01
thought it'd be fun to just take again a fairly simple example which is Wordle. Um, and you know, Wordle is a is a
1:54:07
simple game, but it is a very good example of a of the type of game where helps because it's got this kind of
1:54:13
sequential decision making where the kind of decisions that the model has to learn how to make depend on the previous
1:54:19
state. So, as Ben showed, um you can go
1:54:24
to like the the home base hub and you can um go to the open end
1:54:32
oops the open end organization
1:54:37
and here we have um bunch of like curated environments that have been
1:54:43
tried by us members of the community. Um, and basically in here it's a
1:54:49
collection um of these various environments. And so if you're looking to sort of get started uh on using open
1:54:56
end in an arrow loop, you can take some of these um and the one that I wanted to
1:55:02
look at was from Ben um where he has
1:55:08
example. So typical thing the demo gods are
1:55:14
against me today. So, um if you go to a space at an end and it it's broken for
1:55:20
some reason, then um what you can do is uh duplicate that space and then this
1:55:26
gives you as Ben explained kind of like full control um over that environment.
1:55:32
And so I already did do the duplication myself. And so here I can find
1:55:42
the space I made. So here
1:55:55
okay I have to find this space but basically it's um it's like this and then we can um either run this locally
1:56:02
but what I'm going to do is I'm just going to make you know websocket connection to it directly to the so we
1:56:10
instantiate our environment um and in Wordle the idea is you provide
1:56:15
some initial guess for the word um we're going to reset the state of the environment and then we're just going to
1:56:21
take a single step um with this action which is going to be the guess. So in
1:56:27
this particular environment um we make a guess and we get this kind of result
1:56:32
which has a bunch of information. It has an observation which we talked [clears throat] about is like the thing
1:56:37
that you know the environment gives back action and in that observation um we
1:56:43
have like the reward that was given um and we also have like something about the sort of the history of the state in
1:56:51
terms of the messages and some other things. So if we just pick out for example current state this is typically
1:56:57
what is getting or what is going to be provided to the environment step and
1:57:04
you've got kind of like a prompt that we're going to give to the language model and we're going to tell it that
1:57:09
you know okay um you get feedback like you know green if the in the right place
1:57:16
yellow if it's in the word but in the wrong place and so you can see here I gave it the
1:57:22
the guest crane and it told me that okay the letter C is in the word but and the
1:57:29
idea is I'm going to teach them more to kind of like iteratively get better making these guesses and so if I take
1:57:35
another step then now you can see that it's essentially concatenated
1:57:40
previous guess so here was my initial one and now I've got few guesses left
1:57:46
and now it's given me this from this I've learned that okay the letter C is in the word letter T would you we
1:57:52
[clears throat] could go on and so on. So the way this works in in TRL and it's
1:57:59
I think pretty similar in Unslaw um is that we're first going to instantiate the model we want to use um to run this
1:58:06
kind of like in in a single way we can take any sort of small qu model or whatever your model is um in general you
1:58:14
know if you're going to do RL properly you would you probably want to pick um a model that's like you know better suited
1:58:21
for the for the task you're trying to do but this is good for de purposes. And the the first thing we're going to
1:58:28
do is define kind of like fairly uh detailed system prompt just to sort of
1:58:34
like you know tell the model like you know some of these models maybe don't know about the game work and so we need
1:58:40
to kind of give it some information about the rules and in particular how do we want it to respond um at each step.
1:58:48
Um and then there are some like kind of uh you know tips on like strategy of
1:58:53
doing this and then a kind of example of thinking
1:58:59
and these things you know if we waited as like Daniel explained you know we can
1:59:04
just wait and have patience just wait a long time the model will you know eventually learn uh some of these like
1:59:11
you know rules about like strategies and stuff with working with small models it just helps a bit if you craft
1:59:18
system prompt just to get it kind of you know conditioned in the right place. So let's be system prompt and now the next
1:59:26
thing to do is to define what we call like a rollout function. So what this roll out function is going to do is
1:59:32
we're going to have prompts and each prompt you can basically think of as
1:59:38
like you know a simulation of like the word game. So what we're going to do is
1:59:43
we're going to um wrap these prompts to different sessions um of the word
1:59:48
environment and the model is going to have a shot at making a guess and then we're going to get the response on the
1:59:54
current state and then we're going to use the the resulting rewards to then do the optimization and then in the next
2:00:01
training step we're now going to basically have the history of these guesses and then we're going to update
2:00:07
and solve. So uh there's like some things that you have to specify like for
2:00:12
example um like the kind of things that you need to forward to to the trainer.
2:00:18
Um these depend on on the training framework. Um but open end is like agnostic to that. So the idea is that
2:00:25
open end exposes kind of standardized schema or an API and then it's up to the
2:00:30
training framework to then figure out okay how do I integrate that API then into um you know the training updates.
2:00:37
So as I explained before, you basically just iterate over the prompts in your batch. You generate an episode. So in
2:00:45
this case, we've got like six turns. So the model has basically six attempts to
2:00:50
kind of guess the final solution. And then based on that, we're going to get um some information,
2:00:58
you know, to sort of whether the whether the model is correct, guessing final answer, and we give it some like kind of
2:01:04
partial rewards. um because we wanted to incentivize it to recognize that for example if there's a letter in the
2:01:12
correct position that it should not just randomly change it and that's what for example green award is doing whereas the
2:01:17
yellow reward is incentivizing the model to essentially if it recognized that it got a letter that's correct in the word
2:01:24
but it um it's in the wrong place so it should then learn to eventually you know reuse that different location and this
2:01:31
is something that like I think Daniel alluded to a bit but this idea of like How you shape your rewards um is
2:01:38
actually quite important because um it's not just a question like you know if we just told the model like what is correct
2:01:45
or not just based on whether it's solved the episode this would probably still work but you would have to wait like
2:01:52
considerably longer um in training um versus providing it with like this kind
2:01:58
of like partial reward around you know you already know that if you got a green
2:02:05
a letter in the correct position then you should just keep it. So these are the kind of decisions that one makes as a model trainer is a bit like sort of
2:02:12
how much shaping of the reward you want to provide that depends on the task difficulty and you know
2:02:20
so we're going to define that roll out [clears throat] uh step and then this is
2:02:26
something that we do in TL to basically just define like a single roll out. I'm not going to go into it. It's just a
2:02:32
bunch of um sort of like code but roughly speaking you want to just figure
2:02:38
out okay um given a single prompt how do I then go all the way from feeding that
2:02:44
to the environment and then so here we're going to take some number
2:02:49
of turns and then for each turn we're going to build up um essentially the
2:02:54
state of the messages we convert this into prompt um that we can then generate
2:03:01
rollouts from and basically we compute our roll out. So these are going to basically be the guesses at every step.
2:03:08
We extract the guess with the path and then we do the [clears throat] um part
2:03:14
with open end where now we've got a guess we feed it to the environment we get a result and from that result we can
2:03:20
extract observations the score and also any other types of reward. So this is
2:03:26
like a a kind of standard like little bit of logic that typically one includes to sort of specify precisely how the
2:03:33
rewards are obtained from the environment. Um but in general it's just
2:03:39
a little bit flow that you know nowadays clin
2:03:45
function there are some helper functions but we don't need to look at them um and
2:03:50
as I mentioned before now we just going to define these reward functions so we're going to tell the model that it
2:03:56
gets reward if it gets the correct you know final word and we're going to give it these like partial rewards for like
2:04:03
basically getting letters in the right place and also for getting letters in that are to do this and the final step
2:04:11
is to kind of define our data set which in this case is just a very simple prompts world like an expert
2:04:19
and then [clears throat] we're going to define now the kind of like hyperparameters that drive the training
2:04:25
and I won't go into these in too much detail but there just standard things that you typically get with reinforcement learning so typically want
2:04:32
to define your batch learning rate how many epochs from data there. Then you
2:04:37
want to define like kind of size the group how long your rollout is going to be
2:04:43
because it's Wordle and you know the words at each guess we only need five letters
2:04:49
we don't need tokens um and you know this is all running like
2:04:55
in the back end so it's easier but this is like you know pretty standard stuff um if you've done
2:05:02
so we set that up uh we instantiate um our trainer. So we basically provide
2:05:09
the model, the reward functions, the data set and the logic for how we do
2:05:14
these rollouts. Um and this is now going to basically load the model. Um and kind of get
2:05:20
everything set up training, bunch of warnings, whatever doesn't
2:05:26
matter. Let's get to the meat of it. And then once that's ready, we can then hit
2:05:32
train. And so this will take maybe 10 seconds. There we go.
2:05:41
Maybe while this is going, if there's any questions, Mark.
2:05:50
No, I I think I'm just watching. No questions on my end. Okay. Okay, cool. So, this has now
2:05:57
kicked off training uh with this uh environment. And um one of the kind of
2:06:03
things that we're building at hugging face is like kind of local first fully like open source uh project for
2:06:11
experiment. This is called track and what this will do uh in TRL is it will
2:06:17
then um you know you can run this locally and then you just inspect the logs locally but it will also create um
2:06:24
a space uh for you. So if I go to this space now [clears throat] this base is uh
2:06:30
collecting all the metrics um that exposed by the TRL trainer and so we get
2:06:38
you know all the things that are useful for like kind of diagnosing our run the entropy um kind of fractional like
2:06:44
distribution of the rewards the variance all those kind of things and yeah basically you will be able to then you
2:06:52
know monitor the reward uh of training over time and I would say that's
2:06:58
probably it. I think uh I don't need to go into too much detail. This is just a very quick example of like how you can
2:07:06
in interface open into specific training framework.
2:07:12
Um, so Lewis, a question I have here is like um, uh, at least like for your own
2:07:18
personal experiments, like I think sort of spinning up something on a hugging face like might make sense if you're doing
2:07:24
like a one GPU sort of RL experiment. Uh, and so like at least for your own research, uh, just because I you've
2:07:31
covered spectrum from poor to rich throughout your career. Like what's sort of the sweet spot at which you run
2:07:36
experiments where you're like I just want to see if this thing works or not? Yeah. Are you often running on a single GPU, a node more? Could you sort of
2:07:43
speak to us a bit about how you make these decisions? Yeah. Yeah. So, I guess there's like two. One is like um at what scale of
2:07:50
compute do you run? Um just your training like is it like just general
2:07:56
principles of like how do you start your training runs? Um and in that context um
2:08:02
the general like again it depends on if you're doing SFT or RL but generally
2:08:08
trying to get the simplest possible like sort of yeah I'm mostly referring to RL here
2:08:14
basically. Yeah. Yeah. So, so in in the context and uh
2:08:20
particularly with like um environments um like at face we use slurm. So we have
2:08:25
our own cluster and we use slurm to kind of basically like automate a lot of the
2:08:31
scheduling um of the different pieces that go into dr. So things that we have to consider are
2:08:38
like the VL servers. um you want to create essentially nodes training and
2:08:44
doing optimization and then you need nodes for um the environments and they
2:08:49
live on different devices right so obviously for training and inference you want GPUs but for most of these
2:08:55
environments you want like GPUs and slurm is very good at basically just
2:09:00
creating very large partitions of you know these environments which then you
2:09:05
can directly interface into your training so I would say in like a real world case that's kind of
2:09:12
um but to start with you you typically want to simplify as much as you can and so in that context most of what I'm
2:09:19
doing is typically on like one node hus
2:09:25
for like small training and then you know with like one environment um it's
2:09:30
kind of like you're just trying to check are you getting signal do things work
2:09:35
does does the work well um but then you want to do it properly,
2:09:40
you then do the scale typically at least for base we're talking like roughly say
2:09:46
10 nodes of of GPUs 80 GPUs for like sort of mid scale small scale run and
2:09:52
then some and then [clears throat] so for your training code are you using like sort of
2:09:59
like some of the nano work that Noman has been working on like is it like more like torch titan based is it like
2:10:04
pytorch is it deep speed what's worked well for yeah so that's a really good question. Um, as you may know, the space of
2:10:11
postraium frameworks has exploded. So, there's like I [clears throat] think 20 different uh frameworks and they all
2:10:18
have like same strengths and weaknesses. In the context of RA um there's actually
2:10:23
like I would say a broad spectrum um of like which frameworks are most suitable
2:10:28
for which thing. So for example like Unsplot is extremely well suited in like
2:10:34
the Bruce constraint setting like you know Daniel has done amazing work getting you know Laura and Qura and like
2:10:41
all these different tricks to work um in that way. Um at the other extreme if
2:10:47
you're trying to like train like this like 100 billion prime model with RL um you're going to need some like pretty
2:10:54
dedicated um like framework for that. And in that space there's not too many
2:11:00
uh options. There's a primaril from prime intellect. Um there's um a few
2:11:06
Chinese labs with open sourced things like slime that's from um I think it was
2:11:12
Z AI. And then there's like this Miles framework and Torch Forge. So these are
2:11:17
like kind of like the heavy duty stuff. And in those in the in those cases what you trade like if you look at the
2:11:24
extreme of like unclo you know these other ones you trade the ability to just do stuff on a notebook
2:11:31
or like okay now I have to start thinking about like all those kind of let's say
2:11:37
slightly annoying things. So um and here in the middle it's like you know it can
2:11:42
kind of scale to a certain range but it it's like it's not really built for like these like you know very very large um
2:11:51
Okay, got it. Well, thank you so much, Louis. I guess our next speaker, if I remember correctly, was actually Daniel.
2:11:58
Uh, yes. Hello. Well, actually, I had questions actually. Um,
2:12:03
yes, go for it. I was actually going to say like I actually recommend people to use TRL and like you know, even for large training
2:12:09
runs like it generally works pretty well. Um, for like a previous slide for track IO, um, is it like does it go
2:12:17
inside the collab notebook? Does it just run inside of it? Yeah. Yeah. So, basically um yeah, track
2:12:24
or track, I don't know how you pronounce it. Yeah. Yeah. You can um you can run it in in a
2:12:30
bunch of ways. So, you can be purely local. So, if you have like Python train script, it will then spawn locally um
2:12:38
like an endpoint, you can then interface with the radio basically running
2:12:43
locally. Um or if you're running a collab notebook, it will then, you know, you can create the the dashboard in the
2:12:51
notebook for you. So instead of like having to go up into spaces, it's there. Um or like for me personally, the
2:12:57
advantage of using like spaces is that then you've got persistence. So like in
2:13:03
you know if I accidentally you know delete the logs locally at least I can you know see this uh kind of on the hub
2:13:11
and uh the advantage of the space as well is like for sharing with my collaborates right it's easier to say
2:13:16
hey here's a link to a space look at this experiment and you know you can group them you can learn
2:13:24
yeah that's pretty cool like I yeah because I was like I didn't actually know before when I was using trackio
2:13:30
like you know it can actually render inside of notebooks. And I was like, "Oh, okay. I'm shocked." Um like, you
2:13:36
know, you don't need to like access some other like um you know, logging system. It's just inside of the notebook itself.
2:13:42
And you know, you can see the plots move in the notebook. Um and I was like, "Oh, okay. This is pretty cool." Um yeah.
2:13:52
Okay. I Okay, I guess I will go to Okay. Um I guess it's I I was going to just do
2:13:58
a repeat of Louiswis's um one like you know showcasing how you can do a notebook but I think I'll focus a bit
2:14:03
more on the reward hacking side of things um how to like actually like you know for example so unsoft is built on
2:14:09
top of TRL and we add a few extra things um you know how to do data preparation how to make it a little bit more
2:14:15
efficient and stuff like that so I'll probably showcase that um wait I'll share my screen
2:14:20
um I think and Louis if you like we can just keep you then on the on this is
2:14:25
visible. I I'll I'll go away here then Daniel want to share your screen. Okay. Yes. I will Can you guys see that
2:14:32
the open environments page? No. Yes. Oh, okay. Yeah. So, like you know
2:14:37
definitely go to um you know you can just type in Google open m um and you'll get to like this GitHub page um you know
2:14:44
definitely star um the package. There's a lot and a lot of environments that you know people can you know utilize. Um
2:14:51
yeah there's like so many um but you know if you scroll it down um there's like you know how to use open
2:14:56
environments um and then there's like some examples um and then we'll be using the unsoft collab notebook there's a TR
2:15:02
version and other people's um you know partner um you know notebooks but if you click on unsoft's collab notebook um so
2:15:09
this is what we offer to people um and this actually can run directly inside of a collab T4 which you can utilize for
2:15:16
free um and so the main goal for this notebook is how to do the 2048 game
2:15:22
using reinforcement learning. So I was talking about this previously um at the very beginning phase um for
2:15:27
reinforcement learning um the model um doesn't really know how to play 2048. Um
2:15:33
and so the goal is how do we make reinforcement learning um force it to
2:15:38
play the 2048 game by generating a strategy. Um and so that's kind of the process for the notebook. um you know
2:15:45
this so this notebook will run directly um and it should be for free for people to use I think it's like I think people
2:15:51
get three hours of free compute from Google Collab um per day or something um
2:15:56
if you do run out of credits you can like utilize um Kaggle which has 30 hours for free um per week um hugging
2:16:04
face um also like you know definitely check that out like you know they also have some notebooks and some compute
2:16:09
from hugging face jobs for example um so try that out as well um but like to for installation we have to install some
2:16:15
stuff. It looks a bit complicated um but it's actually just pip and store unsoft and we're also going to be using track
2:16:21
io as well um to show you know to show the notebook um you know show the training loss inside um and you know you
2:16:27
have to like you know get the open environments um uh GitHub repo um and
2:16:33
then we'll be actually fine-tuning GPUs um you know Zach was talking about GPUs like you know GPUs is a great model to
2:16:40
fine-tune um it's already a reasoning model so you know many people keep asking me why would you you Why would
2:16:46
you train a reasoning model um you know to do more like reasoning like what's
2:16:51
the point? Um so the biggest problem is for reasoning models um they are very good for generic tasks but if you want
2:16:58
to specialize them into like one specific task for example just to generate a strategy for 2048 or to
2:17:04
create you know to do weather simulation or to you know um do stock trading like generate strategies to trade stocks for
2:17:10
example. You don't want the models you don't want the model to have like generic capability. you want to somehow
2:17:15
um train, you know, do RL or fine-tune the model to just focus on this one specific task. Um and so we, you know,
2:17:22
this is the whole purpose of the notebook to showcase that if you just want if you want to like do RL to solve
2:17:28
2048, um it's actually not that complicated. Um and so you know we do a lot of optimizations on unsoft side to
2:17:35
like reduce memory usage. For example, we'll be using 4bit. Um this is like dynamic 4bit. So you cannot quantise
2:17:41
every single layer down to 4bit. um because you will damage the model. Um so some important layers are left in higher
2:17:48
precision like you know 8 bit or 16 bit. Um we also for example offload the embeddings to RAM to like save even more
2:17:54
VRAM. I think this saves like 2GB one to two GB of VRAM and as Louis was talking about we'll be using Laura. Um so Laura
2:18:01
what Laura does is you don't need to train the entire model. You essentially
2:18:07
add 1% of extra weights to the model. Um, and you only need to train these extra weights. Um, and interestingly
2:18:14
enough, this actually works. Um, and so the main reason why this works is, um, you know, models, you don't need to
2:18:20
train every single parameter if it's like some specific task. Um, if you want to have like if you want to specialize a
2:18:26
model, you only need to select a few parameters of the model to update. And this is the Laura um, Laura um, weights.
2:18:32
You don't need to update the entire model. Um so you know once you in you import on
2:18:37
stuff and stuff like that you will see some stuff like this. Um but then you know this is to add
2:18:43
yes. Um so I think historically like um with
2:18:48
like Laura and then you know Qura um people were like okay it probably works
2:18:54
for SFT but maybe not for IRL. And I think one of the intuitions there was
2:19:00
that um because you're like you've got this like you know small adapter um
2:19:05
maybe you know you're not going to get the full capacity to actually learn um you know the the kind of meaningful
2:19:12
signals you want and so the question I have for you is um like we know from now this work by thinking machines that like
2:19:18
Laura does really work um but you you use Qur a lot and have you also had
2:19:24
success doing this across a range of tasks like does the quantization affect these in any meaningful way?
2:19:31
Yeah, that's a great question. Um, yes. So, Thinking Machines released a blog post um, Laura Regret. Um, we actually
2:19:38
collaborated with them coincidentally on that. Um, so we actually had a section inside of the blog post um, showcasing
2:19:44
how you can set good parameters. For example, in the blog post, they mention how you need to select good Laura alpha.
2:19:50
you know, you need to target all of the parameters like you don't you're not just supposed to do Laura on the attention matrix, but you must do Laura
2:19:56
also on the MLP. Um so um you need to do Laura on everything um and select some other good parameters. Um but I think so
2:20:04
for Q laura um so what Qura is um and difference from Laura is Laura is you
2:20:10
leave the model in 16 bit so you don't do any quantization. Q Laura is you quantize the model back to four bit um
2:20:16
to essentially you know reduce memory usage and you know can fet it on you know much less memory um you know and
2:20:22
use like consumer GPUs for example um or you can fit like ginormous models that don't even fit on like you know GPUs and
2:20:29
you can like use 4bit um so the trick for Qura is once you finish the Qura
2:20:35
process you do not merge the model back into the four-bit weights you only take the Laura weights and then you merge it
2:20:41
back to the 16 bit original model. Um, and so that's what we found to work very well. You should not actually, so you
2:20:47
should do the training process in Qura, but then you discard the Qura directly. Um, you discard the 4-bit model and then
2:20:54
you take the Laura weights um that you learned via Qura and then you merge it to the 16 bit. Um, and so in UNSOFT, we
2:21:01
actually do this automatically. So we don't like we don't like upcast the 4bit to 16 bit and then we merge it back. We
2:21:07
actually download the original 16- bit version and we put it back. Um and so that's what we found to work very well.
2:21:13
Um yeah, very cool. And my final question is um there's a lot of discussion about this
2:21:18
like trainer inference mismatch, right? So the kind of precision that the model is for generation versus the precision
2:21:24
you use for the optimization. Uh when you do Qura with like onslaught, do you
2:21:31
treat everything the same? So like the inference is in in like with the Laura
2:21:37
with the adapter um or is it like you know you do the merging for the inference and then you switch back to
2:21:43
the train. That is a great question. So what we found to work very well um with the lowest error rate um is you need to so
2:21:51
we do in unsoft what we do is we do something called weight sharing where we actually we launch a VLM instance um and
2:21:58
then we essentially suck the weights out of VLM um into the unsoft land um and
2:22:03
then essentially what we do is we can cut memory usage by two. So previously in the olden days when you have inference and training. So reinforcement
2:22:10
learning you need to have inference and training you need to essentially double the weights that you have to have because you know the training side is
2:22:16
like um you know updating um you know the inference side is also um you know you have to like sync the inference and
2:22:22
training side. You can use like for example standby um or like you know vlm sleep feature to like make this a bit
2:22:27
more efficient. Um and we also like for example have that. Um but what we found to work very well is if you are doing Q
2:22:33
laura the villa weights must also be in Q laura. Um if you leave if you do like Laura for example um then yes you will
2:22:40
get divergence. Um and so during the training phase everything has to be the same precision right if you do cua for
2:22:47
training um inference must also be cura. Um however when you do inference so when
2:22:52
you finish the RL process you do not need to do curora. um you can just take the Laura weights and you know merge it
2:22:59
back to the original 16- bit model or even just do Laura in VLM. Um so we actually found that to work very well.
2:23:05
Um so yeah I that kind of answer your question. Yeah. Awesome. It's really interesting.
2:23:11
Yeah but yeah we also yeah so like continue like you know for example like gradient
2:23:16
checkpointing um definitely use this method. So we offload we offload gradients to RAM asynchronously and then
2:23:23
we bring it back. Um so we introduced this like I think like like two years ago or something in February uh
2:23:28
September 2024 and I think like you know many packages utilize this now. It's very very useful. Um you can increase
2:23:34
context lengths to like very very long. I think GPUs is like 500k context lengths by using this methodology. Um so
2:23:41
this is pretty useful. Um but yeah like once you launch an open environments um you know open environments open m has
2:23:47
like many many many environments and all open source and you know for example we'll be using the 2048 environment um
2:23:53
from open um and to launch this inside of a collab you actually have to do a little bit of um you know you have to do
2:24:00
a bit of tricks to make it work. Um but you can launch the open environment um you know 2048 inside of this collab um
2:24:07
you know workspace. Um and then for example if you try the 2040 game you
2:24:12
know to try the 2040 game you can see the state right this is the state um what are the legal actions um and so
2:24:20
this is kind of like what open envir um open m provides um and you know we don't
2:24:25
we don't just want to look at the state because it kind of looks a bit um you know we we want to somehow render this
2:24:30
um and so we made some function to like somehow render it into like a nicer nicer format um and then we actually
2:24:36
called a language model to generate some asy art for this. Um so for example the current board looks kind of like this.
2:24:42
Um if we do some actions for example um you know the board will update. Um so
2:24:48
open m actually allows you to execute actions for the current environment. Um and so for example this action um you
2:24:55
know we do the action zero and it does something. Um and then we do the action one we do the action two and we do the
2:25:01
action three. Um so one of the questions for reinforcement learning is you know what does action 0 1 2 and three
2:25:08
actually mean. Um and so like you know this you know we essentially force the language model to learn what these
2:25:14
actions actually are doing. Um so we don't actually tell the language model oh okay you know action zero is moving
2:25:19
up or something right action three action three in this case is moving left for the 2048 game. You don't actually
2:25:25
have to tell the language model that you know what these actions actually mean. Um and so during the process of RL the
2:25:32
model will essentially learn what zero one two and three actually are um these types of actions. Um and so we al in
2:25:39
openm we also get legal actions. So for example what is the next actual um action that you can actually take um so
2:25:45
some actions are forbidden. Um for example in chess for example um you can't do some actions for example um so
2:25:50
open m actually limits these and so actually for the environment setup um there is some code um to like
2:25:57
call open sphere from open m and many other things um and then the most important function we um provide to
2:26:03
people is called execute with time limit um so unsoft has this function called execute with time limit where we set it
2:26:10
to 2 seconds um we do not want a process to run for infinity right so we don't
2:26:16
actually want this. Um, and so we actually have a decorator to decorate any single function to execute a
2:26:21
strategy in a specific time limit. Um, and so this is actually very important because pretend some sort of strategy
2:26:28
executes for 10 minutes. Um, or executes for infinity, right? We don't actually want this to happen. Um, and so this
2:26:35
essentially guards these functions. Uh, what what happens if if they have to
2:26:41
end fast? Oh, sorry. like which action ends up getting picked basically like if you
2:26:47
have a very tight time limit. Oh yes. Um so if you have a t so generally we found 2 seconds to be
2:26:52
actually okay. Um you can increase it to like for example 10 seconds like if you want to have like a longer training run
2:26:58
maybe increase it to 10 seconds. Um but in general because it's just generating a strategy. Um the strategy you don't
2:27:04
need to so the execution of the strategy is separate from the actual generation of the strategy. So this execution of
2:27:11
the strategy we don't want it to last for too long um because otherwise it would just be very it would just take
2:27:17
too long right so two seconds generally for execution is actually not that bad um because you know CPU execution of the
2:27:22
environment isn't that long um but the generation so the generation is actually fine we do not put a time limit for the
2:27:28
generation of the strategy um does that kind of make sense or it does yes okay we also for example convert the
2:27:35
board um to like a more nicer format for it to like you know um to process Um and
2:27:41
so there's lots of functions which we provide to like make the process easier for people. Um for example, let's try a
2:27:47
strategy, a very dumb strategy. We just move left, right? Always move left. Um and we see it timed out, right? So we do
2:27:54
not want this strategy. Um this strategy is not very good because the only thing it does is it just keeps moving left. Um
2:28:00
so essentially this guard allows us to filter out um you know like repetitive
2:28:06
repetitive um you know repetitive actions, repetitive strategies and also strategies that go in a loop. Um and so
2:28:12
if strategies that go to a loop, this function can help like stop stop those types of functions. Um yeah. Oh yeah.
2:28:18
For example, here we instead of doing two seconds, we actually set it to 5 seconds in the notebook. Um
2:28:24
and so now I'll be talking about reward hacking for example. Um so you know previously reward hacking does cause
2:28:30
problems for reinforcement learning. Um so now I'm going to tell you how to like guard these types of issues. For
2:28:36
example, the first one is how do we stop um how do we stop you know invalid code execution. Um and so for example um this
2:28:44
is fine right? If you import generic Python functions like you know the math library you know types and stuff like
2:28:50
this this is fine. Um so we essentially we expose a function called check python modules. uh check Python modules inside
2:28:58
of Unsoft and this essentially checks is the function you know calling some other function that is not a generic Python
2:29:05
import. Um so this is good right so like this is true so this is very good however if we import numpy um this is
2:29:14
not good right so essentially this function checks dynamically what packages are imported inside of these
2:29:19
functions um and if this is not good then you decide okay I can see this is a numpy function um do we penalize this um
2:29:27
do we like you know continue on um and so this is one way you can stop like you know reward hacking you know like for
2:29:33
example we just tell the model to generate fast matrix multiplication kernels but then all it does is it uses
2:29:40
numpy or torch pytorch right so we don't actually want that to happen um and so the goal of this function is to stop
2:29:46
stop the strategy from you know stop the model from generating strategies to use pietorch or to use numpy um if you want
2:29:54
to make faster matrix multiplication for example um yeah so definitely use this function
2:29:59
in the context of 2048 yes how could the model how could model cheat if it had access to nai like could it
2:30:06
That's that's a great question. I think I think in the context of 2048 um that's
2:30:11
probably unlikely actually. I think these functions we just expose for like generic notebooks. Um so for example we
2:30:17
have a matrix multiplication notebook as well. Um that will this function definitely will help for that. Um I
2:30:23
think for the yeah for the 2048 game okay I can give you an example. Um this is a very dumb example but for example
2:30:29
let's say we let's say inside of the function it actually calls another language model um for example let's say
2:30:36
it calls Gemini or GPD5 um let's just say that happens or it calls some sort
2:30:41
of like server um which lets someone play the game um it generates a strategy
2:30:47
by itself something like that right so like essentially we disallow any single function which does something like this
2:30:53
um you know like for example importing open open AI's library for example that's not allowed. Um, importing
2:30:59
Gemini's library, that's not allowed. Um, so I guess that's it's probably not a good example, but something like that.
2:31:04
Um, yeah. Yeah. I was kind of curious like how do you think about like, you know,
2:31:10
the prevention of reward hacking like a prior, right? Because I guess we have intuitions of like things we don't want
2:31:16
the model to be able to do like the examples you take. Um but what kind of signs do you look for in training um
2:31:23
that give you some hint that you know the mole has been hacked because you know you did say oh the reward is going
2:31:29
up everything is great but the reward is going up because of the hacks and I'm curious like you know what what do you
2:31:35
guys do at onslaught to sort of understand like okay I've been hacked basically
2:31:40
yeah that is actually the that's probably that is a question also for large labs like I think everyone's
2:31:47
asking this question it's actually the most it's one of the most important questions. Um the main issue is we don't
2:31:52
actually know if a model is reward hacked or not. Even if you use LLM as a judge, for example, if you ask a
2:31:58
language model to, you know, verify if the R output is reward hacked or not,
2:32:03
it's pro it's still very hard, right? How does a language model even know it got reward hacked? Um, and so generally
2:32:09
what we tell people to do is every single like, you know, 10 steps or 100 steps, you would sample um you would
2:32:14
stop the training run or like just look at the output and manually inspect, you know, does this, you know, does it
2:32:20
actually look correct? Um, so I would say it's more like a vibe check. Um yeah the other approach we had is like um you
2:32:26
should for example you you ask another language model like during the LLM as a judge phase um and you don't actually
2:32:32
touch it at all right you don't train it you don't use so there is another approach you can use the same RL model
2:32:39
um to judge itself um that's what people like to do but our view is you probably shouldn't do this you shouldn't do this
2:32:45
like you know using the same model to judge itself approach you you would rather just use the previous SFT model
2:32:51
you know at the very very very beginning of training and they use that to judge the model. Um so that that kind of
2:32:57
works. Um I think yeah in general
2:33:03
in general what we find I think from experience if you're if the reward if
2:33:08
the model is reward hacking um h that is a good question. I don't Yeah.
2:33:18
One thing I've seen myself um is if you look at the lengths of the rollouts. Yes.
2:33:23
Yeah. Exactly. It's often like a a hint when things are going off the rails. So even though your reward curve can look
2:33:29
great if your rollouts are like going to infinity, it suggests that like you know
2:33:34
the model is exploiting you know some peculiar tokens which are getting you know high rewards. No, I yes I was
2:33:41
actually going to mention something to do with length. um looking at the metrics so looking at for example using track io to look at the metrics um you
2:33:48
know definitely look at all of them that is very important um I think actually there is one way so actually I did
2:33:54
remember so if you make more reward functions generally you won't see your
2:33:59
reward hacking um if you make independent reward functions and not just one um so generally what happens
2:34:06
for reward hacking is people just make one reward function um for example is the maths you know is the code is the
2:34:12
code output good or bad? Um, is the maths, you know, is the maths formula at the very end correct or wrong. Um, and
2:34:19
if you have one reward function, you don't actually know if the model's reward hacking or not. Um, so what
2:34:24
normally people should do is you should make multiple reward functions that are independent from each other. Um, and if
2:34:30
you can do that, generally you can like reduce the probability of reward hacking. Um, so for example, you know,
2:34:36
for maths, you know, do LLM as a judge. Um but like do don't just do one model like you know call the model 10 times
2:34:43
for LLM as a judge um and check um or maybe you know yeah something like that but like essentially make more reward
2:34:50
functions um and you can reduce the probability of reward hacking. Cool. Awesome. Thanks.
2:34:57
Um yeah so like oh that's a numpy example. The other very important one is
2:35:02
you want to create a locked down function um which disallows global variables. Um so one of the things in
2:35:09
reward hacking is the language model can actually write to the global space of
2:35:14
your programming language. Um so that is not good. Um especially if you're doing
2:35:19
like some sort of benchmarking or kernel creation, it can actually do caching. Um so if it does caching of the results,
2:35:26
you will actually get incorrect results. Um but your reward is still going to go up. Um and so this you know locked down
2:35:32
function approach essentially locks down the you know function to only call um
2:35:37
you know non-global um variables. Um and so you know this is definitely very helpful as well. Um for example we did
2:35:45
not so this is a this is just a numpy example. We didn't actually import numpy over here right? So it's supposed to
2:35:50
it's supposed to actually do import numpy. Um but we didn't actually do this
2:35:55
um because essentially np was a global um import um and if you look at this you know create lockdown function this is
2:36:02
disallowed um and you know will tell you why this is disallowed because numpy is imported. Um so definitely also use this
2:36:10
um yeah for example this is the one that succeeds. You can like you know nested functions are fine. Um so this should
2:36:16
succeed. Um and yes it definitely succeeds. So create lockdown function um supports nested functions um and
2:36:23
recursive functions. And so this is the most important part
2:36:28
for reinforcement learning. Um if you want to create a 2048 strategy um the
2:36:33
trick of reinforcement learning is you only need one prompt. Um and so the
2:36:38
difference so I always get the question what is the difference between supervised fine-tuning and reinforcement learning? Um if you have lots and lots
2:36:46
and lots of data, you should do supervised fine tuning. Um if you do not have data, you should do reinforcement
2:36:53
learning. Um for example, um for the 2048 game, I don't there is no data,
2:36:58
right? I'm telling the model to create a short 2048 strategy using only Python code. Um and this executes the strategy,
2:37:06
right? This is the only prompt I give the language model. It's just one prompt. Imagine you can like you know customize this you know create a trading
2:37:13
strategy to do blah blah blah or you know do some weather simulation you know predict some new chemical or something I
2:37:19
don't know but essentially this prompt you only need to engineer this one prompt um and you replicate this many
2:37:25
many many times um for reinforcement learning Daniel I had a quick question from
2:37:30
section we just went we just passed which was really about the the reward functions
2:37:36
um and it was basically in short have you never explored like pure rubric
2:37:41
functions. So rewards that we don't necessarily use to update the model. I know when we were doing Wordle, we use
2:37:47
that a lot. Um mainly for the smallest models like under a billion parameters
2:37:52
where they would for example just repeat themselves five times. So we wanted to just understand like what they were
2:37:57
doing and kind of track that over time but not necessarily reward that like be better signals for reward. So do you
2:38:04
mean like for example if the model kept repeating like a strategy or like for example as you said repeating um would
2:38:10
you like just like how would you use the rubric? Would it be like just discard the output or would you like reward it
2:38:17
somehow? So basically it yeah it might just be a specific artifact but it in this example
2:38:24
the models these small quen models consistently created the same guess.
2:38:29
Yes. And we found that we had a richer reward function that was for yellows and greens in that game. And it started off
2:38:35
quite low and it would build up and it was just very rich reward function. And penalizing repetition just didn't
2:38:42
have as much information. So in TRL what I could do is just like say the weight of that reward function is zero. So
2:38:48
ignore this reward function. But then when I'm watching and monitoring my training process like I've
2:38:54
got quite a good view of like what's going on. Okay, this is another kind of repetition run. Let's try something
2:38:59
else. Go back. Yeah. So like so generally what we see
2:39:05
um and we tell people is like you can also do something called um dynamic waiting. For example, if you find your rubric function or like some sort of
2:39:12
style check at the very beginning um is somewhat helpful like you know it's it's it's somewhat helpful but then over time
2:39:18
it's not that useful. You can essentially decrease the weight over time to make it go to zero. Um so you
2:39:23
don't actually have to set the you know weights to go to zero like you know weight as zero at the very beginning. Maybe you should set the weight to be
2:39:29
like 10 or something like some sort of number. Okay. So obviously has to be normalized. Um but essentially you can decrease the weight over time. Um I'm
2:39:36
not sure if that does that kind of answer your question. I'm not sure. Maybe that's a really cool idea. Yeah, I
2:39:41
definitely think that would help in that situation because that was that was the problem. After a certain number of steps
2:39:47
that function became like regressive basically and it probably would have learned from it
2:39:52
in the first kind of hundred steps. So cool. I'll take that away and try that out. Like another good example is like
2:39:58
length for example if you want the model to do okay um you should not at the very
2:40:03
beginning of RL the responses shouldn't be that long um so I would like for example penalize length um but then over
2:40:10
time um you know the model gets you know the model has to understand more complicated questions then you would
2:40:15
like not penalize length right you would decrease the weight of the penalization of length over time um something like
2:40:20
that um and so like these dynamic waitings of the um you know reward functions I guess that's another that's
2:40:27
another hyperparameter. Um another turning knob. Um but I guess like um yeah but definitely like you know
2:40:33
dynamic waiting is like a active error research for people. Um yeah cool. Okay thanks for that.
2:40:41
Um but yeah like for example we create this one prompt for 2048. Um and then if we call GBD you know GBDSS um it does
2:40:50
okay right? It generates some sort of strategy right this is no RL right no RL at all. Um and so it does generate a
2:40:56
strategy. Um and then you know during the overall process we want to improve the accuracy of GWSS. Um and so we
2:41:03
actually created some reward functions as well for 2048. For example um the function works no cheating and strategy
2:41:10
succeeds functions. Um but firstly we actually need to extract the function. Um so previously I said like you know
2:41:16
for example you need to generate it between three back ticks. Um and so we need to somehow extract this from the
2:41:22
generation. Um and so this function kind of like extracts the strategy. Um
2:41:28
and then we for example we use the check Python modules function to check you know does actually the function the
2:41:34
strategy um only use Python functions and not like numpy or pytorch or something like this. And so this you
2:41:41
know we utilize this function inside of this function works um uh reward function. Um and if we find an error
2:41:48
then we penalize by minus two. Um if we try a lockdown function so we try so you
2:41:54
know because of Python Python's very good because we can do like try and accepts um and you know if we create a
2:41:59
lockdown function and it succeeded we score a one um and if it failed then we score minus 0.5 um another question I
2:42:08
always have is you know how do we actually allocate these numbers like what what can they be um and so for
2:42:13
reinforcement learning they can be anything you like right it can be minus 20 minus you know 10 you know minus
2:42:19
whatever you like Um and so these numbers are up to you to decide but in general um I see like normally people do
2:42:25
01 you know binary or just minus one and one um but it's up to up to the um
2:42:31
practitioner to decide. Um if either of you could interview for
2:42:38
like I don't know could intern atropic for a day like what would you do to solve uh coding models using excessive
2:42:44
try except blocks? Is that a question? Yes.
2:42:51
Can you repeat the question? So what? Sorry. Yeah. Yeah. So so so um like a lot like
2:42:57
one of the sort of the most annoying code smells in AI models is their excessive use of try except blocks. Yes.
2:43:02
How would you go about fixing this problem? So it's they're not as defensive. That is actually a very good point. Um
2:43:10
for example, I was like for example when you do actually I just experienced this a few days ago when I was trying to use like AI models to like you know generate
2:43:16
code for like you know training. Um it keeps doing try and accepts for like especially torch functions you know try
2:43:22
to import this torch like for example torch dynamo reset to like reset the compiler cache. It tries to like always
2:43:28
do this try except try except try except um you could I guess force the system prompt to say you know not to do try
2:43:35
except um it doesn't really work. Still does it still does it? Yeah it doesn't really work. Um I would
2:43:41
say that is actually a very hard problem. I'm actually not sure why it does lots
2:43:47
of tri accepts. My view is because it's very the language models are very
2:43:52
pedantic um because it needs to consider all versions of packages. Um for
2:43:58
example, PyTorch you know for example some functions are not existent when it's PyTorch 2.0000 0 0 right so like it
2:44:04
doesn't actually have a specific function or you know pi pytor 210 right so it has a specific function and so
2:44:09
that's that's what I see when it you know language models generate this um it's because it considers too many
2:44:15
factors right it it understands okay what happens if pytorch the version was 2.8 you know that's why I have to do a
2:44:21
try and accept or or um I I guess you could specify you know all the
2:44:26
dependencies like you know you are using pytorch version 2.8 eight. Um, that could help. Um, I I to be honest, I
2:44:33
don't think so. I think I tried it before. It doesn't really help. Um, but I think that's a great question. To be
2:44:38
honest, I'm actually not sure. Um, that's a very good question. Um, yeah. Yeah. I don't know the answer for the
2:44:44
specific case, but I think in the context, um, one of the things that might work would be using like rubrics
2:44:50
and rewards. So the idea here I mean there are many papers doing this in various contexts but the idea is instead
2:44:56
of having a single outcome like you know correct incorrect uh for my reward um I
2:45:02
have a way of assigning like partial credit and the partial credit is based on like a series of like heristics. So
2:45:09
like in the context of say math you say okay I grade my proof with like a bunch
2:45:15
of things and if the model for example makes like a dumb mistake arithmetic
2:45:20
error it gets like a clear deduction from a rubric and so you could imagine a
2:45:25
rubric for a certain set of tasks would say okay like even if you get the answer right you get some part reward but you
2:45:32
get deductions because you used you know unnecessary triet and the way you would
2:45:38
have to then do that is again LLN doing the grading. Um so as long as the LN is good at identifying you know whe
2:45:50
I suspect that yeah I was going to say like yeah that's actually a good idea like you know for
2:45:55
the reinforcement learning side like you if you keep seeing try and accepts maybe like add some sort of counter you know
2:46:01
if you see a try and accept too many times divided by you know how many other things you see then penalize that. Um, I
2:46:07
guess that could be, but but is Mark's question like about reinforcement learning or is it like a generic like a
2:46:13
language model type thing or is it like for reinforcement learning? Um, I mean I I think this seems to be like a
2:46:19
unique property of like it's it's more egregious for reinforcement learning for for RL models.
2:46:25
My suspicion is it's like this is like a hypothesis. It's just that like if they're using a rubric then correct code
2:46:32
is really important and that like trumps a lot of stuff and then if you try accept everything your code's almost
2:46:38
always going to be correct or have some sort of reasonable behavior like you know just reasonable exit codes.
2:46:43
Anyways, yeah, maybe uh maybe maybe I'll just try to ask friends at Antropic if
2:46:50
they're actually working on this. Uh I was going to say that sounds like a that sounds more like a like a actual
2:46:55
case of reward hacking I guess. like you know essentially you just add tries excepts everywhere to solve that one
2:47:01
correctness um you know has to be correct or okay maybe that's interesting
2:47:06
um but you're right like I do always see generated code with try except try except try except um but to be honest I
2:47:11
think like Python's pretty cool like you know like py I think python 210 or two
2:47:16
oh I think it's 210 or 211 try except anyways doesn't take any time um there is no overhead with try except unless if
2:47:23
you enter the accept region then it's actually slower Um, so that's the only problem. Um, yeah,
2:47:29
but I think like try what's wrong with trieps? It's pretty cool. [laughter] All right. Well, okay. Oh, you guys keep
2:47:35
going. Um, but yeah, like uh so the last I think this Okay, there's one there's two
2:47:40
more, but no cheating um essentially if we you know if if the model cheats um you know if it uses numpy or pytorch and
2:47:47
stuff like this um then you know again we will like penalize this. And another one is if we find that the model doesn't
2:47:53
actually follow the format um we will actually for you know we will penalize this quite heavily um and you know this
2:48:00
you know minus one um and if it uses Python functions we will do minus 20 um
2:48:06
this is not allowed right so we will never allow this to happen um and so we give like you can also do minus 200 um
2:48:12
but you know try not to make too large numbers because you know this will skew the training process too much um yeah
2:48:20
and you know we just you know we place everything into like one large function to like check if the strategy has
2:48:25
succeeded. So this this actually executes the strategy on um open m um
2:48:31
and you know this also has a timeout like for example we used execute with time limit so we don't actually want the
2:48:36
model to like you know you know execute the strategy forever um and if it does succeed we massively reward the model um
2:48:43
and if it failed um we still reward it a little bit because it kind of worked um and for timeouts we will you know u
2:48:51
reduce the reward for these um timeouts and yeah so like essentially take the
2:48:56
one prompt that we had and just multiply by 1,00 um so it's the same prompt over and over again you can custo so if
2:49:02
you're a large lab or like you know you don't just want to do one example um then all of these it won't be the same
2:49:08
example right it will be like each of them will be different um but in this small example we can just use 1000 of
2:49:14
the same prompt um and then we'll be using TRL so like essentially unsoft you
2:49:20
know works on top of TRL to make it much more efficient and use less memory we also use trackio to like track um inside
2:49:27
of the collab notebook um the logging um and so once you actually get to the
2:49:33
training stats um you you will actually see the strategy that is generated um and you know for example this is at the
2:49:39
beginning and over time um you will see you know lots of strategies generated um
2:49:44
and the goal is you you have this large table of numbers and the goal is you want the reward to continuously going up
2:49:52
um and so the goal is you want to just look at this column and see if the reward is going up or not. Um and you
2:49:57
can look at the other columns like for example different reward functions have different um you know rewards definitely
2:50:04
look at look at that. Um for example um you know function works is very important. So like you know definitely
2:50:09
check if this column is going up. Um and you know this table is very important. So definitely keep a track of this. Um
2:50:16
yeah and yeah like I think that's and you can save. So for example at the very
2:50:21
beginning um once you actually finish fine-tuning you will get lots of strategies generated you can like you know do inference for example um how
2:50:28
like after the fine tune after the reinforcement learning run what does it generate and then it kind of like generates a better strategy now um and
2:50:35
then finally because this is Qura um you you need to use save pre-trained merged
2:50:42
or use the Laura adapters directly right you cannot you should not upcast the 4-bit model to 16 bit and then merge the
2:50:49
Laura weights um this will actually damage the model entirely. Um we actually did many experiments and I think this reduces accuracy by like 30%
2:50:56
or something. Um and so definitely don't do this. Um instead you know this function merges takes the 16- bit weight
2:51:02
and merges it directly inside. Um and so yeah you can upload to hugging face and stuff like this. Um yeah and also like
2:51:10
you as a final note we also have a guide which you can go to um you know it's if you want to read more about Jupiter's
2:51:16
franch uh reinforcement learning um we have more examples um and there's also a reinforcement learning guide for people
2:51:22
who like want to learn more about reinforcement learning um you know for example Pac-Man example and stuff like this um yeah but that's that's about it
2:51:29
from my side again um yeah thanks a if I may ask one like final question
2:51:36
Um so you know about this time last year DC1 came out and broke the internet and
2:51:43
[clears throat] kind of I think made RL go mainstream. So you know we've now had a lot of practice doing RL. Um I'm
2:51:50
curious where you think this year will go. So where do you think like what role
2:51:55
does play in in in the field and are you expecting any other like bombshells?
2:52:03
That is a very good question. Um, I'm assuming lots of labs are going to be releasing lots of cool models this year.
2:52:09
Um, obviously you know Deep Seek's new model is very interesting. So I'm definitely I think it's like next month
2:52:14
or something according to rumors um during Luna New Year. So I'm waiting for that. Um supposedly so the view is
2:52:20
reinforce so the main view of large labs is reinforcement learning should continue to scale. Um I'll be interested
2:52:26
to know um you know will reward hacking cause a more bigger headache um than
2:52:31
actually if you scale reinforcement learning. So actually my take is reward hacking will actually take over um and
2:52:36
because you like you could you could hire like you know data labelers like for example you could go to mour you
2:52:42
could go to like you know scale to like data labeling um but you know the view of large labs is you want to automate this away um so my take is like you know
2:52:49
you don't actually want data labeling um RL essentially can automate reinforcement learning right so RL can
2:52:55
automate RL um and so I think maybe this year we will see um you know less usage
2:53:01
of data labelers My view is you know we should like we should have we should have models create
2:53:08
environments as well. Um and you know let humans also like test does the environment create you know is the
2:53:14
created environment actually good or bad. Um and these automatically generated environments can probably be
2:53:20
used for RL as well. Um and so that's kind of my my view. Um as we're going to be maybe this year we're going to be
2:53:25
seeing automated RL environment generation. Um and my assumption is open
2:53:31
end will also have something like this inside. Um and so like that's my kind of view. Um yeah.
2:53:37
Cool. Thanks. All right. Uh okay. So we have our last
2:53:43
speaker of the day uh which is Zach Wentz. Zach, please take it from here. Let me share your screen.
2:53:49
Uh Zach, do you want anyone else on on the stage with you
2:53:54
and then I will go away.
2:54:04
You're both muted. Yeah. Can you hear me now?
2:54:10
Yes. Okay, perfect. Um, well, we're we're running a little bit behind, so we'll
2:54:15
jump right into it. But David and I work closely uh here at Meta um on the
2:54:21
PyTorch team uh on OpenM and other things. Um, so this talk is entitled CPU
2:54:28
mode. We've got slow Goku there representing the CPU. Um, and we're going to talk about in
2:54:35
this talk a little bit about where we think things are going as well as um,
2:54:41
uh, the the the side of the fleet that when we're talking about reinforcement learning on an environment that you're
2:54:48
not considering. Um, so the GPU fleet gets all the attention.
2:54:54
Um, but we have to scale up our sandboxes alongside it. And our sandboxes are lower than our GPUs. And
2:55:02
so we've got to scale it up quite a bit. Yes, it's cheaper. Um, but it is compute you have to think about. And you'd be
2:55:08
surprised how frequently people do forget about it and how surprised they are at the size of compute we need.
2:55:15
We're going to talk through some of the scaling problems in this session. Um, so throughout this whole thing, we're
2:55:21
trying to maximize throughput. This sandbox demand formula here that we have, I think is a bit outdated when you
2:55:28
move into the world that we're moving into with really long trajectories that you're trying to train on. Um, as this
2:55:36
increases and as they continue to be stateful, we start to lose the tricks that we had before. Um, and we'll talk
2:55:42
about some of those. So, just to give you a background, like
2:55:47
what do we do? Been doing this for about a year and a half now. uh started out as
2:55:53
like code interpretation simple tool for meta AI and it's grown increasingly
2:55:58
since then um to where today you know we are supporting at any given moment uh
2:56:05
the platform is supporting 10 million or more concurrent sandboxes at a steady state and we have an order of magnitude
2:56:12
bigger spikes um our pre-warms 150 milliseconds pretty much standard um
2:56:19
there's folks like Daytona for example that are doing better than this. But at a certain point, you're you're running
2:56:25
into the laws of fix physics with uh network requests anyway. So it doesn't
2:56:30
matter much at a certain point, especially on a long trajectory where you amvertise that cost over a long
2:56:36
time. Um for cold starts, we'll talk about different regimes, but for cold starts, we got those down as well. We're
2:56:43
using Docker layer caching and we're using um registry peering as well. So when I say registry peering, we're
2:56:50
peering from the hosts that are executing these sandboxes um the different layers rather than going back
2:56:56
to the registry. That's the better way to think about this. Um yeah, we started with batch snippet execution and uh
2:57:04
we've now moved through out the entirety of the pipeline pre-training, post- training of valves, really long
2:57:09
trajectories. We're moving into inference. Um we'll talk a little bit later about the year of the sandbox.
2:57:18
uh scaling regimes. So there's two main scaling regimes to think about um that
2:57:24
you need to scale differently. So there's the hot path which is these are our battle tested environments. You can
2:57:30
think of image and environment as sort of interchangeable here, but these are our battle tested environments. These
2:57:36
are ones that we're using for a well-honed use case. Um, and uh, you
2:57:42
know, we've gone through all the stability and reliability issues that we've had before. We're going to pre-warm a pool with these. We know
2:57:49
generally how much we're going to be scheduling for at any given moment. And we're going to cache a lot of those image layers just directly on the host
2:57:55
itself. And we're shooting for 200 milliseconds or less to get started. This is typically what you're going to
2:58:01
see in like when you when you finally got your model trained on some agentic workflow and now you want to deploy this
2:58:06
agent with an environment. um you're going to be looking at the hot path. The
2:58:11
long tail is where we talk about like getting a generalizable model. So you've got a bunch of different unique
2:58:17
environments. Um you have a very different scaling form here. So um you
2:58:25
can't pre-warm all these. It's just not possible. You know half a million environments. Uh that's not going to
2:58:32
work. So you need some other mechanism. This is actually where like open im starts uh back in the day which is we
2:58:39
need some base image with a lot of common dependencies. Everyone is authoring these environments in
2:58:45
different ways. They're you know you can even have the same dependency but uh if
2:58:50
you have like one you know one version number off or something like that it's going to have a different check sum and
2:58:56
now your cache is less effective. So getting as many commonbased layers as we can get in some con you know so our
2:59:03
content addressable layer caching works um is something that open M can start to
2:59:09
you know help you benefit from when you're running Docker environments. There's also micro ENV uh microVM
2:59:17
environments but those are a little bit different. We'll talk about that. Um but
2:59:23
here you know you're you're generally shooting for uh less than 10 seconds of startup time. The the lower you get the
2:59:30
better but again long trajectories. Um the other thing to talk about in terms
2:59:36
of regime where you can get some performance improvements is how much do you trust the workload in this environment. So like we started all this
2:59:42
in you know the sandbox from I want to lock down generated code that I don't trust. Um but we are starting to see
2:59:51
companies training on I mean we're training on it too internal services right um we trust those services we we
2:59:58
wrote those services and so there's some like semirusted place here where we
3:00:05
trust the services code you can't do too much as an agent in this environment and so we will relax some of the protections
3:00:12
we have here and therefore we'll be faster to execute and run and then there's this trusted territory that, you
3:00:19
know, starting to see pop up. Um, we don't necessarily have this use case
3:00:24
yet, but it's something to start poking on. This is where you could just run a container as is without like a VM
3:00:31
sandbox around it. And David, please feel free to chime in at any point to add color, but I think
3:00:37
your slides are coming up. Um, this one was kind of a big one for us, middle of the year. I alluded to it in uh chat. um
3:00:46
the the registry bottleneck um so once you get to where like the infrastructure
3:00:53
will scale it starts to become just a problem of moving bits around um and that's where you get into the the work
3:01:00
we've done on like the the high cardality side if I go back to this to the longtail just getting these bits
3:01:07
onto a host starts to become a real problem and so you may have like dockeraware caching um you know where
3:01:15
caching the layers. Uh but if you're not peering those, you're just going to be slamming your registry with a ton of
3:01:22
requests anyway. Uh even though it's a small just what we need that isn't cached locally, it's still going to
3:01:28
overload your registry after a certain scale. Um, at one point, I believe, just as
3:01:35
like a fun fact, at one point without the benefit of peering, our cash was
3:01:41
like a fourth of Facebook's entire cache just for sandboxes. We quickly had to get out of that, but you know, for a
3:01:47
brief period, that's how big it was. Um,
3:01:52
VM snapshots. So, this is something else. One of the things that we're not doing but that you can do and a lot of
3:01:59
sandbox platforms do is to make your your uh cold start even
3:02:06
faster, right? You don't have to boot the Docker image itself. You can upon
3:02:11
ingest of a new environment into your registry. Go ahead and spin it up and
3:02:17
then right at the point of being ready to execute in, take a snapshot of your microVM and store that. And actually the
3:02:24
layers of that are are easier to cache um than say Docker layers. Uh you don't
3:02:30
have to you don't have to do all the work to make them common necessarily. Um but then when you spin it back up it's
3:02:35
like 4 to 10 milliseconds to be ready to execute. And so then network becomes your bottleneck. Lots of folks do this.
3:02:42
Um, lambda snap start is a great example of that where you know your lambdas run
3:02:47
fast but again it stops helping and that's the direction we're going generally is long trajectories
3:02:54
networkbound. So this is like the two ways to reason
3:03:00
about it. It it goes back to those um uh trusted regimes or not. Um, this starts
3:03:06
to lead into something that I want to talk about today that I'm like most excited about this talk uh to to talk
3:03:12
with you all about which is where we see this going. Um, which is a combination
3:03:19
of these two things. So why do we see it going in this
3:03:24
direction? I think everyone is doing claude code dangerously skip permissions. You get tired of accepting
3:03:31
permissions. It was trending on Twitter just this morning. people complaining about the frequent permission prompts.
3:03:37
Even when they say, you know, they didn't yellow it, but they said bypass uh requests, they're still approving a
3:03:44
lot of stuff. Where we're going to see this happen more often, though, is not from like user fatigue, but from sub
3:03:51
agents. Sub agents starting to take over. When you Ralph wiggum your cloud code, you're not going to want to inject
3:03:57
the human back into the loop to go start approving permissions. You're just going to say dangerously skip the whole way.
3:04:03
And so you're going to need a safe bydefault sandbox to operate in um as
3:04:08
opposed to flagging it to say that you can be dangerous. Um that's where we're
3:04:14
going and the reasons why David can speak to more so. But how we see this
3:04:19
starting to combine and this is where I want David to jump in as well is I've
3:04:25
been calling [snorts] it an agentic kernel. I think probably from this talk the better way to look at it is an agentic microVM. But like imagine a
3:04:33
world where you know you have your Docker image, you have your microVM that starts fast, it's safe. Uh your Docker
3:04:40
image though is easy to author. It's quite observable. Um and you can put the dependencies you want in it with ease.
3:04:47
Imagine a world where these two combine. I think we're quickly headed in this direction. Um I think we'll see multiple
3:04:53
projects like this ship in 2026. Um, I think you might see me and David's
3:04:59
face again talking about this in 2026. I think it's quite exciting. I especially
3:05:04
think the architecture bit of this is very exciting which is thinking about an execution environment and an agentic
3:05:11
environment as one and the same but with separations much like we have the Linux
3:05:16
kernel today where you have kernel space and user space. Using that same metaphor but applied to this problem, I think
3:05:23
brings some very interesting things that we can start to do and that we need to do in order to keep going in the
3:05:29
direction we're going with all these agents supporting us.
3:05:35
Yeah. And then to add color here, I think that what we're seeing is um you know, as things become more clear, this
3:05:42
is an entirely new world, right? And so as things become more clear, I think um it becomes obvious that some of the
3:05:48
infrastructure that we had was built for similar but fundamentally different use
3:05:54
cases and different requirements. Um and so this is what we're we're going again and we're going towards and uh I think
3:06:01
that this intersection of like low trust but high performance is something that
3:06:07
was not exactly quite touched upon by the existing systems. And so you're going to see a lot of activity here and
3:06:14
from us as well. Um, this is pretty cool. Like I guess I
3:06:19
had two unrelated questions. Uh, Zach, you mentioned something around like private Docker registries being rate
3:06:26
limited. Is this like a fundamental technological constraint or is just like a policy choice on their end? Like
3:06:32
because that just feels like why have it at all for private? Doesn't feel like it matters. Sorry, I I didn't mean to say
3:06:38
necessarily rate limited so much as like it's limited by the the pipe that you have. Um, so like even internally, we
3:06:46
had this conversation about the number of images in our registry. And at a certain point, we realized we had more
3:06:52
images in our registry than DockerHub has accured over 14 years. And we're
3:06:57
pulling them at a pretty active clip and concurrently. And so we were just bottlenecking like we could not no
3:07:03
matter how much bandwidth we threw at it, how many hosts get enough through the pipe to make this work at the scale
3:07:09
we wanted to. We had to move to something like peering. Ah, I see. Interesting. Okay. Or or or I
3:07:16
guess like the simpler solution would be like just have a lot less images internally too, right? But that's probably it's much more politically
3:07:22
difficult and so then the peering ends up Okay. Yeah. Well, and that that is where a long trajectory for example might help,
3:07:29
right? because we can we can keep the image more common and we can accept some
3:07:35
build step configuration that an environment author might do to where they say once we've got this image up
3:07:41
and running you need to perform these few commands. Yes, it's going to take a bit of time, but amortized over the time
3:07:46
that we're going to be interacting with this environment, that's fine. And then your images become much more common. Um,
3:07:52
but your configuration starts to change a little bit. And that's that's you know, that's nothing.
3:07:58
And and sort of the second question is like I think because uh because this is very timely. I think like yesterday like
3:08:03
Entropic released like their uh like their their coworker product and I think people sort of reverse engineer it and
3:08:09
figure it out that it's like some using some container service on Mac. Uh as you look at it, do you go like oh yeah like
3:08:15
makes sense or do you just feel like containers are fundamentally the wrong level of abstraction here or is it like just good enough and then there's like a
3:08:22
few patches we need to do here and there? Yeah, I I think I mean I'd be
3:08:28
curious to hear what David says, but I'll just jump in with with my own answer to start.
3:08:34
I think that containers are the the the choice right now because they're easy to
3:08:42
author and easy to set up and researchers are driving a lot of this and that makes its way into a product
3:08:48
and if we can make it such that something like a microVM is as in distribution such that you can vibe code
3:08:54
it right um as Docker is today um to where a researcher can start from that
3:09:01
point and where safety you know you start from safety and in fact has some agentic primitives to where they would
3:09:06
want to do that in the first place to help sweeten the deal. Um I think that tips things in the favor of microVMs
3:09:13
which are going to be much faster uh and safer and and specifically if people are
3:09:18
interested in the space like what which projects would you recommend they follow like is like firecracker like for example like one good one or are there
3:09:24
others that you're you're interested in? I would look more at the um sandbox
3:09:29
providers especially the cutting edge ones. So I would look at Daytona. they they have got their their thumb like I
3:09:37
don't want to say thumb on the scale that's the wrong way to think of but they know where this space is going and they've got a good sense the other one
3:09:42
is spritesdev um they rock they've built like currently I would say that is like
3:09:48
state-of-the-art that you can buy and at least that you should follow uh for a sandboxing platform like it it's awesome
3:09:56
I think it nails it I think I'll invite them for a talk then I really don't have to rush you guys by the way I just looked at time I have a
3:10:02
hard stop. So I think we have about maybe seven minutes for you all to finish up if that's okay and then I just want to give everyone the chance to do
3:10:09
some closing thoughts as well at the end. David, did you want to answer that question uh as well like provide your or
3:10:15
did you I think you are the better person here in a way. So like I you know like I think the for everybody else I think our
3:10:22
relationship is more I'm more on the kind of machine learning side or else side and so I kind of pester back with
3:10:27
requirements and then that figures out how to how to get it done. Um but I
3:10:33
think that like the this actually works because I think that you know ML
3:10:38
researchers you know tend to kind of do whatever to get unblocked and so we get to like these uh you know big
3:10:44
repositories where like 90% of the code is not necessary uh it's just there because we didn't have proper
3:10:49
abstractions the proper infrastructure um and so I think that one way of measuring you know whether we're going
3:10:56
to be successful here is really to see this like the size of ML project's code kind of go now.
3:11:02
Yeah. And become like a oneliner. We're like, okay, I obviously the sandbox is doing what it's supposed to be doing and
3:11:08
therefore I can just trust it. That's kind of where we want to be. Yeah. Yeah. We honestly that I think to
3:11:14
to tlddr that like we're we're talking a lot about sandboxes right now. We probably shouldn't be talking this much
3:11:19
about sandboxes in in the right future. It just is a solved problem. Like just like we don't talk a lot about Linux
3:11:25
unless you know you're on my former team working. Exactly. Right. Think about go into like reading a file on disk, right? We just
3:11:33
Yeah, read the file. [laughter] I'm sure there's a way it's there.
3:11:38
All right, I'll let you guys like then uh finish things up. I'll go away. This is the last slide. We end with
3:11:44
questions. So, uh like I said, we can do your timing mark. [laughter] Um yeah, like I I think like I want to
3:11:52
talk more about sandboxing then if [laughter] that's okay. Yeah, let me let me also bring everyone on stage like I
3:11:58
think in case people want to also jump in with other thoughts here. I'll remove this. Should I kill screen share and we just
3:12:05
vibe it out? Yeah. Uh yeah. So for example, like one product that's been like a favorite of
3:12:11
people in GPU mode has been modal. It's also sort of like sandboxed execution
3:12:16
of like simple code. And I think we've really liked it because you can do like these kernel microbenchmarks like very
3:12:22
easily. It's quite cheap because it's like Q- based. Uh it has just been like overall like a wonderful experience. I'm
3:12:28
not sure what their secret sauce actually is to make sure things like run really fast. Uh but you know like I'm
3:12:35
sort of curious like like I guess Zach you surprised me to say like oh it seems like this is actually a vibrant space and many people have products and
3:12:41
opinions and I'm sure cold starts are important for them and Yeah. So I'm wondering if you could just maybe
3:12:46
elaborate a bit more on projects you'd like to see here like Yeah. Yeah. Yeah. I mean so we've even used
3:12:54
modal uh you know we talked a little bit about that code world model paper uh earlier Lewis did um a lot of that was
3:13:02
trained in modal a lot of it was trained in the platform that I support at meta
3:13:07
um uh I think modal does some really good stuff as it relates to like developer experience and working around
3:13:14
some compute constraints that we haven't had to do um and in fact we need to do now like we probably need to catch up on
3:13:21
DevX and internally. Uh but um yeah, I I I think modal they don't have secret
3:13:28
sauce. I don't think they're all that fast if if I'm honest. Uh and I think they've got like they found their niche.
3:13:35
I think there's faster ones that are that are more on where we need to be. Sprites being the top one in my list. Um
3:13:42
but yeah, sprites. Okay, let me take a look.
3:13:50
I see. And I guess like eventually like your hope is that this is maybe question relates to Ben which is that like there
3:13:55
would be all these like sort of I forget what was the terminology you used like it wasn't back end it was like something
3:14:01
basically you had the UV thing the docker thing I think you had terminology providers the provider yeah so you would imagine
3:14:07
that eventually these like sandbox environments would be like providers and they'd be sort of like very additive to the work you're doing
3:14:13
yeah I mean so the interactions that we're having with researchers are kind yeah as I said before at the opposite
3:14:19
end of the scale to Zach and they're like often like researchers in in university labs, they've got a series of
3:14:25
very interesting environments that have come out of a paper and they're probably quite hacky and they're not necessarily
3:14:32
something you can combine. And so kind of open brings that layer to that environment that you can then make it
3:14:38
interoperable and you can take something like a provider and it could start off as like just local docker provider. I
3:14:45
want to evaluate this environment and then you can plug in like slurm and kubernetes provider with the same
3:14:51
abstraction and just kind of deal with that.
3:14:57
Um, and I guess like then question maybe more to Louisis, Daniel, and D like which is like like it seems like I mean
3:15:04
y'all are a bit like from different like slightly different worlds like which is like there's sort of the like what I would call infra infra and then there's
3:15:10
like the MLSIS infra with kernels and then there's the science. Um like what kinds of tools like do you think would
3:15:16
make your life easier? Like where do you think like this whole ecosystem of RL is really underserved? Do you think like
3:15:21
fundamentally these are like the important pieces and people should keep going? Yeah. Any sort of open challenges
3:15:26
you have to people in the GPU mode server? Um I I think um the biggest pain point
3:15:33
I've experienced is that um the the field of like say RL algorithms moves
3:15:39
relatively fast. So there's like a lot of like I mean there's all the permutations of GPO now that you can
3:15:45
imagine and it's never entirely clear like if they are meaningful improvements
3:15:51
or if it's just you know like noise in the eval or something like the these things are like quite hard to validate
3:15:58
and test. And um part of the the reason why it's tricky is that some of these things work at one scale but don't work
3:16:04
at another. And my experience so far with like the open source ecosystem is like with the RL tooling we we don't
3:16:12
really have anything I think that is like probably what like is internal at like OpenAI and and other labs which
3:16:18
like lets people do this like fast experimentation at different scales without having to think about okay what
3:16:24
do I trade off if I want to go to like 100B versus 1B. um most of the open source ecosystem is like fragmented in
3:16:30
this thing is like they target one of those axes of scale and then they make that work but then you make trade-offs.
3:16:36
So I think a bit of an open problem is like better kind of tooling overall um
3:16:41
to make that that thing easier. And the other thing that's still a bit missing is like the observability of like RL
3:16:48
runs like we have metrics like reward and stuff but like you know having ways to kind of easily collect all the
3:16:54
rollouts and like inspect them and that kind of stuff like this is still a bit early. So that's my my two cents.
3:17:01
[clears throat] I think in my case the the feature that
3:17:07
also will intersect the most here with this audience is also parallelisms
3:17:13
um that has been um um I think it's just they're not into at least I've never
3:17:19
found them intuitive and especially like once you start adding four five dimensions of parallelisms it's very
3:17:24
hard for me to keep track of what's even happening like what what work is being
3:17:29
done by any GPU at any given point in time um and certainly you can treat
3:17:35
these as hyperparameters, you know, sweep them, figure out where you have the most throughput and what
3:17:40
not. Um, but at the same time, you know, on boarding new models is difficult and
3:17:45
um starting to do a bunch of these wizardries with interle forwards, backwards, and all the stuff that Deep
3:17:51
Seek has been doing. Um, it's just a major kind of slog on kind of my understanding of the flow and my
3:17:57
productivity in general in the new models. That's a really I just want to iterate a
3:18:03
bit on is like um the the ecosystem has migrated to mixture of expert models
3:18:09
like pretty uniformly um but most of the
3:18:14
tooling for training is not really well suited for mixture of experts. So you kind of have like this hard choice. It's
3:18:20
like I use Megatron from Nvidia um which you know is very powerful efficient but
3:18:26
has its trade-off that it's like not perhaps the most userfriendly. and then you want to say go to torch titan or
3:18:32
torch forge but then you you sacrifice like genuine performance and this is like a bottleneck and I think we don't
3:18:38
yet have in the open source ecosystem something that is like flexible but really highly performant for for mixture
3:18:44
of expert models and maybe that will come this year I guess. All right then I'll close things maybe
3:18:49
with one last question. I think this is uh directed to Daniel but if anyone else has a hot take I'd love to hear it. So
3:18:55
then I think when you first started Onslaught like the it was kind of quite clear like basically there are important
3:19:00
LLM kernels and you had like written them all out like in and Trident. Uh these days it seems like you're relying
3:19:06
a lot more on other libraries, right? Like you're like on VLM and like these like training libraries. Uh do you worry
3:19:13
then about numeric? Do you feel like the sort of concerns around this are a bit exaggerated? Are you planning on just
3:19:18
rewriting everything with your own kernels? How do you sort of think through this like in a bit of a longer time frame? Yeah, that's a great
3:19:25
question. Yes, we did start off with Triton kernels and like it's great. Um,
3:19:30
but I think to be honest, I think I would divert the workload to the PyTorch
3:19:37
smart engineers, right? So like my view is torch compile will actually
3:19:42
will take over. Um, and so like I think so there are two types of algorithms which still cannot be like so there are
3:19:47
two types of algorithms. There's like kernel fusion type algorithms and then there's actually mathematical type algorithms. So Torch compile still does
3:19:54
not do very well with mathematical type algorithms, right? So like data move essentially for data movement changing.
3:19:59
So changing data movement patterns for example like flash attention three or flash attention. It's more like data
3:20:04
movement and it's probably hard for torch compile to like do something like that. Um for example like async grading
3:20:10
checkpointing or like you know chunked losses and stuff like that. Um however my view is like maybe over time there
3:20:17
are like specific methodologies which are like repeatable for like for example like chunk losses it might actually be
3:20:24
you know inside of torch compile it might be like some sort of setting right um everything can be chunked everything can be like async to like RAM and GPU
3:20:31
memory and stuff like something like that. Um so I think like my take I'm not I'm not sure if that's probably correct
3:20:36
but like my view is like torch compile everyone will just use torch compile um and just enable it by default. Um that's
3:20:43
kind of my view. Um but I think the only problem is that torch compile does have a warm up problem. Um it is very slow at
3:20:49
the very beginning. Um but once you like you know bypass the warm up problem um I think like you know torch torch compile
3:20:55
has like cache. Um so like you can use inductor cache and stuff like that. So it does help. Um yeah but I'm not sure
3:21:01
if that kind of answers your question. Um it does and I think we unfortunately need to end things here. But I think what I liked about your point is you're
3:21:07
sort of suggesting that like maybe the kernels isn't really the most productive thing. Like for instance, I don't know
3:21:12
if there's like tens of thousands developers working on efficient sandboxes, but you know, maybe there
3:21:17
should be like a bit more than what we have today. Um, so I think uh I will certainly be reaching out more with you
3:21:24
guys like privately to figure out if we can do some more interesting working groups on the server and if there's more open source projects that would help
3:21:31
researchers such as yourself. But honestly guys, like thank you so much. I'm very honored you made uh made the
3:21:36
time for this. This was three and a half hours long. Uh, I think they they for Yeah, I mean I was expecting only two
3:21:41
hours, but yeah, this was this was very nice. So, thank you and I appreciate this and I'll see you soon. Thank you, Mike. Thanks for having us.
3:21:47
Thank you. Thanks.


Resources & Materials
Official hackathon resources for all finalists. Use these to build, train, and iterate.
1. OpenEnv Core
An interface library for RL post-training with environments.
GitHub: https://github.com/meta-pytorch/OpenEnv
Docs: https://meta-pytorch.org/OpenEnv/
2. HuggingFace OpenEnv Hub
Environments: https://huggingface.co/openenv
Spaces: https://huggingface.co/openenv/spaces
3. Tutorials
Build, run, and train RL environments and training pipelines.
All Tutorials: https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial
Training Examples: https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial/examples
Environment Examples: https://github.com/meta-pytorch/OpenEnv/tree/main/envs
4. YouTube: Building RL Environments
https://www.youtube.com/watch?v=0airz7BhBiA
https://www.youtube.com/watch?v=ap4q4sAK4OY
https://www.youtube.com/watch?v=Jew4lhAiqnw
https://www.youtube.com/live/kkCNMz0Ptd8?si=JJ7og8x5qc7_Gi0e 
https://www.youtube.com/watch?v=Jew4lhAiqnw 
5. Research Papers: Reward Engineering
https://arxiv.org/abs/2408.10215
https://arxiv.org/abs/2601.19100
Tip: Chat with these papers using LLMs for quick insights on engineering rewards.

1) What is reinforcement learning in the context of LLMs?
Reinforcement learning for LLMs is a loop where the model generates an answer, code snippet, plan, or action sequence; that output is evaluated by a verifier or environment; and the resulting reward is used to update the model so higher-reward behaviors become more likely over time. In practice, this is often used after pretraining and supervised fine-tuning to sharpen behaviors like reasoning, code generation, or tool use. The session framed this intuition as turning repeated trial-and-error into weight updates instead of stuffing more and more examples into the prompt.
A good mental model is: supervised fine-tuning tells the model “copy this good target,” while RL tells it “try many possibilities and move probability mass toward the ones that score better.” PPO is one classic algorithm for this style of training, and GRPO is a later variant used heavily in modern LLM work because it can be more memory-efficient for certain setups. (arXiv)
For deeper reading:
TRL docs for RL trainers and workflows. (Hugging Face)
PPO paper. (arXiv)
DeepSeekMath for GRPO. (arXiv)
2) Why do rewards matter so much?
Rewards are the only signal telling the model what “better” means. If your reward is well aligned with the real task, RL can push the model toward genuinely useful behavior. If your reward is incomplete or easy to game, the model will optimize the wrong thing very effectively. The session emphasized that RL gives you what you asked for, not necessarily what you meant.
For example, if you reward generated code only for passing a shallow regex or a weak unit test, the model may learn to exploit those checks instead of solving the underlying problem. This is why reward design is not a detail; it is the task specification. DeepMind’s discussion of “specification gaming” makes the same point in broader RL terms: weakly specified rewards create loopholes that search will discover. (Google DeepMind)
Useful reading:
DeepMind on specification gaming. (Google DeepMind)
Lilian Weng on reward hacking. (Lil'Log)
3) What is rewards engineering?
Rewards engineering is the work of designing, combining, validating, and monitoring reward signals so that optimization pressure produces the behavior you actually want. In LLM RL, that usually means deciding:
what gets rewarded,
how much it gets rewarded,
when it gets rewarded,
what gets penalized,
and how you audit whether the reward is being gamed.
A practical reward function often has several components. For a code task, you might combine syntax validity, execution success, unit test pass rate, latency, memory use, formatting compliance, and safety checks. The session highlighted verifier-based reward design such as formatting checks, execution checks, regex checks, and environment-based evaluation instead of a learned reward model alone.
A useful principle is to reward outcomes first, then add process constraints only where needed. Over-shaping the reward can make training brittle or bias the model into narrow strategies, while under-shaping makes hacking easier. (Google DeepMind)
4) What is RLVR, and how is it different from using a reward model?
RLVR usually means reinforcement learning with verifiable rewards. Instead of asking a learned reward model to score outputs, you use a verifier, tester, or environment that can check correctness more directly. The session gave examples like formatting checks, execution checks, regex-based checks, and environment rollouts.
This is powerful when correctness is externally testable. Code can be compiled and unit-tested. Math can often be checked against a final answer or symbolic verifier. Games can expose reward from the environment. Browser tasks can be checked by page state or task completion. In such cases, verifier-driven rewards are often more trustworthy than a purely learned scalar reward model.
TRL documents this broader environment-based training pattern, and OpenEnv is meant to standardize how such environments are defined and used. (Hugging Face)
5) Why do RL environments matter for LLMs?
Static prompt-response datasets are useful, but they are limited. Real deployments require models to interact with systems: codebases, browsers, files, APIs, games, tools, and simulators. RL environments let the model act, observe consequences, and keep going across multiple steps, which is much closer to real agent behavior. The session described environments as the bridge from isolated prompt solving to real-world interaction.
They also enable dynamic difficulty and richer feedback. Instead of training forever on a fixed set of prompts, the environment can generate or surface tasks that are more appropriate for the current model, which makes curriculum learning and continual challenge easier. This matches the broader “RL with environments” direction discussed in recent OpenEnv and TRL material. (Hugging Face)
For examples:
BrowserGym for web-task environments. (GitHub)
OpenEnv course and TRL integration docs. (GitHub)
6) What is OpenEnv, and why would a hackathon team use it?
OpenEnv is an open-source framework for defining and interacting with RL environments for LLM and agent training. The session described it as a standardized interface around concepts like reset, step, state, observations, actions, and rewards, with deployment built around Hugging Face Spaces and containerized execution.
A hackathon team would use OpenEnv because it reduces environment plumbing. Instead of inventing a new interface for each task, you can standardize how the model talks to the environment and then connect that to a trainer like TRL. That means you spend more time on task design and rewards, and less on adapter glue. The session also highlighted openenv init for bootstrapping an environment skeleton quickly.
Good starting points:
OpenEnv repo. (GitHub)
OpenEnv course. (GitHub)
TRL’s OpenEnv integration guide. (Hugging Face)
7) How does OpenEnv work at a high level?
At a high level, an OpenEnv environment exposes a small set of standard operations:
reset the environment,
step the environment with an action,
return observations, rewards, and state.
The session described OpenEnv environments as FastAPI applications that can be run locally, deployed on Hugging Face Spaces, or pulled as containers. That gives teams several options: they can use the remote environment directly, install client code from the repo, or run the environment locally through the container image.
This design is useful because it treats environments as portable, versioned software artifacts rather than ad hoc scripts. Hugging Face’s own TRL docs describe OpenEnv similarly, including support for backend-server execution and standardized APIs. (Hugging Face)
8) Where do TRL and Unsloth fit in this stack?
TRL is the training library. It provides trainers and workflows for SFT, DPO, PPO, GRPO, reward modeling, and related post-training methods for transformer models. In a typical hackathon setup, TRL handles rollout collection, reward integration, optimization, logging, and trainer configuration. (Hugging Face)
Unsloth fits in as the acceleration and memory-efficiency layer for training and RL fine-tuning. The session described Unsloth as making RL training more efficient and inference faster, which matters because rollout generation often dominates runtime in RL loops. It also noted a practical QLoRA warning: don’t naively upcast a 4-bit model to 16-bit and then merge adapters, because that can damage model quality; use the proper merge path instead.
Relevant docs:
TRL docs and GRPO cookbook. (Hugging Face)
Unsloth repository/readme. (GitHub)
9) What is the difference between PPO and GRPO?
PPO is a classic policy optimization algorithm that stabilizes updates by constraining how much the policy changes between iterations. It is one of the most influential RL algorithms in modern deep learning. (arXiv)
GRPO is a later group-relative variant used in LLM training that compares sampled outputs within a group to estimate relative advantage, and it is often discussed as a more memory-efficient alternative to full PPO-style setups in some LLM post-training pipelines. The session summarized GRPO as a more efficient version of PPO and specifically noted removing the value model from the setup.
For deeper details:
PPO paper. (arXiv)
DeepSeekMath / GRPO references via TRL paper index and cookbook. (arXiv)
10) Why is RL often described as inefficient, yet still useful?
RL is often inefficient because the feedback is sparse and delayed. A long rollout may end in one scalar reward, and that weak signal has to train many decisions. The session used a simple example: if a code answer fails at one line but you assign the same negative reward to every token, you’re throwing away a lot of structure.
It is still useful because it can optimize behaviors where exact supervised targets are unavailable, too expensive, or too limiting. If you can verify success but cannot easily author perfect demonstrations for every scenario, RL can still improve the model by repeated interaction. This is why RL is especially attractive for code execution, tool use, games, browser tasks, and agent workflows.
A practical takeaway: use RL where verifiers exist and where exploration is worth the extra compute.
11) What is process supervision, and why is it important?
Process supervision means giving feedback on intermediate reasoning or intermediate steps, not only on the final outcome. The session contrasted this with assigning the same reward to every token in the answer, which can be very wasteful. Under process supervision, you try to identify which parts of a trace were good, irrelevant, or harmful.
This matters because not all failures are equal. Maybe the model chose the right algorithmic approach but made one implementation mistake. Final-outcome-only rewards blur that distinction. Step-aware rewards can improve sample efficiency and make debugging easier, though they also raise new risks if the step labels are noisy or exploitable.
The session also noted that process supervision is often approximated with humans or LLM-as-a-judge. That can help, but it creates another optimization target that itself may be gamed.
12) What is reward hacking?
Reward hacking is when the model finds a way to maximize reward without genuinely doing the intended task. In other words, the optimization succeeds, but the task specification failed. The session gave intuitive examples such as editing variables, bypassing intended checks, or exploiting quirks in the environment rather than solving the real problem.
This is the same phenomenon often called specification gaming. DeepMind describes it as agents exploiting flaws or ambiguities in the reward function, and Lilian Weng’s overview covers how common and fundamental this problem is in RL systems. (Google DeepMind)
A useful mindset is: reward hacking is not proof the model is “evil”; it is proof that optimization pressure found a loophole.
13) How can a hackathon team reduce reward hacking in practice?
Use strong verifiers. Prefer executable checks over stylistic heuristics. For code, run tests, time the solution, validate output shapes and edge cases, and isolate execution. For tool use, verify actual state transitions, not just verbal claims. The session repeatedly emphasized verifiers and environments over vague reward signals.
Monitor training actively. The session recommended sampling outputs periodically, looking for suspicious patterns, and terminating or rolling back runs when drift appears. It also suggested filtering bad responses and adding guardrails when patterns of exploitation are observed.
Use layered rewards. Combine success criteria with anti-cheat constraints. For example:
pass tests,
do not edit protected files,
do not bypass timers,
stay within time and memory budget,
preserve task-required formatting,
and log intermediate actions for audit.
This general strategy aligns with broader RL safety guidance on specification gaming. (Google DeepMind)
14) What is curriculum learning, and why does it help RL?
Curriculum learning means controlling the order or difficulty of training tasks so the model learns from easier tasks first and gradually moves to harder ones. The session directly recommended this for RL: if tasks are too hard at the start, the model may never produce a successful rollout, which means the reward signal is effectively zero and learning stalls.
This is especially important in LLM RL because many tasks are long-horizon and brittle. An easier initial distribution can bootstrap behavior, after which harder tasks become reachable. In the RL literature more broadly, curriculum learning is a standard way to improve exploration and sample efficiency in difficult environments. (arXiv)
Practical idea for hackathons:
start with short horizons,
fewer tools,
simpler state spaces,
stronger hints,
easier test cases,
then gradually remove scaffolding.
15) How do I know whether a task is suitable for RL?
A task is a good candidate for RL if:
you can verify success or partial progress,
exploration is meaningful,
multi-step interaction matters,
and you do not already have abundant high-quality demonstrations.
The session highlighted a key rule of thumb: the probability of a good answer must be greater than zero. If the task is so hard that the model never stumbles into any rewarding behavior, RL will waste compute. That means task selection, warm starts, formatting scaffolds, or light SFT can be essential.
Good hackathon candidates include:
code generation with executable tests,
browser navigation with page-state checks,
games with clear win conditions,
API/tool workflows with verifiable side effects.
16) Should we jump straight into RL, or do some SFT first?
Usually, do some SFT or at least a warm start first. The session’s guidance was that pretraining carries most of the capability burden, SFT helps shape the behavior, and RL refines it. It explicitly argued against relying on RL alone from scratch for most practical settings.
That matches modern post-training stacks: pretrain heavily, align or instruct-tune, then apply preference optimization and/or RL where it adds value. TRL’s supported workflows reflect exactly this broader stack. (Hugging Face)
A hackathon-friendly recipe is:
Start from a solid instruct model.
Add a tiny amount of task-format SFT if needed.
Build a strong verifier.
Use GRPO/PPO-style RL only after the model can at least occasionally succeed.
17) What should we actually monitor during RL training?
Monitor more than the headline reward. The session specifically called out tracking reward trends, component rewards, and whether important success columns are improving over time. It also recommended checking generated strategies and periodically sampling outputs during training rather than letting runs continue blindly.
Useful metrics include:
average reward,
verifier pass rate,
timeout rate,
format adherence,
rollout length,
diversity of successful solutions,
frequency of suspicious shortcuts,
and cost per useful trajectory.
If the average reward rises but the actual task quality drops or becomes brittle, that is often a reward-design problem rather than a model-capability problem.
18) What is a strong hackathon strategy for building an RL environment fast?
Pick a task with a crisp verifier. Build the smallest environment that exposes reset, step, observations, and reward. Use OpenEnv to standardize the interface and TRL to handle training. Use Unsloth if you need to fit training into tighter hardware budgets. (Hugging Face)
A practical sequence:
Define the task and what “success” means.
Write the verifier before writing the policy loop.
Create a few toy tasks the model can solve.
Add curriculum or easier variants first.
Run small-scale debugging before long training.
Sample outputs constantly for reward hacking.
Only then scale rollouts and environment diversity.
19) What are good starter resources for participants?
For TRL:
Main docs. (Hugging Face)
PPO trainer docs. (Hugging Face)
GRPO cookbook. (Hugging Face)
Paper index for GRPO/DeepSeekMath references. (Hugging Face)
For OpenEnv:
OpenEnv GitHub repo. (GitHub)
OpenEnv course. (GitHub)
TRL’s OpenEnv integration docs. (Hugging Face)
For environments and benchmarks:
BrowserGym. (GitHub)
For reward design and failure modes:
DeepMind on specification gaming. (Google DeepMind)
Lilian Weng on reward hacking. (Lil'Log)
For RL algorithms:
PPO paper. (arXiv)
DeepSeekMath / GRPO paper. (arXiv)
For Unsloth:
Unsloth repo/readme. (GitHub)
20) What is the one-sentence summary participants should remember?
If you can build a task where success is verifiable, difficulty is controllable, and loopholes are monitored, RL can turn an LLM from “good at answering” into “better at acting.”

21) What is RLVR?
RLVR stands for reinforcement learning with verifiable rewards. Instead of relying only on a learned reward model or human preference model, the training loop uses programmatic checks to determine whether an output is correct. Typical examples include exact-answer checks for math, unit tests for code, schema validation for structured output, or environment-based task completion checks. This makes RLVR especially attractive for domains where correctness can be verified automatically and consistently. (Label Studio)
22) What is RLVE?
RLVE is reinforcement learning with verifiable environments. The key idea is to train on environments that can procedurally generate tasks, expose adjustable difficulty, and provide algorithmically verifiable rewards. Recent work on adaptive verifiable environments argues that static prompt datasets often become either too easy or too hard during training, causing learning to stall, while adaptive environments keep the model near its capability frontier. (arXiv)
23) How is RLVE different from RLVR?
RLVR usually refers to verifiable rewards on a fixed or semi-fixed set of prompts or problems. RLVE goes a step further by making the task source itself dynamic: the environment can generate new problems, vary difficulty, and keep serving appropriately challenging tasks as the model improves. In practice, RLVE is often better for preventing saturation on static datasets and for building curriculum naturally into training. (arXiv)
24) Why are RL environments useful for LLM post-training?
They let the model interact, not just answer. In a real environment, the model can act, observe consequences, act again, and get reward from actual task outcomes. That makes environments a better fit for tool use, browsers, APIs, coding agents, games, and long-horizon tasks than plain prompt-response datasets. Hugging Face’s OpenEnv and TRL material reflects this shift toward environment-based agent training. (Hugging Face)
25) Where do TRL, GRPO, and Unsloth fit in?
TRL is the training framework that provides RL trainers and infrastructure for post-training transformer models, including GRPO. GRPO is the RL optimization method popularized in DeepSeekMath and now widely used in open LLM RL pipelines because it can be more memory-efficient than PPO-style setups in this context. Unsloth is typically used as the efficiency layer to make fine-tuning and RL training faster and more affordable on limited hardware. (Hugging Face)
26) Why do rewards matter so much?
Because the reward is the task definition as far as optimization is concerned. If your reward captures the real objective, RL can improve useful behavior. If your reward is incomplete, noisy, or hackable, the model will optimize the proxy instead of the real task. DeepMind’s write-up on specification gaming makes this point very clearly: the agent’s ingenuity is helpful only when the specification is correct. (Google DeepMind)
27) What is reward engineering?
Reward engineering is the design of the reward function, the verifier, the shaping terms, the penalties, and the monitoring strategy. In LLM RL, this includes deciding what counts as success, how partial progress is rewarded, what shortcuts are forbidden, and how to detect reward hacking. OpenEnv’s reward-design guide explicitly warns about reward hacking, sparse rewards, and conflicting signals as common pitfalls. (Meta-PyTorch)
28) What is reward hacking?
Reward hacking happens when a model finds a way to maximize the reward without actually doing the intended task. DeepMind describes this as specification gaming: the system satisfies the literal reward but not the real goal. Classic causes include poorly designed shaping rewards, missing constraints in the success condition, and simulator or verifier loopholes. (Google DeepMind)
29) Why is sparse reward a common problem?
If successful trajectories are too rare, the model may never get enough positive signal to improve. OpenEnv’s docs explicitly call sparse rewards a common pitfall because the agent may never find positive signal. RLVE work similarly notes that overly difficult tasks can yield consistently poor rewards and stall gradient-based learning. (Meta-PyTorch)
30) Why can dense rewards also be dangerous?
Dense rewards can speed up learning, but they can also create local optima and incentive misalignment. OpenEnv recommends starting simple and shaping carefully, because intermediate rewards can steer the model toward proxy behaviors. DeepMind gives the broader warning that poorly designed shaping can change the optimal policy itself rather than just helping the model reach the intended outcome faster. (Meta-PyTorch)

Common Pitfalls in Building RL Environments
31) What is the most common mistake when designing an RL environment?
Making the environment easy to verify but not faithful to the real task. A verifier that checks only the final string, a regex, or a narrow success pattern may be convenient, but it often misses equivalent correct answers or allows degenerate shortcuts. Recent verifier analysis on mathematical RL found that rule-based verifiers often reject correct but differently formatted answers, while model-based verifiers can be exploited to produce false positives during RL. (arXiv)
32) What goes wrong with weak verifiers?
Two opposite failure modes are common. Rule-based verifiers can be too brittle and produce false negatives when the answer is correct but phrased differently. Model-based verifiers can be too permissive and produce false positives that the policy learns to exploit. The verifier study on mathematical reasoning reports both problems and shows that stronger policies make verifier weaknesses more obvious. (arXiv)
33) Why is “just use an LLM as judge” often risky?
Because the judge becomes part of the optimization target. If the policy can find surface patterns that fool the judge, training can inflate reward without improving real task quality. That is exactly why model-based verifiers, despite better static accuracy, can be vulnerable during RL training. Use them carefully, stress-test them, and combine them with hard checks whenever possible. (arXiv)
34) What is a common environment-design pitfall for tool-using agents?
Not modeling realistic failure modes. Real APIs fail because of permissions, invalid formats, missing fields, timezones, or bad parameters. Hugging Face’s OpenEnv blog highlights examples like missing OAuth scopes and bad RFC3339 datetime formatting. If the environment hides these realities, the resulting policy will be overfit to a toy setup and brittle in deployment. (Hugging Face)
35) Why is static task difficulty a problem?
Because the learning signal collapses at both extremes. Tasks that are too easy stop teaching the model anything useful. Tasks that are too hard yield near-zero reward and also stop teaching. RLVE was proposed largely to solve this problem by dynamically adjusting task difficulty as the policy improves. (arXiv)
36) What is a common pitfall in environment diversity?
Training on too few task types. Recent RLVE results argue that scaling the number of environments improves generalizable reasoning capability, and Reasoning Gym was built around procedurally generated tasks across many domains for exactly this reason. A narrow environment set often produces narrow competence and fragile transfer. (arXiv)
37) Why do many RL environments fail to transfer to real-world performance?
Because they optimize the wrong abstraction level. If the environment is too toy-like, omits realistic constraints, or over-simplifies tool feedback, the model may become good at the benchmark but not at the actual workflow. This is a practical version of specification gaming: the benchmark is solved, the real job is not. (Google DeepMind)

Common Pitfalls in Reward Engineering
38) What is the biggest reward-engineering mistake?
Using a proxy metric as if it were the goal. Goodhart-style failures are everywhere in RL: token count, response format, test count, or intermediate progress can all become targets the model exploits. DeepMind’s examples of shaping mistakes and reward misspecification are the canonical warning here. (Google DeepMind)
39) Should I start with a complicated reward function?
Usually no. OpenEnv explicitly recommends starting simple, often with sparse success/failure reward, before layering in shaping terms. This makes debugging easier and reduces the chance that the model learns the wrong intermediate incentives before it learns the actual task. (Meta-PyTorch)
40) What happens when reward components conflict?
Learning becomes unstable or confused. OpenEnv lists conflicting signals as a common pitfall: if one term rewards brevity, another rewards verbosity, a third rewards format, and a fourth rewards exploration, the policy may oscillate or learn brittle shortcuts instead of coherent behavior. (Meta-PyTorch)
41) Why is binary reward often appealing?
Because it is easy to reason about and harder to game superficially. Label Studio’s RLVR overview notes that verifiable rewards are often binary and directly tied to correctness criteria, which makes evaluation simple and scalable. Binary reward is not always sufficient, but it is often a good starting point for precision-critical tasks like code and math. (Label Studio)
42) Why is binary reward sometimes not enough?
Because it can be too sparse, especially for long-horizon tasks. If success only happens at the very end, the model may not learn at all. That is where carefully designed shaping, step-level evaluation, or adaptive curriculum can help — but only if you can add them without creating easy-to-game shortcuts. (Meta-PyTorch)
43) How do I know whether my reward is being hacked?
Watch for rising reward without corresponding task-quality gains. Typical signs are strange formatting habits, repetitive surface patterns, degenerate short solutions, suspiciously high judge scores, or solutions that pass weak checks but fail stronger ones. The verifier case study is a strong reminder that static verification accuracy is not enough; you must observe what happens under optimization pressure. (arXiv)
44) What is a safe pattern for reward engineering?
Use layered verification. Start with hard outcome checks. Add anti-cheat constraints. Then add minimal shaping only where the sparse reward is too weak. Keep a holdout evaluator separate from the training reward when possible. This matches both OpenEnv’s “start simple, shape carefully” guidance and DeepMind’s warning about shaping altering the true objective. (Meta-PyTorch)

Common Pitfalls in RL Post-Training Pipelines with RLVR / RLVE / GRPO
45) What is a common mistake in GRPO training runs?
Using RL before the base model is ready. GRPO is powerful, but it is a post-training method, not a substitute for capability. TRL’s own GRPO examples start from instruct models and task datasets rather than from weak base checkpoints. If the model almost never produces a correct rollout, the reward signal is too sparse for productive RL. (Hugging Face)
46) Why does RL post-training plateau?
Because the model saturates the available prompt distribution or the reward signal no longer differentiates useful improvements. RLVE explicitly frames static data saturation as a problem and shows that adaptive environments can keep learning going after conventional RLVR pipelines flatten out. (arXiv)
47) Why can “more RL” make a model worse?
Because optimization pressure amplifies whatever the reward favors, including undesirable shortcuts. If the verifier is noisy, if the environment is unrealistic, or if the reward overvalues superficial structure, more training can push the model deeper into those artifacts rather than improving real competence. (arXiv)
48) What is a common pitfall in RLVR datasets?
Finite, static datasets get stale. Once the model has mastered or overfit their distribution, additional RL yields little signal. RLVE work argues that procedurally generated environments with adjustable difficulty are one way around this limitation. Reasoning Gym makes a similar case for unlimited data generation with controllable complexity. (arXiv)
49) Why do identical-looking GRPO runs produce different outcomes?
Because RL is highly sensitive to rollout quality, verifier behavior, reward scaling, task mix, generation parameters, and environment bugs. Even if the trainer code is the same, small differences in reward computation or environment behavior can change optimization dynamics substantially. The verifier study is a good reminder that the reward pipeline itself is part of the model. (arXiv)
50) What is a common pitfall when mixing many environments?
Using an unbalanced mixture. If some environments are much easier, much denser in reward, or much shorter in trajectory length, they can dominate training and starve harder but more important environments. RLVE’s adaptive-difficulty framing exists partly to keep the training distribution informative instead of letting it collapse into easy tasks. (arXiv)
51) Why are long-horizon tasks especially hard in RL post-training?
Because reward arrives late and useful trajectories are rare. Long tasks need either decomposition, better intermediate signals, stronger initialization, or curriculum. Otherwise, the rollout cost is high and the success rate stays near zero. This is one reason why adaptive environments and procedural curricula are getting attention. (arXiv)
52) What monitoring mistake do teams make most often?
They monitor the training reward but not actual behavior. Reward alone is not enough because the reward channel can be flawed. You need sampled rollout audits, stronger offline evaluation, and held-out environments or benchmarks. The verifier case study shows why this matters: reward can rise while real quality does not. (arXiv)
53) What is the safest way to structure an RL post-training pipeline?
A good pattern is:
start from a strong instruct or SFT checkpoint, use a task with a strong verifier, begin with simple reward, validate the environment thoroughly, run small-scale debug experiments, audit rollouts manually, then scale training and only later add curriculum or more shaping. This is consistent with TRL’s practical GRPO examples, OpenEnv’s reward guidance, and the lessons from verifier-failure studies. (Hugging Face)

Practical “What should we do in a hackathon?” FAQs
54) What kind of project is most likely to succeed in a hackathon?
Pick a task with:
a clear success condition,
a verifier you trust,
short to medium trajectory length,
few external dependencies,
and adjustable difficulty.
Good examples are code repair with tests, structured extraction with schema validation, grid or puzzle games, tool-using workflows with exact state checks, and browser tasks with explicit completion criteria. These are the sweet spot for RLVR and lightweight RLVE prototypes. (Label Studio)
55) What should we avoid building?
Avoid tasks that are subjective, hard to verify, require massive infrastructure, or depend heavily on an LLM judge without hard backstops. Also avoid environments whose failure cases you do not understand. If you cannot explain how the reward could be hacked, you are not ready to optimize it yet. (arXiv)
56) What is the best debugging order?
First debug the environment manually.
Then debug the verifier.
Then run scripted baseline policies.
Then run a frozen model.
Then run a tiny RL experiment.
Only then scale.
This order isolates bugs early and prevents you from blaming the optimizer for what is really an environment or reward bug. It follows directly from the fact that verifier reliability is foundational in RLVR. (arXiv)
57) What is one rule the team should remember?
Do not optimize a reward you have not tried to break yourself first. The easiest way to avoid reward hacking is to adversarially test your environment and reward design before the model does. (Google DeepMind)

58) Strong references for deeper learning
For GRPO and TRL:
TRL GRPO Trainer docs. (Hugging Face)
Hugging Face GRPO cookbook. (Hugging Face)
For RL environments and reward design:
OpenEnv reward design guide. (Meta-PyTorch)
OpenEnv tool-using environment examples. (Hugging Face)
For pitfalls and failure modes:
DeepMind on specification gaming. (Google DeepMind)
Pitfalls of rule-based and model-based verifiers. (arXiv)
For scalable environment-based training:
RLVE paper on adaptive verifiable environments. (arXiv)
Reasoning Gym. (OpenReview)

Here are solid Unsloth RL post-training recipes worth checking out, with a bias toward official or close-to-official examples.
59) Core Unsloth GRPO recipes
Qwen2.5 (3B) GRPO notebook
A straightforward starter recipe for GRPO with Unsloth. It covers data prep, training, inference, and saving, so it is a good baseline if you want the least opinionated end-to-end example. (GitHub)
Llama 3.1 (8B) GRPO notebook
Same general pattern, but on a larger model family. Useful if you want a more realistic “reasoning/capability uplift” recipe without jumping straight to very large models. (GitHub)
Gemma 3 (1B) GRPO notebook
A smaller-scale recipe that is easier to run and debug. Good for iterating on reward functions and rollout settings before spending more compute on larger checkpoints. (GitHub)
59.1) Advanced Unsloth GRPO recipes
Advanced Qwen3 (4B) GRPO notebook
This is one of the more interesting recipes because it adds more than the bare trainer loop. Unsloth’s June 2025 discussion explicitly calls out: proximity scoring for more nuanced rewards, OpenR1 dataset support, advanced templates, and “prefinetuning to skip GRPO format learning.” That makes it a better recipe when you care about reward shaping and format bootstrapping, not just getting GRPO to run. (GitHub)
HF LLM Course: Practical Exercise — GRPO with Unsloth
Not an Unsloth-maintained notebook repo entry, but it is a structured learning recipe that uses Unsloth specifically to fine-tune a model with GRPO for reasoning. It is a good companion when you want a didactic walkthrough instead of just notebook cells. (Hugging Face)
59.2) Environment / agent-style RL recipes
GPT-OSS 20B + 2048 game RL notebook
This is closer to “RL with an environment” than plain static-prompt RLVR. The notebook goal is explicitly to make GPT-OSS play 2048 with reinforcement learning / GRPO, which makes it a useful recipe if you want to move beyond math/code answer verification into interactive environment training. (GitHub)
59.3) Broader recipe collection
Unsloth notebooks repository
The main repo currently advertises “250+ Fine-tuning & RL Notebooks,” including GRPO and reinforcement learning notebooks. If you want the widest set of recipes in one place, this is the best starting point. (GitHub)
59.4) Useful adjacent recipes and examples
Scheduler GRPO example using Unsloth
A community example that trains a scheduling model with GRPO using Unsloth and QLoRA. It is useful because it shows a non-math, non-code structured-output task where rewards are tied to output format and schedule correctness. (Hugging Face)
SFT → GRPO pipeline example
There is a community “show and tell” example for a full SFT-then-GRPO pipeline. I would treat it as inspiration rather than an official recipe, but it is valuable if your intended workflow is “teach format first, then do RL.” (GitHub)
59.5) What these recipes collectively cover
Across these examples, the main recipe patterns are:
plain GRPO on reasoning-style tasks,
GRPO with better reward shaping like proximity scoring,
pre-SFT or preformatting before RL,
QLoRA-based memory-efficient RL fine-tuning,
and environment-style RL with game interaction. (GitHub)
59.6) Two gaps to keep in mind
One gap is multi-turn GRPO with stepwise rewards. There is a feature request asking for reward on each step plus a final reward, which suggests this is not yet a mature first-class recipe in Unsloth. (GitHub)
Another gap is notebook stability across versions/hardware. Several issue threads mention breakage or edge cases in GRPO notebooks, including fast inference assumptions, VRAM growth, and vision-GRPO issues. That does not make the recipes unusable, but it does mean you should pin versions and test on a small run first. (GitHub)
59.7) Best recipes by use case
If you want the simplest starting point:
Qwen2.5 (3B) GRPO
Gemma 3 (1B) GRPO (GitHub)
If you care about reward engineering:
Advanced Qwen3 (4B) GRPO (GitHub)
If you care about environment-style RL:
GPT-OSS 20B 2048 notebook (GitHub)
If you want the most guided learning path:
HF practical exercise with Unsloth + GRPO (Hugging Face)
If helpful, I can turn this into a curated table with columns for model, task type, reward type, hardware footprint, and what each recipe teaches.

Additional Resources:
OpenEnv Core (An interface library for RL post training with environments)
https://github.com/meta-pytorch/OpenEnv
OpenEnv-PyTorch Docs
https://meta-pytorch.org/OpenEnv/
HuggingFace OpenEnv Environments Hub
https://huggingface.co/openenv
https://huggingface.co/openenv/spaces
Tutorials to build, run and train RL environments and training pipelines
https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial
RL Training Examples: https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial/examples
RL Environment Examples: https://github.com/meta-pytorch/OpenEnv/tree/main/envs
Few additional YT Videos on building RL Environments:
https://www.youtube.com/watch?v=0airz7BhBiA
https://www.youtube.com/watch?v=ap4q4sAK4OY
https://www.youtube.com/watch?v=Jew4lhAiqnw 
https://openenv-india-apr-2026.lovable.app/ (Recommended: Chaptered Lectures)

Hackathon Self-Serve Guide: Build an RL Environment, Train an LLM, Ship a Demo
0) What you are building
The core idea is not just to fine-tune a text model, but to build a specialized LLM system that can act inside an environment, get feedback, and improve through reinforcement learning. The practical stack discussed here is:
Environment → verifier/reward functions → TRL trainer → Unsloth for efficiency → deployment on OpenEnv / Spaces.
A strong project usually looks like one of these,
Please refer to [External] Apr ‘26 OpenEnv Hackathon Themes for theme guidelines on selecting & forming problem statements.
1) Start with the right project idea
Pick a task that has all three of these properties:
The model can act step by step
You can verify success programmatically
The task is hard enough to be interesting, but not so hard that the model never succeeds
This last point matters a lot. RL only works if the probability of getting a good answer is greater than zero. If your task is so hard that the model never gets any reward, you will burn compute and learn nothing.
Please refer to [External] Apr ‘26 OpenEnv Hackathon Themes for theme guidelines on selecting & forming problem statements.
A useful rule: prefer tasks with crisp verification over tasks that only “look good” to a human. RL gets easier when the reward is objective.
2) Understand the minimum RL loop before you build
At a high level, your loop is:
Give the model a prompt
Let it generate an action, strategy, answer, or code
Execute that output in an environment or verifier
Convert the result into a reward
Update the model so higher-reward behavior becomes more likely
That is the practical mental model for RL here. The system samples many outputs, scores them, and shifts probability mass away from bad outputs and toward better ones.
One especially useful framing is that RL is like a more efficient version of repeated in-context improvement. Instead of repeatedly stuffing previous examples into the context, you let backpropagation store what worked into the weights.
3) Decide whether you need SFT first
Use this simple rule:
If you have a lot of good data, use SFT
If you do not have data but can verify outputs, use RL
In many practical cases, do a little SFT first, then RL
Why this matters:
SFT is generally more sample-efficient
RL is useful when you can test outcomes but cannot cheaply author ideal traces
RL often needs some warm start, formatting priming, or easy tasks first so that good rollouts happen at all
For hackathon teams, the best path is usually:
Start from a capable base/instruct model
Add light formatting or task scaffolding if needed
Use RL for improvement, not as magic from scratch
4) Design the environment before you design the trainer
Treat the environment as a first-class artifact. It should define:
reset(): start a fresh episode
step(action): apply an action and return the next result
state() / observation: what the agent sees
reward: what counts as progress or success
OpenEnv standardizes this so the same training code can work across many environments, instead of every team inventing a different API. That is one of the main reasons to use it in a hackathon.
Think about your environment in this order:
What does the agent observe?
What actions can it take?
What ends an episode?
How do you compute reward?
How do you stop abuse, infinite loops, or cheating?
5) Build the environment using OpenEnv
The intended workflow is to bootstrap an environment skeleton and then fill in the behavior. OpenEnv’s CLI creates the scaffolding for you. The environment is implemented as a Python package and exposed via a FastAPI app.
Your implementation typically defines:
action dataclass
observation dataclass
state representation
environment methods like reset and step
FastAPI wrapper / client-server interface
That gives you a clean separation:
the environment handles world dynamics and scoring,
the trainer handles optimization,
and the model just learns to act inside the interface.
6) Keep the task simple at first
Do not begin with your hardest benchmark. Start with the easiest version of your environment that still proves the concept. This is where curriculum learning helps.
A good progression:
easy tasks with short horizons,
medium tasks with a little more branching,
harder tasks only after the model starts getting non-zero reward.
The principle is simple: make success possible early. If the model never sees successful trajectories, learning stalls.
7) Design rewards carefully
Your reward function is your task specification. If it is weak, incomplete, or easy to exploit, the model will optimize the wrong thing very efficiently.
A strong reward design usually includes multiple components, for example:
execution success,
correctness,
format compliance,
timeouts,
resource usage,
safety constraints,
and anti-cheating checks.
One explicit recommendation was to use multiple independent reward functions, not just one. If you only have a single reward signal, it is easier for the model to hack it. Multiple independent checks reduce that risk.
For example, for a coding environment:
reward passing tests,
penalize timeouts,
reward format compliance,
reject use of forbidden globals,
and separately verify the function contract.
8) Protect yourself against reward hacking
Reward hacking is one of the biggest practical failure modes. The model may learn shortcuts that maximize your reward without solving the real task. Examples mentioned include:
editing timers,
caching results,
abusing globals,
mutating protected state,
or exploiting environment bugs.
What to do:
Use multiple independent reward functions
Lock down execution where possible
Add time limits
Avoid unrestricted global state
Sample outputs frequently and inspect them
Terminate or roll back runs if behavior drifts badly
A particularly practical recommendation was to use a locked-down function or restricted execution approach so the model cannot rely on undeclared globals or hidden cached state.
Also, do not just let training run forever without checking generations. Periodic human inspection is still necessary.
9) Use process-aware feedback when you can
Naively assigning the same final reward to every token is inefficient. If possible, use richer supervision that distinguishes good intermediate steps from bad ones. That is the idea behind process supervision.
In practice, this can be approximated by:
line-by-line checks,
step-level verifiers,
program trace analysis,
or LLM-as-a-judge for intermediate reasoning.
But be careful: LLM-as-a-judge can itself be gamed. Use it as one signal, not the only signal.
For a hackathon, outcome-based verification plus a few lightweight process checks is usually the sweet spot.
10) Pick the right training stack
The intended stack here is:
TRL for RL training algorithms
Unsloth to make RL training and inference more efficient
OpenEnv to standardize environment interaction
This combination works because:
OpenEnv gives you a common environment interface
TRL gives you RL trainers like GRPO
Unsloth reduces memory use and improves efficiency on top of TRL
One of the practical examples used the same prompt repeated many times, routed through an environment, with TRL driving training and Unsloth helping with performance.
11) Prefer GRPO / RLVR style training for verifiable tasks
The RL setup discussed here leans toward RL with verifiable rewards:
instead of a learned reward model,
use a verifier, test harness, regex check, executor, or environment.
GRPO was described as a more efficient evolution relative to older PPO-style setups, especially by simplifying away parts like the value model.
For hackathon purposes, the key practical takeaway is:
if the task is verifiable,
build the verifier first,
then plug that verifier into RL training.
12) Keep inference fast
One important point: in RL for LLMs, inference can dominate total runtime. Over time, rollout generation often becomes the bottleneck, not the optimizer step.
That means your project speed depends heavily on:
fast sampling,
tight environment loops,
low-overhead execution,
and efficient model runtime.
This is one reason Unsloth matters in the stack, and another reason to avoid overly heavy environments early in the hackathon.
13) Deploy your environment early
OpenEnv environments are designed to be deployed as Hugging Face Spaces, which provide:
a running server,
a Git repository,
and a container registry.
That gives you several ways to work:
interact with the remote Space directly,
install the client code from the repo,
pull and run the container locally,
or run the FastAPI app locally via Python/Uvicorn.
Why this is good for a hackathon:
one shared source of truth,
easier collaboration,
easier demos,
easier switching between local and remote execution.
A good habit is to deploy an early version of the environment before training seriously. That catches API and packaging issues early.
14) Scale only after the environment is stable
There was a dedicated tutorial flow around:
environment,
deployment,
scaling,
training with TRL and Wordle.
Follow the same order.
Do not start with scale. First confirm:
reset works,
step works,
rewards are sensible,
timeouts work,
logs are visible,
and the environment can be run locally and remotely.
Only then:
increase batch sizes,
duplicate prompts or tasks,
expand task diversity,
and benchmark throughput.
15) Monitor the right things during training
Do not watch only one scalar. Monitor:
overall reward,
individual reward function columns,
success indicators,
timeout frequency,
and generated strategies over time.
A very concrete suggestion was:
watch whether the reward is going up,
and separately watch critical columns like “function works.”
Also inspect actual generations during training. A rising reward is not enough if the model is learning to exploit bugs.
16) Save models correctly
If you use QLoRA / LoRA-style training, be careful when saving. One explicit warning was:
Do not upcast a 4-bit model to 16-bit and then merge the LoRA weights naively. That can badly damage model quality. Instead, use the proper merged-save path, or use the adapters directly.
For participants, that means:
keep your training save path simple,
test post-training inference immediately,
and do not leave export until the end.
17) How to structure your team over the hackathon
A very effective team split is:
Person A: Environment
builds reset/step/state
adds timeouts and safety constraints
makes local and remote execution work
Person B: Verifier / Rewards
writes multiple reward functions
adds anti-hacking checks
makes failure cases visible
Person C: Training
sets up TRL + Unsloth
runs experiments
tracks metrics and generations
Person D: Demo / Product
prepares the Space demo
creates a simple interface
records examples and final benchmarks
This split matches the way the stack naturally decomposes in practice.
18) A practical 1-day execution plan
Phase 1: Pick a narrow task
Choose a small, verifiable environment. Avoid huge long-horizon tasks first.
Phase 2: Build the environment
Use OpenEnv init, implement reset/step/state, and get a local loop working.
Phase 3: Build rewards
Add at least 2–4 independent reward checks, plus timeout and anti-cheat logic.
Phase 4: Deploy
Push to a Space or run locally via container/Uvicorn so teammates can use the same environment.
Phase 5: Train small
Run a tiny TRL + Unsloth experiment first. Look at outputs, not just metrics.
Phase 6: Inspect for hacking
Sample generations. Check for globals, hacks, environment abuse, or suspicious shortcuts.
Phase 7: Add curriculum
If the model gets zero reward too often, simplify tasks or add easier start states.
Phase 8: Train bigger
Only after the loop is stable should you increase scale, batch size, or environment diversity.
Phase 9: Save and demo
Export the trained model correctly, test inference, and show before/after behavior.
19) What judges or reviewers will likely find compelling
The strongest hackathon projects usually show:
a clear environment design,
objective reward functions,
evidence that the model improved,
prevention against reward hacking,
a reproducible deployment story,
and a sharp demo.
A simple but strong demo format is:
baseline model attempt,
reward/verifier output,
trained model attempt,
measurable improvement,
short explanation of safeguards.
20) Suggested problem statement theme directions
Please Refer to [External] Apr ‘26 OpenEnv Hackathon Themes
21) Common mistakes to avoid
Picking a task so hard that success probability is zero
Using only one reward function
Not checking for reward hacking
Training before the environment is stable
Relying only on average reward and not inspecting outputs
Forgetting timeouts and sandbox limits
Saving LoRA/QLoRA models incorrectly

22) Learning Resources

(Recommended) RL Environment Lecture Chapters:
https://openenv-india-apr-2026.lovable.app/


Module 1: Why OpenEnv? (~7 min)
▸ Workshop 8:02–15:05 — https://www.youtube.com/watch?v=1jU05MlENOI&t=482s
▸ Sanyam: RL loop, fragmented env APIs, OpenEnv as universal interface, Gymnasium spec + Docker
▸ Alt: Mega Lecture 40:01–46:00 — https://www.youtube.com/watch?v=Jew4lhAiqnw&t=2401s

Module 2: Using Existing Envs (~7.5 min)
▸ Workshop 35:33–43:05 — https://www.youtube.com/watch?v=1jU05MlENOI&t=2133s
▸ Ben: Hub org, env collections, 3 Space interfaces (server/repo/registry), from_hub
▸ Alt: Mega Lecture 1:24:11–1:30:00 — https://www.youtube.com/watch?v=Jew4lhAiqnw&t=5051s

Module 3: Deploying Envs (~9 min)
▸ Mega Lecture 1:30:00–1:39:07 — https://www.youtube.com/watch?v=Jew4lhAiqnw&t=5400s
▸ Ben: live openenv init, scaffold, running locally, openenv push, Docker run from Space
▸ Alt: Workshop 43:05–48:30 — https://www.youtube.com/watch?v=1jU05MlENOI&t=2585s

Module 4: Building Your Own (~6.5 min)
▸ Workshop 43:45–50:20 — https://www.youtube.com/watch?v=1jU05MlENOI&t=2625s
▸ Ben: scaffold files, business logic (reset/step), models, client, publishing
▸ Alt: Mega Lecture 1:33:30–1:39:07 — https://www.youtube.com/watch?v=Jew4lhAiqnw&t=5610s

Module 5: Training + TRL (~14 min)
▸ Mega Lecture 1:53:20–2:07:12 — https://www.youtube.com/watch?v=Jew4lhAiqnw&t=6800s
▸ Lewis: Wordle GRPO walkthrough — rollout function, reward shaping, GRPOTrainer, live training
▸ Alt: Workshop 22:24–34:12 — https://www.youtube.com/watch?v=1jU05MlENOI&t=1344s
