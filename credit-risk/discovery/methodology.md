# Methodology

What is the purpose of Data Science?

At its best, Data Science is a resourceful tool to make informed decisions when compared to old-school hunch-in-the-gut decision making. Basing decisions on gut feelings or small samples of data puts a team, and sometimes a whole company, at risk due to unknown or undisclosed bias and motivations.

So, Data Science is here to save the day, right? Not exactly.

A tool is only as good as the person yielding it. Many practitioners end up making the same mistakes of the past, but with fancier laptops and jargon. What is that mistake? One's desire for success can sometimes overrule better judgment and hide the risk it entails. Most of the time, it's an honest mistake of trying to please, help the team get to short-term goals; seldom it's greed or pride, and rarely malicious.

So how can Data Science be used in the proper way and bring dividends in the long run?

The best way to mitigate the risk of one's conscious and unconscious bias is to address is straight on. What does that mean? It means gathering enough domain and problem understanding to form your opinion about what you expect the solution or answer to be, and making one or several predictions **before doing any deep-dives on the data**. One more crucial step is to then make explicit scenarios or evidence that would make you lose confidence in your predictions, and perhaps bend towards alternative ideas. In summary:
 - state the problem clearly
 - state your initial opinion of what outcome you expect -- **make predictions**
 - state what evidence would make you lose confidence in your predictions
 - use the data to test your predictions
 - gather results, brainstorm and repeat

All of those steps make up an experiment, which is the Scientific Method materialized, and that's what Data Science can bring to the table.

Predictions we make are also known as hypothesis:
 - null hypothesis is what you believe to be true, your prediction
 - alternate hypothesis is the competing prediction with yours, that would take *enough* evidence for you to be convinced

Now, what does *enough* mean anyways? One may say: "I want to be 100% sure about that!", but we know that in the real world only death and taxes are guaranteed. Instead, we have to understand that every experiment comes with its trade-offs and uncertainties. For instance, this case study. I would love to have a whole month just to dissect every angle and test every possible hypothesis, but my time is constrained -- a trade-off will be made for depth.

What about the uncertainty in our experiments?
 
That's where Statistics comes in. It basically boils down to processes, techniques, theorems, etc, to help us estimate uncertainty in our tests of hypothesis, in order to understand what's required to minimize it. Just like our conscious and unconscious bias, we must address uncertainty loudly and clearly. Gladly, Statistics is here to help.

Now, let's start with my predictions for this case study. Note that I have done initial data discovery, and that's in fact required in order for me to understand the product and domain. Of course, I also brushed up a bit on some concepts related to credit risk. However, I did not analyze the data itself in order to gain insight into its behavior before making the following predictions.

## Hypothesis
Lorem Ipsum



## Avoiding Data Leak
DEFINE DATA LEAK

These columns will be removed from feature engineering:
 - REPAYMENT_ID
 - FINAL_COST
 - REPAYMENT_UPDATED
 - SPENT
 - TOTAL_FINAL_AMOUNT
 - FIRST_TRIAL_BALANCE
 - FIRST_TRAIL_DELAYS
 - PAYMENT_AMOUNT
 - LOAN_PAYMENT_DATE
 - REPAYMENT_AMOUNT
 - CUMMULATIVE_OUTSTANDING
 