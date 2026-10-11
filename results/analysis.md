# Task 6 - Analysis

**6.1 Why do logistic regression (Set R) and the linear SVM score almost the same?**
Both models draw ONE straight line (a flat boundary) through the 6 raw features. They choose the line in slightly different ways, so they land on almost the same line and make almost the same mistakes. Logistic regression on Set R got 72.64% test accuracy and the linear SVM got 73.64%, only 1 percentage point apart. Both are low and both have zero or very few false alarms but hundreds of missed faults (657 and 633). This tells us the data is NOT linearly separable: no single straight line can split fault from no-fault.

**6.2 Signs of the Set Q weights**
The weights on the squared current terms (Ia^2, Ib^2, Ic^2) are positive: +18.89, +4.75, +13.96. The weights on the squared voltage terms (Va^2, Vb^2, Vc^2) are negative: -6.03, -6.46, -6.14.
Why: during a fault the current jumps, so its size becomes big. A big current gives a big I^2 whether the current is positive or negative at that instant, so a big I^2 pushes the answer toward "fault" (positive weight). During a fault the voltage sags (gets smaller), so V^2 becomes small. A big V^2 means the line is healthy, so the weight is negative: a small V^2 pushes toward "fault".

**6.3 Support vectors**
Best linear SVM (C=1): 5834 support vectors, which is 81.0% of the 7200 training rows. Best RBF SVM (C=100, gamma=1): 266 support vectors, which is 3.7%.
A support vector is a training point that sits on the margin or on the wrong side of it. A very large number (like most of the training set, as with the linear SVM) means the model cannot separate the classes well: almost every point is close to the boundary or on the wrong side. This is what happens when a straight line is forced on data that is not linearly separable. The RBF SVM can bend its boundary around the data, so it needs far fewer support vectors.

**6.4 What happens when C increases?**
From Table 3, as C goes from 0.1 to 100:
- Linear SVM: validation accuracy stays at 72.58% and support vectors stay about the same (5849 to 5839). A bigger C cannot help because a straight line is the real problem.
- RBF SVM (gamma=0.1): accuracy goes up from 95.00% to 99.54% and support vectors go down from 1347 to 174.
- RBF SVM (gamma=1): accuracy goes up from 98.04% to 99.67% and support vectors go down from 848 to 266.
Why: C is the penalty for a wrong or too-close point. A small C is relaxed: the margin is wide and many points are allowed inside it, so there are many support vectors and a simpler boundary. A large C is strict: the margin is narrower and the model tries hard to classify every training point correctly, so there are fewer support vectors. A very large C can overfit (memorise the training noise), so we choose C on the validation set.

**6.5 Is our test accuracy optimistic or pessimistic for a real relay?**
Optimistic. The rows are consecutive time samples that look almost the same. Because we split randomly, a test row has a near-twin in the training set, so the model is partly tested on things it has already seen. A real relay sees new fault events, different fault types and locations, and noise, so real accuracy will usually be lower than our 99.58%. A fairer test would split by whole time blocks or by whole fault events.

**6.6 Which model would I deploy?**
I would deploy the RBF SVM (C=100, gamma=1). In Table 4 it has only 6 missed faults and 4 false alarms (accuracy 99.58%). A missed fault means the breaker does not trip, which is dangerous and expensive, so I choose the model with the fewest missed faults first, then the fewest false alarms. Set R logistic regression (657 missed) and the linear SVM (633 missed) miss hundreds of faults, so they are not safe. Set Q logistic regression is much better (97.88%) but still misses 45 faults, which is about 7 times more than the RBF SVM. Set Q logistic regression is simpler and faster (just a sum and a sigmoid), so it is a good backup if the relay has very little computing power, but for safety the RBF SVM is the better choice.