# Can a Straight Line Detect a Power-Line Fault? Logistic Regression vs SVM

## 1. The problem
When a wire in a power system touches the ground, or two wires touch each other, the line current jumps and the voltage drops. A protective relay has only a few milliseconds to notice this and trip the breaker. If it misses a fault, equipment can burn and people can get hurt.

We looked at one single instant of three line currents (Ia, Ib, Ic) and three line voltages (Va, Vb, Vc) and asked: **fault or no fault?** Our big question was: can a straight-line decision boundary do this from raw instantaneous numbers? If not, what is the easiest fix: better features, or a better classifier?

## 2. The data
We used the Kaggle dataset "Electrical Fault Detection and Classification", a three-phase power system simulated in MATLAB/Simulink. The file has 12,001 rows. Each row is one instant of the six measurements, with a label: 1 for fault and 0 for no fault. We removed two completely empty columns (Unnamed: 7 and Unnamed: 8), renamed the label column to `fault`, and checked that all six measurement columns are numeric. There were no missing values. The data has 5,496 fault rows and 6,505 no-fault rows, so the two classes are reasonably balanced.

We split the rows 60% / 20% / 20% into train, validation and test sets with a fixed random seed (42). Then we standardised every feature using only the training mean and the training standard deviation, so nothing leaks from the validation or test rows.

| Item | Value |
|---|---|
| Raw rows / raw columns | 12001 / 9 |
| Columns removed | Unnamed: 7, Unnamed: 8 |
| Missing values after cleaning | 0 |
| Fault rows / no-fault rows | 5496 / 6505 |
| Train / validation / test rows | 7200 / 2400 / 2401 |
| Fault fraction in train / validation / test | 0.4607 / 0.4512 / 0.4565 |

## 3. What the data looks like
![Figure 1: Line currents for the first 2000 rows. The red zone is a fault.](fig1a)

The rows are consecutive time samples. Outside the red zone the three currents are small, smooth waves of about 60 A. When the fault starts (around row 505), phase A shakes for a moment and then swings between about -875 A and +875 A, roughly fifteen times bigger than normal. The fault lasts about three cycles, then the currents return to normal.

![Figure 2: Ia against Ib, coloured by fault.](fig1b)

The no-fault points form one small blob in the centre, where both currents stay below about 80 A. The fault points trace huge ellipses around that blob, and a few scattered points sit far to the left. Some fault points also pass through the middle of the blob, because the wave crosses zero.

![Figure 3: Histogram of Ia for fault and no-fault rows.](fig1c)

The no-fault values of Ia are packed in a narrow band around zero (about -75 A to +75 A). The fault values are spread over the whole range from -880 A to +880 A, with long tails on both the positive and the negative side. In the middle there is a noticeable pile of fault rows (roughly one in five) hiding inside the no-fault band.

**Could a single straight line in the (Ia, Ib) plane separate the two classes?** No. A straight line cuts the plane into two halves, so it can only say "big on this side, small on the other side". But the faulty points surround the normal points on every side. The reason is that current is an AC wave. During a fault its size becomes large, but at any single instant the value can be big and positive, big and negative, or close to zero while the wave crosses zero. What we need is a rule like "far from zero on either side", which a straight line cannot give.

## 4. Logistic regression by hand
We wrote logistic regression ourselves in NumPy: the sigmoid, the hypothesis, the cost J(theta), and batch gradient descent for 2000 iterations. The cost starts at 0.693 when theta is zero, as it should. We tried four learning rates (0.01, 0.1, 1 and 10) and chose the largest one for which the cost went down at every iteration.

We used two feature sets. **Set R** is the six raw numbers. **Set Q** is the six raw numbers plus their six squares (12 features).

![Figure 4: Cost against iteration for Set R.](fig2R)

For Set R, the rates 0.01, 0.1 and 1 all go down smoothly and settle at the same cost (about 0.677). The rate 10 is too big: the cost jumps up and down forever and never settles. So we chose alpha = 1.

![Figure 5: Cost against iteration for Set Q.](fig2Q)

For Set Q, all four rates go down at every step, and the biggest one is the fastest. So we chose alpha = 10. Its final cost (0.053) is far lower than Set R's.

| Feature set | Chosen alpha | Final J(theta) | Train accuracy | Validation accuracy |
|---|---|---|---|---|
| Set R | 1 | 0.67716 | 72.58% | 73.08% |
| Set Q | 10 | 0.05343 | 97.88% | 97.83% |

## 5. Support vector machines
We then trained support vector machines (scikit-learn SVC) on the six raw standardised features. We tried a linear SVM and an RBF-kernel SVM, with C in {0.1, 1, 10, 100} and gamma in {0.1, 1}, and kept the best setting of each kernel based on validation accuracy.

![Figure 6: SVM validation accuracy against C.](fig3)

| Kernel | C | gamma | Validation accuracy | Support vectors |
|---|---|---|---|---|
| linear | 0.1 | - | 72.58% | 5849 |
| linear | 1 | - | 72.58% | 5834 |
| linear | 10 | - | 72.58% | 5837 |
| linear | 100 | - | 72.58% | 5839 |
| rbf | 0.1 | 0.1 | 95.00% | 1347 |
| rbf | 0.1 | 1 | 98.04% | 848 |
| rbf | 1 | 0.1 | 97.50% | 723 |
| rbf | 1 | 1 | 99.17% | 482 |
| rbf | 10 | 0.1 | 99.21% | 381 |
| rbf | 10 | 1 | 99.62% | 342 |
| rbf | 100 | 0.1 | 99.54% | 174 |
| rbf | 100 | 1 | 99.67% | 266 |

The linear SVM stays flat at 72.6% whatever C we pick. The RBF SVM climbs from 95% to almost 99.7%. The best linear model is C = 1; the best RBF model is C = 100 with gamma = 1.

## 6. Final test results
We used the test rows only once, at the very end. A missed fault means the breaker does not trip,

| Model | Accuracy | TN | False alarms | Missed faults | TP |
|---|---|---|---|---|---|
| Logistic regression, Set R | 72.64% | 1305 | 0 | 657 | 439 |
| Logistic regression, Set Q | 97.88% | 1299 | 6 | 45 | 1051 |
| Linear SVM (C = 1) | 73.64% | 1305 | 0 | 633 | 463 |
| RBF SVM (C = 100, gamma = 1) | 99.58% | 1301 | 4 | 6 | 1090 |

The test set has 1,096 real faults. The two straight-line models missed more than half of them (657 and 633). Adding squares cut the misses to 45, and the RBF SVM missed only 6.

The Set Q weights, learned by gradient descent:

| Weight | Value |
|---|---|
| theta0 (intercept) | 24.9209 |
| Ia | 2.3273 |
| Ib | -5.3581 |
| Ic | -1.7365 |
| Va | -0.5336 |
| Vb | 0.4167 |
| Vc | 0.0986 |
| Ia^2 | 18.8912 |
| Ib^2 | 4.7543 |
| Ic^2 | 13.9569 |
| Va^2 | -6.0276 |
| Vb^2 | -6.4643 |
| Vc^2 | -6.1407 |

## 7. Answers to our analysis questions
**Why do logistic regression on Set R and the linear SVM score almost the same?** Both draw one flat boundary through the six raw features, so they land on almost the same line: 72.64% and 73.64%, both with hundreds of missed faults. This tells us the data is not linearly separable.

**What do the signs of the Set Q weights mean?** The weights on the squared currents are positive (+18.89, +4.75, +13.96) and the weights on the squared voltages are negative (-6.03, -6.46, -6.14). In a fault the current becomes big, and squaring removes its sign, so a big I^2 means "fault". In a fault the voltage sags, so V^2 becomes small, and a small V^2 also means "fault".

**What does the number of support vectors tell us?** The best linear SVM uses 5,834 support vectors, 81% of the 7,200 training rows. The best RBF SVM uses only 266 (3.7%). Support vectors are points on the margin or on the wrong side of it. A huge number means the boundary does not fit the data, as happens when a straight line is forced on data that is not linearly separable.

**What does C do?** C is the penalty for a wrong or too-close point. A small C gives a wide, relaxed margin with many support vectors. A large C is strict: the margin narrows and the model tries hard to get every training point right. For the RBF SVM, as C goes from 0.1 to 100, accuracy rises (95.00% to 99.54% for gamma = 0.1) and support vectors fall (1347 to 174). The linear SVM does not change at all, because the straight line is the real problem, not C.

**Is our test accuracy realistic?** Probably optimistic. The rows are consecutive time samples that look almost identical, and we split them randomly, so many test rows have a near-twin in the training set. A real relay sees new fault events with different places, types and noise, so its true accuracy would be lower than our 99.58%. A fairer test would split by whole time blocks or fault events.

**Which model would we deploy?** The RBF SVM (C = 100, gamma = 1). It has only 6 missed faults and 4 false alarms. A missed fault is the dangerous mistake, so we rank models by missed faults first and false alarms second. Set Q logistic regression is simpler and faster (a sum and a sigmoid) and reaches 97.88%, but it misses 45 faults, about seven times more. It could be a backup if the relay had very little computing power.

## 8. The front end
We also built a small Streamlit web page. You type Ia, Ib, Ic, Va, Vb and Vc, press Predict, and see the Set Q logistic regression and the RBF SVM side by side. In the example below, both models say FAULT, and the logistic regression gives a fault probability of 0.894.

![Figure 7: The Streamlit front end.](fig4)

## 9. What we learned
1. A straight line cannot detect faults from raw instantaneous values (both stopped at about 73%).
2. Two easy fixes work: add squared features (97.88%) or use an RBF-kernel SVM (99.58%).
3. In protection, a missed fault is worse than a false alarm, so we must read the confusion matrix and not only the accuracy.
4. A random split of time-series data can make the test accuracy look better than reality.
