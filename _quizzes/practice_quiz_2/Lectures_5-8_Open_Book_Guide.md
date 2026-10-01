# Lectures 5–8: Open-book exam guide and practice

Based on all four uploaded lecture notebooks, their solution notebooks, and practice_quiz-solutions.ipynb. The practice questions below are new exercises, not predictions of the exam. Examples use the approaches taught in your notes. English terminology is paired with short Chinese reminders where useful.

## How to use this guide

Search for a function name or pattern during the exam. Read the question's boundaries carefully: “above” means `>`, “at least” means `>=`, and “in total” does not mean “consecutively.” Practice Questions 1–6 mirror quiz A–F. Questions 7–18 cover additional lecture material. The answer key is at the end.

## Quick lookup

| Task | Code / pattern | Important detail |
|---|---|---|
| Pick one item | `random.choice(items)` | Returns one element |
| Pick k with replacement | `random.choices(items, k=k)` | Returns a list; repeats allowed |
| Pick k without replacement | `random.sample(items, k=k)` | No repeated positions; k cannot exceed list length |
| Shuffle a list | `random.shuffle(items)` | Changes list; returns `None` |
| Integer from a through b | `random.randint(a, b)` | Both endpoints included |
| Repeat random sequence | `random.seed(1337)` | Reset before repeating the same calls |
| Binomial probability | `stats.binom.pmf(k, n, p)` | k successes in n fixed trials |
| Negative binomial probability | `stats.nbinom.pmf(k, n, p)` | k failures before nth success |
| Numerical vector | `np.array([1, 2, 3])` | Arithmetic is element by element |
| Equally spaced points | `np.linspace(a, b, N)` | N points; endpoint included by default |
| Points with fixed step | `np.arange(a, b, step)` | Stop excluded; float steps can have rounding issues |
| Repeat known sequence | `for value in items:` | Each item once, unless interrupted |
| Repeat until condition changes | `while condition:` | Condition tested before each iteration |
| Stop loop | `break` | Leaves nearest enclosing loop |
| Skip iteration | `continue` | Skips remaining body; continues loop |
| Find first position | `items.index(value)` | Raises `ValueError` if missing |
| Count occurrences | `items.count(value)` | Returns 0 if missing |
| Last element | `items[-1]` | Negative indexing counts backward |
| Slice | `items[start:stop]` | Stop excluded |
| Independent NumPy copy | `a.copy()` | Avoids modifying original through shared data |
| Unique values | `set(items)` | Unordered; removes duplicates |
| Empty set | `set()` | `{}` is an empty dictionary |
| Safe set removal | `if x in s: s.remove(x)` | Avoids `KeyError` |
| Swap variables | `a, b = b, a` | Right side evaluated before assignment |

## Lecture 5 — Libraries, randomness, probability, NumPy, plotting

### 1. Imports and libraries

Standard libraries ship with Python: `random`, `math`, `statistics`. Third-party packages include NumPy, SciPy, pandas, and matplotlib. A package must be installed in the active environment before you can import it. Importing makes its names available; it does not install it. Anaconda may already include third-party packages.

```python
import random
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
```

Aliases such as `np`, `stats`, and `plt` are names you choose. Use the alias consistently. `ModuleNotFoundError` means the current environment cannot find the package.

Built-ins reviewed in the lecture: `print(x)`, `type(x)`, `round(x, 2)`, `abs(x)`, `len(items)`. For printing a variable inside text:

```python
name = "June"
score = 95
print(f"{name} scored {score}.")
```

### 2. Random selection and seeds 随机抽样

```python
items = ["A", "B", "C", "D"]
one = random.choice(items)
with_repeats = random.choices(items, k=3)
without_repeats = random.sample(items, k=3)
die = random.randint(1, 6)
```

`choice` returns an item; `choices` and `sample` return lists. Sampling without replacement prevents selecting the same list position twice. If the input itself has duplicate values, `sample` can still return equal values; the quiz's deck has unique values.

```python
random.shuffle(items)
print(items)
```

Do not write `items = random.shuffle(items)`: this replaces `items` with `None`. A shuffled independent list can be made with `random.sample(items, k=len(items))`.

```python
random.seed(1337)
first = random.sample(items, k=2)
random.seed(1337)
second = random.sample(items, k=2)
print(first == second)  # True
```

A seed starts a reproducible sequence. It does not make every subsequent call return the same result. The input and sequence of random calls must also match. `random` and `np.random` have separate random states: `random.seed(...)` does not seed NumPy.

NumPy calls appearing in your notes:

```python
np.random.seed(88)
values = np.random.uniform(low=-5, high=10, size=25)
birthdays = np.random.randint(0, 365, size=(5, 5))
```

`np.random.randint` excludes its upper bound: 0 through 364 here. This differs from `random.randint`! The birthday array contains 25 simulated dates; `len(np.unique(birthdays)) < 25` detects at least one repeated date.

### 3. Binomial vs. negative binomial

| Distribution | Fixed quantity | Count being measured | Support |
|---|---|---|---|
| Binomial | n trials | k successes | 0 through n |
| Negative binomial | n successes required to stop | k failures before stopping | 0, 1, 2, … |

Binomial assumes independent trials with the same success probability p.

\[P(X=k)=\frac{n!}{k!(n-k)!}p^k(1-p)^{n-k}.\]

```python
n = 10
p = 0.5
k = 5
prob = stats.binom.pmf(k, n, p)
# 0.24609375, approximately
```

`math.factorial(5)` gives 120. Direct formulas with large factorials and floating-point powers can run into conversion, overflow, or underflow problems. Python's integers themselves can grow very large; the difficulty comes when mixed with floating-point arithmetic. Specialized library routines handle these calculations more reliably.

For the negative binomial, SciPy's convention is:

\[P(K=k)=\frac{(k+n-1)!}{k!(n-1)!}p^n(1-p)^k.\]

**Correction to Lecture 5:** its printed negative-binomial formula reverses the exponents on p and 1−p. With p=0.5 this happens to give the same answer, masking the typo. Use `nbinom`, as the solution notebook does; `binom` in the original exercise prompt is also a typo. Here n is successes required, not total flips.

```python
k_vals = np.arange(0, 11)
prob = stats.nbinom.pmf(k_vals, 10, 0.5)
plt.bar(k_vals, prob)
plt.xlabel("Tails before the 10th head")
plt.ylabel("Probability")
plt.show()
```

Plotting only 0–10 failures does not show the whole negative-binomial distribution, so those probabilities need not sum to 1. For binomial, plotting 0–n covers all outcomes. Raising p from 0.5 to 0.7 shifts a binomial distribution toward more successes.

### 4. Arithmetic and arrays

Use `**` for powers, not `^`. Multiplication/division precede addition/subtraction. Parentheses control grouping.

```python
27 / (12 - 3)       # 3.0
(5 + 3) * 2        # 16
5 * 3 + 2 * 4      # 23
(5 + 3) * 2 / 4    # 4.0
```

Lists concatenate; NumPy arrays calculate entry by entry:

```python
print([1, 2] + [3, 4])                     # [1, 2, 3, 4]
print(np.array([1, 2]) + np.array([3, 4]))  # [4 6]
```

```python
a = np.array([2, 4, 6])
b = np.array([1, 2, 3])
c = a * b          # [2, 8, 18]
d = 5 - a          # [3, 1, -1]
e = (a - b) / 2    # [0.5, 1, 1.5]
f = b**a           # [1, 16, 729]
g = 2**a           # [4, 16, 64]
h = np.exp(a + b)
```

These operations use compatible array shapes; the introductory examples have equal lengths or combine an array with a scalar. `a*b` is elementwise multiplication, not a dot product.

| Mathematics | NumPy |
|---|---|
| π; e | `np.pi`; `np.e` |
| ln(x) | `np.log(x)` |
| e to the x | `np.exp(x)` |
| square root | `np.sqrt(x)` |
| sine; cosine | `np.sin(x)`; `np.cos(x)` |
| Sum elements | `np.sum(a)` |

Trig functions use radians. NumPy functions work on arrays. `math` remains useful for scalar operations and functions such as factorial; NumPy does not literally replace every use of `math`.

```python
x = 0.5
print(np.pi * x**2)
print(np.exp(-x**2) / np.sqrt(2*np.pi))
print(np.log((2*x + 1) / (x + 2)))
```

In your Lecture 5 copy, Exercise 4's last line prints the fraction but omits `np.log`; the last line above evaluates the requested logarithm.

For ordinary real-valued NumPy arrays: 0/0 produces `nan`; positive nonzero/0 produces `inf`; a negative base raised to a noninteger real power can produce `nan`. Such operations generally issue runtime warnings and continue. A warning differs from an exception that stops execution. `nan` means “not a number.” NumPy's signed `inf` here should not be read as a claim that an ordinary two-sided limit exists.

### 5. Plotting templates

```python
x = np.linspace(0, 2, 25)
y = np.sin(np.pi*x) + 1
plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("q(x)")
plt.title("q(x) = sin(pi*x) + 1")
plt.show()
```

`linspace`'s third argument is number of points, not distance between points. `plt.scatter` draws separate points; `plt.plot` connects points; `plt.bar` uses category positions and heights; `plt.hist` bins raw observations.

Composition: calculate the inner function first and graph against the original input.

```python
t = np.linspace(2*np.pi, 4*np.pi, 25)
g = np.sin(t + 3)
f_of_g = g**2 - 5*g + 2
plt.scatter(t, f_of_g)
plt.show()
```

## Lecture 6 — Conditionals and nested lists

### 1. Comparisons and logic

`==` checks equality; `=` assigns a value. Comparisons: `!=`, `<`, `<=`, `>`, `>=`. Logical operators: `and`, `or`, `not`. Membership: `x in items`, `x not in items`.

```python
print(1 <= 3 < 5)  # True
print("Monday" in ["Monday", "Friday"])  # True
```

For a right triangle, `a**2 + b**2 == c**2` assumes c is the largest side. Without that assumption, test each possible hypotenuse with `or`.

### 2. Branching 分支

```python
if score >= 80:
    label = "good"
elif score >= 60:
    label = "okay"
else:
    label = "low"
print(label)
```

Only the first true branch of an `if/elif/else` chain runs. Later tests are not evaluated once a branch is chosen. Separate `if` statements are independent and can all run. An `else` is optional; it has no condition. Colons and indentation are required. Put code outside the blocks when it should run regardless of the branch.

```python
if is_pink:
    plt.bar(categories, counts, color="hotpink")
else:
    plt.bar(categories, counts)
plt.show()
```

Test boundary values explicitly. For the lecture's random integer in 1–20:

```python
if n < 5:
    label = "[1, 5)"
elif n <= 10:
    label = "[5, 10]"
elif n < 15:
    label = "(10, 15)"
else:
    label = "15 or above"
```

The second branch already implies n≥5 because the first test failed. This shorthand depends on knowing n is in 1–20.

Nested conditions solve the quiz temperature problem without `and`: first choose the scale; inside that branch, compare temperature. In Celsius, test `>48` before a broader `>26`, or use ordered upper thresholds. Otherwise a temperature of 50 could be incorrectly classified as merely hot.

### 3. Comparing strings and lists

Python compares strings lexicographically: first differing character determines the answer. For ordinary English letters, uppercase comes before lowercase: `"Z" < "a"`. List comparisons are also lexicographic, not elementwise:

```python
[0, 0, 0] <= [5, 50, -1]  # True: first pair decides
[1, 9] < [1, 10]          # True: first pair equal; second decides
```

NumPy comparisons are elementwise: `np.array([0, 0]) <= np.array([5, -1])` gives `[True, False]`. Do not put such a multiple-element Boolean array directly in `if`; it has no single unambiguous truth value.

### 4. Nested indexing and changing data

```python
people = [["June", "March", 12, 2006], ["Alex", "July", 4, 1999]]
print(people[1])       # whole Alex record
print(people[1][0])    # Alex
print(people[-1][1])   # July
print(len(people))     # 2 records
print(len(people[0]))  # 4 fields
```

```python
person = random.choice(people)
year = person[3]
if year < 2000:
    year = random.randint(2000, 2004)
```

Assigning a new integer to local `year` does not change the stored record. `person[3] = year` does, because `person` refers to one of the inner lists in `people`.

**Correction:** Lecture 6's solution uses `[2000, 2001, 2002, 2004]`, omitting 2003. Use `random.randint(2000, 2004)` for all five years. For the condition “before 2000,” use `<2000`; a person born in 2000 should remain unchanged.

## Lecture 7 — Loops and debugging

### 1. For loops and range

```python
for name in ["June", "Alex"]:
    print(f"Hello, {name}")
```

`name` takes each value in turn. After a nonempty loop, it remains bound to the last processed value. For matching entries in parallel lists:

```python
titles = ["A", "B", "C"]
ratings = [7.2, 8.2, 5.5]
good, okay, bad = [], [], []
for i in range(len(titles)):
    if ratings[i] >= 8:
        good.append(titles[i])
    elif ratings[i] >= 6:
        okay.append(titles[i])
    else:
        bad.append(titles[i])
```

The parallel lists must be aligned and sufficiently long.

```python
list(range(4))        # [0, 1, 2, 3]
list(range(1, 5))     # [1, 2, 3, 4]
list(range(1, 11, 2)) # [1, 3, 5, 7, 9]
```

`range` excludes stop; `range(1, n+1)` includes n. `print(range(4))` displays the range object, not a list of its values.

### 2. Accumulators and counters 累加与计数

Initialize before the loop; update inside it. Initializing inside would reset progress on every iteration.

```python
total = 0
for i in range(1, n+1):
    total = total + i
```

```python
values = [2, None, 4.5, "missing", -1]
total = 0
for i in range(len(values)):
    value = values[i]
    if (type(value) != int) and (type(value) != float):
        print("Non-numerical item at index", i)
        continue
    total = total + value
print(total)  # 5.5
```

The invalid-type test uses `and`: invalid means neither int nor float. Using `or` would incorrectly reject every value. This exact `type` check follows the lecture and is intended for Python int/float values; it is not a universal test for every numerical type.

`.append(value)` modifies a list and returns `None`. Do not assign its return to the list. `items = items + [value]` also grows a list, but creates a new list.

### 3. While loops

```python
hits = 0
num_rolls = 0
while hits < 2:
    roll = random.choice([1, 2, 3, 4, 5, 6])
    num_rolls = num_rolls + 1
    if roll == 3:
        hits = hits + 1
print(num_rolls)
```

Count every roll, including the final successful one. Count hits only when the target appears. Never reset hits on a failure when the question says “in total.” The condition is checked before entering: if already false, the body runs zero times.

For scanning a finite list until enough matches:

```python
index = 0
found = []
while len(found) < 2 and index < len(sports):
    if sports[index] == "tennis":
        found.append(index)
    index = index + 1
```

The bounds test prevents an `IndexError` when there are fewer than two matches. Update index whether or not a match was found.

### 4. Break and continue

| Keyword | Effect |
|---|---|
| `break` | Ends the loop immediately |
| `continue` | Ends current iteration only |

```python
total = 0
last_index = -1
for i in range(len(receipts)):
    total = total + receipts[i]
    last_index = i
    if total > 90:
        break
print(total, last_index)
```

This includes the receipt that makes the total exceed 90, matching quiz C. If asked to stay within the budget, test `total + receipts[i] > 90` before adding. “Exceeds” is strict `>`, so reaching exactly 90 does not stop the quiz's loop. If the list is empty, last_index remains −1; if the threshold is never crossed, the loop processes all receipts.

```python
positive_sum = 0
for value in values:
    if value <= 0:
        continue
    positive_sum = positive_sum + value
```

### 5. Tolerances and floating-point values

```python
x = 1
xs, ys = [], []
eps = 0.025
while 1/x > eps:
    xs.append(x)
    ys.append(1/x)
    x = x + 1
```

Stores x=1,…,39; exits at x=40 because 1/40 equals the threshold. It does not append the stopping point. A nonpositive threshold makes this mathematical stopping criterion unattainable for positive x.

Repeated additions of 0.1 may not equal exactly 1.0 because floating-point representation is approximate. The lecture uses `abs(t-final_time) > 1e-13` as a tolerance. The steps must actually approach within the tolerance; a tolerance cannot fix a sequence that overshoots and moves away. If a fixed number of steps is intended, an integer-counted for loop is often clearer.

```python
approximation = 0
tol = 1e-6
for k in range(1, 1_000_001):
    term = 1/k**2
    approximation = approximation + term
    if term < tol:
        break
pi_approx = np.sqrt(6*approximation)
```

This follows the official solution: add the first small term, then stop. Checking before adding excludes that term, so read the requested convention carefully. A small last term does not by itself guarantee the entire remaining tail is below tol.

### 6. Repeated plots and debugging

```python
cars = pd.read_csv("data/features.csv")
for x_name in ["displacement", "weight", "acceleration"]:
    plt.scatter(cars[x_name], cars["mpg"])
    plt.xlabel(x_name)
    plt.ylabel("mpg")
    plt.show()
```

The CSV must exist at that relative path. It was not attached here. This is the lecture's pattern for iterating over column names.

To debug: add a breakpoint, select Debug Cell, and inspect variables as execution advances. A breakpoint pauses execution before that line runs. If a later print has not been reached, no output from it appears yet. `del variable` removes a name. Notebook cells retain variables from earlier runs: rerun setup cells and reset counters/lists; a fresh-kernel top-to-bottom run catches hidden dependencies.

## Lecture 8 — Tuples, sets, arrays, views and copies

### 1. Choose the container

| Type | Syntax | Ordered/indexed? | Mutable? | Duplicates? |
|---|---|---|---|---|
| List | `[1, 2]` | Yes | Yes | Allowed |
| Tuple | `(1, 2)` | Yes | No element reassignment | Allowed |
| Set | `{1, 2}` | No positional indexing | Yes | Removed |
| NumPy array | `np.array([1, 2])` | Yes | Yes | Allowed |

Tuples may contain mutable objects, but you cannot replace a tuple element. At this course level, use tuples for fixed ordered records.

```python
(1)          # int
(1,)         # one-element tuple
1,           # also a one-element tuple
[1]          # list
{}           # dictionary, not set
set()        # empty set
(1, 2) + (3,)  # (1, 2, 3)
a, b = b, a  # swap
```

`.index(value)` works on lists and tuples; it returns the first matching position. Use it to access corresponding values in aligned sequences:

```python
index = formal_names.index("Hellenic Republic")
name = informal_names[index]
print(f"Old estimate for {name}: {pop_estimate[index]}")
pop_estimate[index] = 9.267
print(f"Current estimate for {name}: {pop_estimate[index]}")
```

Find the position by value rather than hard-coding the index.

### 2. Sets and set operations

```python
s = set(["red", "blue", "red"])
s.add("green")
if "blue" in s:
    s.remove("blue")
```

Adding a present item changes nothing. Removing an absent item with `.remove` raises `KeyError`. `s[0]` fails because there is no first position. Converting a set to a list does not give a guaranteed sorted order.

```python
a = {"Monday", "Thursday", "Friday"}
b = {"Tuesday", "Thursday", "Friday"}
a | b  # union: either set
a & b  # intersection: both sets
a - b  # difference: in a, not b
b - a  # difference: in b, not a
```

Use `|` and `&`, not `or` and `and`, for union/intersection. Set difference is directional.

Safe removal of multiple candidates:

```python
schedule.add("BIOL 301")
for course in ["MATH 250", "DATASCI 325", "MATH 346"]:
    if course in schedule:
        schedule.remove(course)
```

Loop over a separate removal list; changing a set while iterating directly over it can fail.

### 3. Count categories for a bar plot

Using `.index`:

```python
responses = ["red", "blue", "red", "green", "blue", "red"]
categories = list(set(responses))
counts = [0]*len(categories)
for response in responses:
    i = categories.index(response)
    counts[i] = counts[i] + 1
plt.bar(categories, counts)
plt.show()
```

Using `.count`, as requested by the exercise:

```python
categories = list(set(responses))
counts = []
for category in categories:
    counts.append(responses.count(category))
plt.bar(categories, counts)
plt.show()
```

Both produce red=3, blue=2, green=1, possibly in different category orders. Keep categories and counts aligned. Count raw responses, not the deduplicated category list! A bar chart suits categorical counts; a histogram usually suits distributions of numerical observations.

### 4. Array creation and slicing

```python
np.zeros(5)         # five 0.0 values
np.ones(3)          # three 1.0 values
2*np.ones(3)        # three 2.0 values
np.arange(0, 11, 2) # [0, 2, 4, 6, 8, 10]
```

Indexing starts at zero. `-1` is last, `-2` second last. Basic slicing excludes stop:

```python
temps = np.array([71, 68, 74, 80, 77, 65, 69])
temps[-1]   # 69: Saturday
temps[1:6]  # Monday–Friday
temps[1:-1] # same
temps[-2:]  # Friday and Saturday
```

### 5. Aliases, views and copies 引用、视图、复制

| Assignment | Relationship to original | Does changing an element affect original? |
|---|---|---|
| `b = a` | Same object | Yes |
| `b = a[0:2]` for NumPy | Basic slice view sharing data | Yes |
| `b = a.copy()` | Independent array | No |
| `b = a[0:2].copy()` | Independent slice copy | No |
| `b = a[0:2]; b = b + 10` | Rebind b to new result | Original unaffected by that addition |

```python
scores = np.array([90, 85, 70, 100])
backup = scores.copy()
first_two = scores[0:2]
first_two[1] = 0
print(scores)     # [90, 0, 70, 100]
print(backup)     # [90, 85, 70, 100]
print(first_two)  # [90, 0]
```

Fix: `first_two = scores[0:2].copy()`. A Python list slice creates a new outer list, whereas a basic NumPy slice is a view. Both kinds of copy are shallow in general; the examples here use plain numerical elements.

`view = view + 10` makes a new result and reassigns the name. `view += 10` changes the existing data in place, affecting the original through the view. Reassigning `original = np.arange(11)` creates a new object; an old view still refers to the old array's data.

Array swap trap:

```python
a = np.array([1, 1])
b = np.array([2, 2])
temp = a
a[0] = b[0]
b[0] = temp[0]
# a and b now both start with 2: temp shared a's data!
```

Fix with `temp = a.copy()` or `a[0], b[0] = b[0], a[0]`.

## Practice questions — attempt before reading the key

Use the standard imports above. Do not rely on values left over from earlier notebook cells.

### Q1 — Sampling and seed (like quiz A)

`students = ["June", "Alex", "Mina", "Sam", "Chris", "Taylor", "Robin", "Kai"]`

Use seed 2026 to select 4 different students for a group. Reset the seed and select another group identically. Print both. Then write a separate line to make 6 selections with repeats allowed. Explain why `random.choice(students)` cannot alone produce a list of 4 students.

### Q2 — Nested conditions (like quiz B)

A random age is chosen with `random.randint(5, 70)` and ticket type with `random.choice(["standard", "premium"])`. Set price as follows, without using `and`:

| Ticket | Age <12 | 12–59 inclusive | Age ≥60 |
|---|---:|---:|---:|
| standard | 8 | 15 | 10 |
| premium | 12 | 25 | 18 |

Print age, ticket type, and price. What are the standard prices for ages 11, 12, 59, and 60?

### Q3 — Stop after crossing a threshold (like quiz C)

`receipts = [18, 27, 15, 42, 10]`. Use a for loop to add receipts until the total exceeds 80. Include the receipt that crosses the threshold. Print total and zero-based index of the last included receipt. How would you change the code to avoid exceeding 80?

### Q4 — Repeated successes (like quiz D)

Use `random.choice` and a while loop to roll a die until you have rolled a 5 three times in total. Store total rolls in `num_rolls`. For the sequence `[5, 1, 2, 5, 4, 5]`, what should num_rolls be? Would resetting your success counter on every non-5 work?

### Q5 — Aligned tuples and a mutable list (like quiz E)

```python
codes = ("ATL", "BOS", "NYC", "SEA")
names = ("Atlanta", "Boston", "New York", "Seattle")
prices = [120, 180, 210, 260]
```

Find `"BOS"` using `.index`. Print Boston's old price using aligned indexing, change it to 195, and print the new price. Explain why you can update prices but cannot assign `names[index] = "Cambridge"`.

### Q6 — Safe set edits (like quiz F)

Write code that works for any of these starting sets:

```python
{"CS 170", "MATH 210", "ECON 201"}
{"CS 170", "HIST 100", "KRN 101"}
{"MATH 210", "ECON 201", "HIST 100"}
```

Add `"DATASCI 151"`, then remove `"MATH 210"` and `"ECON 201"` if present. Do not cause an error when either is absent.

### Q7 — List arithmetic vs array arithmetic

Predict: `[2, 4]+[1, 3]`, `np.array([2, 4])+np.array([1, 3])`, `np.array([2, 4])*2`, and `[2, 4]*2`. Explain the difference.

### Q8 — Probability and PMF plot

(a) Compute probability of exactly 7 successes in 12 trials with p=0.6. (b) Plot all possible binomial outcomes as a bar chart with labels. (c) Which function gives probability of exactly 4 failures before the third success with p=0.6? Explain n and k in that call.

### Q9 — Composition and plotting

Let `g(t)=cos(t)` and `f(x)=x**2+2*x`. Make a scatter plot of f(g(t)) at 50 equally spaced points from 0 through 2π. Which variable goes on the horizontal axis?

### Q10 — Classify parallel lists

```python
names = ["A", "B", "C", "D", "E"]
scores = [90, 75, 59, 80, 60]
```

With one for loop, put names into high (≥80), medium (60–79), or low (<60) lists. Print the lists. Explain why separate independent `if score >= 80` and `if score >= 60` statements could double-count.

### Q11 — Continue and invalid entries

`values = [4, None, -2, "missing", 3.5, 0, 7]`. Sum only positive Python int/float entries, using `continue`. Print each nonnumerical entry's index. What are the final sum and printed indices?

### Q12 — While-loop trace and fix

```python
x = 1
values = []
while 1/x > 0.2:
    values.append(x)
```

Why does this fail to terminate? Add the missing update. What are the final list and x? Is x=5 appended?

### Q13 — Indexing and slices

`a = np.array([10, 20, 30, 40, 50, 60])`. Predict `a[-2]`, `a[1:4]`, `a[-3:]`, `a[:3]`, and `a[2:2]`.

### Q14 — Views and copies

```python
a = np.array([10, 20, 30, 40])
backup = a.copy()
b = a[1:3]
b[0] = 99
b = b + 1
print(a)
print(backup)
print(b)
```

Predict all outputs. Change one line so modifying b never changes a.

### Q15 — Sets and operations

`a={"red", "blue", "green"}`, `b={"blue", "yellow"}`. Find `a|b`, `a&b`, `a-b`, and `b-a`. What types are `{}`, `set()`, `(5)`, and `(5,)`? Does `a[0]` work?

### Q16 — Category counts

`responses=["tea", "coffee", "tea", "water", "coffee", "tea"]`. Use `set`, a for loop, and `.count()` to create category and count lists, then make a bar plot. What counts should you get? Is the category order guaranteed?

### Q17 — Nested records and reassignment

```python
people = [["June", "May", 10, 2006], ["Alex", "July", 4, 1999]]
person = people[1]
year = person[3]
year = 2002
print(people[1][3])
person[3] = year
print(people[1][3])
```

Predict the two outputs and explain the difference. Write an f-string using the record to describe Alex's updated date of birth.

### Q18 — Threshold series

Using a for loop and break, compute `1/1**2 + 1/2**2 + ...`. Stop immediately after adding the first term strictly below 0.01. Print last k and total. Which k is last? How would testing before addition change the result?

---

## Answer key

### A1

```python
students = ["June", "Alex", "Mina", "Sam", "Chris", "Taylor", "Robin", "Kai"]
random.seed(2026)
first = random.sample(students, k=4)
random.seed(2026)
second = random.sample(students, k=4)
print(first)
print(second)
with_repeats = random.choices(students, k=6)
print(with_repeats)
```

The first two groups match. `choice` returns a single element. `choices` allows repeats but does not require them.

### A2

```python
age = random.randint(5, 70)
ticket = random.choice(["standard", "premium"])
if ticket == "standard":
    if age < 12:
        price = 8
    elif age < 60:
        price = 15
    else:
        price = 10
else:
    if age < 12:
        price = 12
    elif age < 60:
        price = 25
    else:
        price = 18
print(age, ticket, price)
```

Standard boundary prices: 11→8, 12→15, 59→15, 60→10.

### A3

```python
receipts = [18, 27, 15, 42, 10]
total = 0
last_index = -1
for i in range(len(receipts)):
    total = total + receipts[i]
    last_index = i
    if total > 80:
        break
print(total, last_index)  # 102, 3
```

To stay within budget:

```python
total = 0
last_index = -1
for i in range(len(receipts)):
    if total + receipts[i] > 80:
        break
    total = total + receipts[i]
    last_index = i
print(total, last_index)  # 60, 2
```

### A4

```python
num_rolls = 0
num_fives = 0
while num_fives < 3:
    roll = random.choice([1, 2, 3, 4, 5, 6])
    num_rolls = num_rolls + 1
    if roll == 5:
        num_fives = num_fives + 1
print(num_rolls)
```

Example gives 6. Resetting on failure would count consecutive successes instead.

### A5

```python
codes = ("ATL", "BOS", "NYC", "SEA")
names = ("Atlanta", "Boston", "New York", "Seattle")
prices = [120, 180, 210, 260]
index = codes.index("BOS")
print(f"Old price for {names[index]} is {prices[index]}")
prices[index] = 195
print(f"New price for {names[index]} is {prices[index]}")
```

Index is 1; old price 180, new price 195. Lists support element reassignment; tuples do not.

### A6

```python
schedule = {"CS 170", "MATH 210", "ECON 201"}  # replace to try others
schedule.add("DATASCI 151")
for course in ["MATH 210", "ECON 201"]:
    if course in schedule:
        schedule.remove(course)
print(schedule)
```

The final set depends on starting schedule; its displayed order is not guaranteed.

### A7

Outputs: `[2, 4, 1, 3]`; array `[3, 7]`; array `[4, 8]`; list `[2, 4, 2, 4]`. For lists, `+` concatenates and multiplying by an integer repeats the sequence. Arrays perform numerical operations elementwise.

### A8

```python
prob = stats.binom.pmf(7, 12, 0.6)
print(prob)
k_vals = np.arange(13)
pmf = stats.binom.pmf(k_vals, 12, 0.6)
plt.bar(k_vals, pmf)
plt.xlabel("Successes")
plt.ylabel("Probability")
plt.show()
print(stats.nbinom.pmf(4, 3, 0.6))
```

First probability ≈0.227030335488; negative-binomial probability 0.082944. In the latter, k=4 failures, n=3 required successes, p=0.6 success probability. Total trials for that event are 7.

### A9

```python
t = np.linspace(0, 2*np.pi, 50)
g = np.cos(t)
y = g**2 + 2*g
plt.scatter(t, y)
plt.xlabel("t")
plt.ylabel("f(g(t))")
plt.show()
```

Horizontal axis is t.

### A10

```python
names = ["A", "B", "C", "D", "E"]
scores = [90, 75, 59, 80, 60]
high, medium, low = [], [], []
for i in range(len(scores)):
    if scores[i] >= 80:
        high.append(names[i])
    elif scores[i] >= 60:
        medium.append(names[i])
    else:
        low.append(names[i])
print(high, medium, low)
```

High `['A', 'D']`; medium `['B', 'E']`; low `['C']`. A score of 90 satisfies both independent tests; elif prevents overlap.

### A11

```python
values = [4, None, -2, "missing", 3.5, 0, 7]
total = 0
for i in range(len(values)):
    value = values[i]
    if (type(value) != int) and (type(value) != float):
        print("Non-numerical index:", i)
        continue
    if value <= 0:
        continue
    total = total + value
print(total)
```

Sum 14.5; invalid indices 1 and 3.

### A12

```python
x = 1
values = []
while 1/x > 0.2:
    values.append(x)
    x = x + 1
print(values, x)
```

Missing update keeps x=1 forever. Corrected values are `[1, 2, 3, 4]`, final x=5. Equality at 0.2 makes the strict test false, so 5 is not appended.

### A13

40; `[20, 30, 40]`; `[40, 50, 60]`; `[10, 20, 30]`; empty array. These bracketed sequences describe array values; NumPy's printed format generally omits commas.

### A14

Outputs: a `[10, 99, 30, 40]`; backup `[10, 20, 30, 40]`; b `[100, 31]`. Element assignment first changes shared data. Later addition creates a new array for b, so that addition does not change a. Fix: `b = a[1:3].copy()`; then a and backup remain original, b ends as `[100, 31]`.

### A15

Union `{red, blue, green, yellow}`; intersection `{blue}`; a−b `{red, green}`; b−a `{yellow}` (all labels are strings). Types: dict, set, int, tuple. `a[0]` raises TypeError because sets are not subscriptable.

### A16

```python
responses = ["tea", "coffee", "tea", "water", "coffee", "tea"]
categories = list(set(responses))
counts = []
for category in categories:
    counts.append(responses.count(category))
plt.bar(categories, counts)
plt.xlabel("Drink")
plt.ylabel("Count")
plt.show()
```

Tea=3, coffee=2, water=1. Order is not guaranteed, but paired lists remain aligned.

### A17

Outputs 1999 then 2002. Reassigning year changes only that name; assigning to person[3] changes the shared inner list.

```python
print(f"{person[0]} was born {person[1]} {person[2]}, {person[3]}.")
```

Prints `Alex was born July 4, 2002.`

### A18

```python
total = 0
for k in range(1, 1001):
    term = 1/k**2
    total = total + term
    if term < 0.01:
        break
print(k, total)
```

Last k=11; total≈1.558032193976458. At k=10, term equals 0.01 so strict `<` does not stop. Testing before adding excludes k=11, leaving the sum through 10, ≈1.549767731166541.

## Final exam checklist

- Choose `choice`, `choices`, or `sample` from the requested replacement rule.
- Reset a seed before reproducing a sequence; distinguish Python and NumPy RNGs.
- Check inclusive vs exclusive endpoints in randint, range, arange, and slices.
- Initialize counters, sums, and output lists before the loop.
- Count every trial, but only successful outcomes in the success counter.
- Check whether a threshold-crossing item should be included before placing break.
- Ensure while-loop state changes and finite-list indexing stays in bounds.
- Use elif for mutually exclusive categories; test boundary values.
- Keep aligned lists aligned when using indices.
- Use set membership checks before remove; use set(), not {}, for an empty set.
- Use a copy when a basic NumPy slice should not modify the original.
- Put final output outside the loop if it should print only once.
