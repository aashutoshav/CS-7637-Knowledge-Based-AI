One of the most fundamental abilities of an intelligent agent is recognizing an instance of some category, concept, or pattern. We can state this task as:

*   **Given:** A pattern that describes some class of situations;
*   **Given:** A description of some specific situation;
*   **Find:** All ways in which the pattern matches the situation.

This homework will explore pattern matching over knowledge represented using predicate logic. To support pattern matching over this knowledge representation, you will be implementing two key functions (`unify` and `pattern_match`), and creating several patterns of your own (`get_patterns`). As we will see in activities and discussion these functions will ultimately enable several other knowledge-based AI approaches (e.g., inference, decision making, planning, etc.).

You will submit code implementing these three functions to GradeScope, which will automatically assess your submission and provide you with feedback on how it did against our hidden test cases. You can submit to GradeScope as many times as you want up until the deadline. After submitting, you must select which of your submissions you want to count for a grade prior to the deadline. Note that by default, Gradescope marks your last submission as your submission to be graded. We cannot automatically select your best submission. 

## About the Project

For this project, we will utilize a predicate logic representation of knowledge expressed in Python using tuples and strings. For example, here is how the following statements would translate into a predicate logic representation into our Python format:

*   Chris likes dogs &rarr; `('likes', 'Chris', 'dogs')`
*   Fred likes cats &rarr; `('likes', 'Fred', 'cats')`
*   Dogs like to play fetch &rarr; `('likes', 'dogs', ('play', 'fetch'))`
*   Chris has food &rarr; `('has', 'Chris', 'food')`
*   Dogs like food &rarr; `('like', 'dogs', 'food')`

Notice that in this third example, we can nest tuples within our predicates. For this homework, you can ignore negation, quantifiers (there exists and for all), and disjunction.

We will, however, utilize conjunction in this assignment. In this case, a conjunction of predicate logic expressions will be represented as a list. For example, a knowledge base with the example expressions from above would be represented in Python as:
```python
[
    ('likes', 'Chris', 'dogs'),
    ('likes', 'Fred', 'dogs'),
    ('likes', 'dogs', ('play', 'fetch')),
    ('has', 'Chris', 'food'),
    ('like', 'dogs', 'food')
]
```

In this assignment we will implement tools for evaluating relational patterns against knowledge represented in this format. Patterns will look similar to the predicates above, but will also support the use of pattern matching variables, which can be bound to different values.

For example, I might express a pattern to identify all the people that like dogs as `('likes', '?x', 'dogs')`. In this pattern, I've denoted the pattern matching variable as a string that starts with a question mark. If we matched this pattern against the knowledge base shown above, we would find two possible matches of the pattern: one where `?x` binds to `Chris` and another where it binds to `Fred`. 

Patterns can also utilize conjunction to increase their expressiveness. For example, we know that dogs like food, so maybe we want to find people that both like dogs and have food (maybe dogs will like them more because they have food!). To do this, we might construct the following conjunctive query: `[('likes', '?x', 'dogs'), ('has', '?x', 'food')]`. Given the additional requirement that `?x` has food, now there is only one valid binding that maps `?x` to `Chris`. 

For this assignment, you will implement your own Python pattern matcher. You will start by implementing a function called `unify(x, y, s)`, which takes two expressions to match (`x` and `y`) and a dictionary `s` containing mappings from variables to their bindings. If `x` and `y` can be unified, then this function will return a dictionary containing a substitution that will make `x` and `y` equivalent (if no variable bindings are needed then this will be an empty dictionary). If there is no way to make `x` and `y` equal, then this function will return `None` (the special Python object). If `s` is not provided when calling this function, then the function will assume that `s` is an empty binding. 

For example, the function call `unify(('likes', 'Chris', 'dogs'), ('likes', '?y', '?z'))` will output `{'?y': 'Chris', '?z': 'dogs'}`. 

We recommend that you implement this function first and get it to pass all of GradeScope's test cases.

Once you implement `unify`, you should then move on to implementing `pattern_match(query, kb, substitution)`. This function is similar to `unify`, but the `query` and `kb` (the knowledge base) are lists representing conjunctions of predicates. `substitution` is a dictionary mapping the variables to their bindings. If there is no way to satisfy the query with the given knowledge base, then this function will return an empty list. Otherwise, it will return a list of all the possible matches (each match will be represented by a dictionary mapping variables to bindings, or an empty dictionary if there are no variable mappings).

Here are two example inputs and outputs to this function based on the example above:

*   `pattern_match([('likes', '?x', 'dogs')], [('likes', 'Chris', 'dogs'), ('likes', 'Fred', 'dogs'), ('likes', 'dogs', ('play', 'fetch')), ('has', 'Chris', 'food'), ('like', 'dogs', 'food')])` will return `[{'?x': 'Chris'}, {'?x': 'Fred'}]`
*   `pattern_match([('likes', '?x', 'dogs'), ('has', '?x', 'food')], [('likes', 'Chris', 'dogs'), ('likes', 'Fred', 'dogs'), ('likes', 'dogs', ('play', 'fetch')), ('has', 'Chris', 'food'), ('like', 'dogs', 'food')])` will return `[{'?x': 'Chris'}]`

Finally, you will implement `get_patterns(pattern_name)` to create your own patterns. We will use your implementations of `unify` and `pattern_match` to test these patterns, so please complete this portion last. You will work with ARC grids to identify patterns between cells.

## Starter Code

To get started please download the starter code file: [`pattern_match.py`](file:///Users/aashutosh/Documents/Academics%20-%20GaTech/Fall%202026/KBAI/HW1/pattern_match.py) (Download `pattern_match.py`).

Within this code, there are three functions for you to implement (`unify`, `pattern_match`, and `get_patterns`). Note, there are also a few helper functions that you might use to support your implementations. 

As mentioned before, we recommend you start by implementing `unify`, test it with GradeScope, then implement `pattern_match` using your `unify` function, and finally implement `get_patterns`.

We also recommend that before you submit your implementation to GradeScope you also create your own set of test cases to evaluate your implementations of `unify`, `pattern_match`, and `get_patterns` (as far as workflows go, submitting to GradeScope and then using the feedback to iterate on your implementation might be a little slow). 

Chapter 9 (Inference in First-Order Logic) of *Artificial Intelligence: A Modern Approach* by Russell and Norvig might be a useful reference for learning more about unification. You may also want to reference the interactive ARC puzzles [https://tail.cc.gatech.edu/kbai/arc-prize/](https://tail.cc.gatech.edu/kbai/arc-prize/) to get familiar with this format.

A couple of additional things you do **NOT** need to do in this assignment:

*   You do **NOT** need to implement disjunction.
*   You do **NOT** need to implement negation.
*   You do **NOT** need to implement standardizing apart.
*   You do **NOT** need to implement an occurs check. 
*   You do **NOT** need to implement any kind of inference or forward chaining where new knowledge is added to the knowledge base.

## Submitting Your Solution

To submit your code, go to the course in Canvas and click **Gradescope** on the left side. Then, select **CS 4635/7637** if need be.

You will see an assignment named **Homework 1**. Select this project, then drag your `pattern_match.py` file into the autograder. For this project, please put your entire implementation in this one file. If you submit other files, or rename `pattern_match.py` to something else, then it will not run correctly.

When your submission is done running, you’ll see how well your solution did against the hidden test cases.

## How You Will Be Graded

Your agent will be run against several test cases. Approximately 1/3 of these test cases will evaluate the unification, 1/3 will evaluate the pattern matcher, and 1/3 will evaluate the patterns you wrote. The names of the tests will give you a little insight into what is being tested, but for some tests, the actual evaluations will be hidden. 

Each test case will be worth between 2 and 6 points if you pass it and 0 if you fail it. Your grade on the assignment is the sum of the points you accrue, adding up to a possible 100/100 points.

You may submit solutions as many times as you want prior to the deadline. You must select which of your submissions you want to count for a grade prior to the deadline. Note that by default, Gradescope marks your last submission as your submission to be graded. We cannot automatically select your best submission. 

## Libraries Allowed

You do not need to make use of any external libraries for this project (none will be available on the autograder instance).

Additionally, we use Python 3 for our autograder.