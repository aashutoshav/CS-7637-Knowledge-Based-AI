
### START HELPER FUNCTIONS ###
def is_var(x):
    """
    A helper function that checks if the provided expression x is a variable,
    i.e., a string that starts with ?.

    >>> is_variable('?x')
    True
    >>> is_variable('x')
    False
    """
    return isinstance(x, str) and len(x) > 0 and x[0] == "?"

def substitute(s: dict, x: tuple):
    """
    A helper function that substitute the bindings from s into the expression x.

    >>> substitute({'?x': 'Chris', '?y': 'Dog'}, ('likes', '?x', '?y'))
    ('likes', 'Chris', 'Dog')

    >>> substitute({'?x': 'Dog', '?y': ('owner', '?x')}, ('likes', '?y', '?x'))
    ('likes', ('owner', 'Dog'), 'Dog')

    >>> substitute({'?x': 'Dog'}, '?x')
    'Dog'
    """
    if x in s:
        return substitute(s, s[x])
    elif isinstance(x, tuple):
        return tuple(substitute(s, xi) for xi in x)
    else:
        return x
    
def build_kb(grid):
    """
    Convert a 2D ARC grid into a list of facts, including:
    - ('cell', x, y, color)
    - ('adjacent', x1, y1, x2, y2) for all adjacent cell pairs
    - ('diagonal', x1, y1, x2, y2) for all diagonal cell pairs
    - ('not_equals', a, b) for all distinct values from the grid
    - ('less_than_pair', x1, y1, x2, y2) for all cell pairs to enforce ordering
    """
    kb = []
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Add cell facts
    for y in range(rows):
        for x in range(cols):
            kb.append(('cell', x, y, grid[y][x]))

    # Add adjacency and diagonal facts, with ordering
    for y1 in range(rows):
        for x1 in range(cols):
            for y2 in range(rows):
                for x2 in range(cols):
                    if (x1, y1) < (x2, y2):  # enforce order
                        kb.append(('less_than_pair', x1, y1, x2, y2))
                        if abs(x1 - x2) + abs(y1 - y2) == 1:
                            kb.append(('adjacent', x1, y1, x2, y2))
                        if abs(x1 - x2) == 1 and abs(y1 - y2) == 1:
                            kb.append(('diagonal', x1, y1, x2, y2))

    # Add not_equals for all distinct values in grid
    colors = list(set(val for row in grid for val in row))
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            kb.append(('not_equals', colors[i], colors[j]))

    return kb


def less_than_pair(x1, y1, x2, y2):
    """Returns True if (x1, y1) < (x2, y2) lexicographically."""
    return (x1, y1) < (x2, y2)


def adjacent(x1, y1, x2, y2):
    """True if (x1,y1) and (x2,y2) share a side."""
    return abs(x1 - x2) + abs(y1 - y2) == 1


def diagonal(x1, y1, x2, y2):
    """True if (x1,y1) and (x2,y2) share a corner."""
    return abs(x1 - x2) == 1 and abs(y1 - y2) == 1


def not_equals(a, b):
    """Returns True if two values are not equal."""
    return a != b

### END HELPER FUNCTIONS ###

### START STUDENT FUNCTIONS ###
def unify_var(var, val, s):
    if var in s:
        return unify(s[var], val, s)
    elif is_var(val) and val in s:
        return unify(var, s[val], s)
    else:
        s_copy = s.copy()
        s_copy[var] = val
        return s_copy


def unify(x, y, s=()):
    """
    Unify expressions x and y given a provided substitution s.  By default s is
    (), which gets recognized and replaced with an empty dictionary.  Return a
    substitution (a dict) that will make x and y equal or, if this is not
    possible, then it returns None.

    >>> unify(('likes', '?a', 'B'), ('likes', 'A', 'B'), {})
    {'?a': 'A'}

    >>> unify(('likes', '?a', 'B'), ('likes', 'A', '?b'), {})
    {'?a': 'A', '?b': 'B'}
    """
    if s == ():
        s = {}

    if s is None:
        return None
    elif x == y:
        return s
    elif is_var(x):
        return unify_var(x, y, s)
    elif is_var(y):
        return unify_var(y, x, s)
    elif isinstance(x, tuple) and isinstance(y, tuple):
        if len(x) != len(y):
            return None
        for xi, yi in zip(x, y):
            s = unify(xi, yi, s)
            if s is None:
                return None
        return s
    else:
        return None


def pattern_match(query, kb, substitution=None):
    """
    Similar to unify, but operates over multiple predictes. A query is a list
    of predicates, some of which may contain variables. A knowledge base (kb) is
    a list of predicates without any variables. Substitutions is a dictionary
    mapping variable to values.

    >>> pattern_match([('likes', '?x', 'Dog'), ('has', '?x', 'food')], [('likes', 'Chris', 'Dog'), ('likes', 'Fred', 'Dog'), ('likes', 'Elizabeth', 'Dog'), ('has', 'Chris', 'food'), ('has', 'Elizabeth', 'food')])
    [{'?x': 'Chris'}, {'?x': 'Elizabeth'}]
    """
    if substitution is None:
        substitution = {}

    if not query:
        return [substitution]

    results = []
    p = query[0]
    p_sub = substitute(substitution, p)

    for f in kb:
        s_new = unify(p_sub, f, substitution)
        if s_new is not None:
            results.extend(pattern_match(query[1:], kb, s_new))

    unique_results = []
    for r in results:
        if r not in unique_results:
            unique_results.append(r)
    return unique_results


def get_patterns(pattern_name):
    """
    Returns one of four example patterns for ARC grids.

    ARC grids are 2D puzzles in which each cell has:
      - an x-coordinate (column, starting at 0)
      - a y-coordinate (row, starting at 0)
      - a color (represented by an integer)

    Example grid:

        [[7, 0, 5],
         [7, 3, 5],
         [2, 3, 0]]

    Coordinates:

        (0,0)  (1,0)  (2,0)
          7      0      5
        (0,1)  (1,1)  (2,1)
          7      3      7
        (0,2)  (1,2)  (2,2)
          2      2      0

    Predicates representing this grid:

        # The color of each cell
        ('cell', 0, 0, 7)
        ('cell', 1, 0, 0)
        ('cell', 2, 0, 5)
        ('cell', 0, 1, 7)
        ('cell', 1, 1, 3)
        ('cell', 2, 1, 7)
        ('cell', 0, 2, 2)
        ('cell', 1, 2, 2)
        ('cell', 2, 2, 0)

    Patterns:
    1. Two cells in the same row with the same color.
    2. Two cells in the same column with color 7.
    3. Two adjacent cells with the same color.
    4. Two diagonal cells with the same color.

    Instructions:
    - Each pattern is a list of tuples using variables (start with '?').  
    - Do not hardcode cell coordinates; use variables to generalize.
    - You may use the predicates: 'cell', 'adjacent', 'diagonal', 'not_equals', 
        and 'less_than_pair', which have been provided.
    - You can also write your own predicates if necessary.
    - Cells should not match with themselves.
    - Answers should not have duplicates.
    - These patterns will be tested in the autograder against multiple grids.

    Example usage:

    >>> kb = build_kb([[7, 0, 5],
    ...                [7, 3, 7],
    ...                [2, 2, 0]])
    >>> pattern = get_patterns('same_row_same_color')
    >>> pattern_match(pattern, kb)
    [{'?x1': 0, '?y': 1, '?c': 7, '?x2': 2}, {'?x1': 0, '?y': 2, '?c': 2, '?x2': 1}]
    """
    ####### IMPLEMENT THIS FUNCTION #########
    
    patterns = {
        # 1. Two cells in the same row with the same color
        'same_row_same_color': [
            ('cell', '?x1', '?y', '?c'),
            ('cell', '?x2', '?y', '?c'),
            ('less_than_pair', '?x1', '?y', '?x2', '?y')
        ],

        # 2. Two cells in the same column with color 7
        'same_col_color_7': [
            ('cell', '?x', '?y1', 7),
            ('cell', '?x', '?y2', 7),
            ('less_than_pair', '?x', '?y1', '?x', '?y2')
        ],

        # 3. Two adjacent cells with the same color
        'adjacent_same_color': [
            ('adjacent', '?x1', '?y1', '?x2', '?y2'),
            ('cell', '?x1', '?y1', '?c'),
            ('cell', '?x2', '?y2', '?c')
        ],

        # 4. Two diagonal cells with the same color
        'diagonal_same_color': [
            ('diagonal', '?x1', '?y1', '?x2', '?y2'),
            ('cell', '?x1', '?y1', '?c'),
            ('cell', '?x2', '?y2', '?c')
        ]
    }
    
    return patterns[pattern_name]

### END STUDENT FUNCTIONS ###

if __name__ == "__main__":
    # ------------------------------
    # UNIFY EXAMPLES
    # ------------------------------
    # Students can run this after implementing 'unify' to see simple examples
    
    print("=== UNIFY EXAMPLES ===")
    
    # Example 1: variable '?x' matches 'A'
    print(unify(('Value', '?x', '8'), ('Value', 'cell1', '8'), {}))
    
    # Example 2: two variables '?a' and '?b'
    print(unify(('Value', '?a', '8'), ('Value', 'cell1', '?b'), {}))
    
    # ------------------------------
    # PATTERN_MATCH EXAMPLES
    # ------------------------------
    # Students can run this after implementing 'pattern_match'
    
    print("\n=== PATTERN_MATCH EXAMPLES ===")
    
    # Example KB
    kb_blocks = [
        ('block', 'A'),
        ('block', 'B'),
        ('block', 'C'),
        ('block', 'D'),
        ('on', 'A', 'A'),
        ('on', 'B', 'B'),
        ('on', 'A', 'B'),
    ]
    
    # Example query
    q_blocks = [
        ('block', '?x'),
        ('on', '?x', '?y')
    ]
    
    print(pattern_match(q_blocks, kb_blocks))
    
    # ------------------------------
    # GET_PATTERNS EXAMPLES
    # ------------------------------
    # Run this after implementing 'get_patterns'
    
    print("\n=== ARC GRID EXAMPLE ===")
    
    grid = [
        [7, 0, 5],
        [7, 3, 7],
        [2, 2, 0]
    ]

    kb_grid = build_kb(grid)
    
    print("Knowledge Base:", kb_grid)
    
    # Example pattern: two cells in the same row with the same color
    pattern_arc = get_patterns('same_row_same_color')
    
    print("Query:", pattern_arc)
    print("Matches:", pattern_match(pattern_arc, kb_grid))
