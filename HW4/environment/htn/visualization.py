def print_htn(tasks: dict, operators: dict) -> None:
    """
    Prints a textual representation of the HTN network to the console.
    @param tasks:
    @param operators:
    @return: None
    """
    print("\n" + "=" * 50)
    print(" " * 18 + "HTN VISUALIZATION")
    print("=" * 50)

    print("\n--- TASKS & METHODS ---")
    if not tasks:
        print("No tasks defined.")
    for task_name, task in tasks.items():
        print(f"\n[T] Task: {task_name}{task.args}")
        if not task.methods:
            print("    No methods for this task.")
        for method in task.get_methods():
            print(f"  [M] Method: {method.name}{method.args}")
            print(f"    - Precondition Template:")
            for fact in method.precondition_template:
                print(f"        - {fact}")
            print(f"    - Subtask Templates:")
            if not method.subtasks:
                print("        None")
            for subtask in method.subtasks:
                subtask_type = "Operator" if subtask[0] in operators else "Task"
                print(f"        - ({subtask_type}) {subtask[0]}{subtask[1:]}")

    print("\n--- OPERATORS ---")
    if not operators:
        print("No operators defined.")
    for op_name, op in operators.items():
        print(f"\n[O] Operator: {op_name}{op.args}")
        print(f"  - Preconditions:")
        if not op.preconditions:
            print("      None")
        for precon in op.preconditions:
            print(f"      - {precon}")

    print("\n" + "=" * 50 + "\n")
