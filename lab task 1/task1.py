def todo_list():
    tasks = []  

    print("=== To-Do List ===")
    print("Commands: add, remove, show, exit\n")

    while True:
        action = input("Command: ").strip().lower()

        if action == "add":
            task = input("Task name: ").strip()
            tasks.append(task)
            print("Added:", task, "\n")

        elif action == "remove":
            task = input("Task to remove: ").strip()
            if task in tasks:
                tasks.remove(task)
                print("Removed:", task, "\n")
            else:
                print("Not found\n")

        elif action == "show":
            if not tasks:
                print("List empty\n")
            else:
                print("Tasks:")
                for i, t in enumerate(tasks, 1):
                    print(i, "-", t)
                print()

        elif action == "exit":
            print("Bye!")
            break

        else:
            print("Wrong command\n")


todo_list()
