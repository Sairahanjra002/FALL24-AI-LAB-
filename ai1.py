# Dynamic Calculator

def dynamic_calculator():
    print("=== Dynamic Calculator ===")
    print("Supported operations: +  -  *  /  and ( ) for brackets")
    print("Type 'exit' to quit\n")

    while True:
        expr = input("Enter expression: ")

        if expr.lower() == "exit":
            print("Calculator closed.")
            break

        try:
           
            result = eval(expr, {"__builtins__": None}, {})
            print("Result =", result)
        except Exception:
            print(" Invalid expression! Please try again.\n")


if __name__ == "__main__":
    dynamic_calculator()
