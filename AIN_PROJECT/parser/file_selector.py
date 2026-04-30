import os

def select_file(input_dir="data/input"):
    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    if not files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")

    print("Available files:")
    for idx, file in enumerate(files):
        print(f"{idx}: {file}")

    while True:
        try:
            choice = int(input("Select a file by index: "))
            if 0 <= choice < len(files):
                break
            else:
                print("Invalid index, try again.")
        except ValueError:
            print("Please enter a valid number.")

    return os.path.join(input_dir, files[choice])
