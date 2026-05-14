import os

ORDERED_INPUT_FILES = [
    "toy.json",
    "croatia_tv_input.json",
    "germany_tv_input.json",
    "kosovo_tv_input.json",
    "netherlands_tv_input.json",
    "uk_tv_input.json",
    "usa_tv_input.json",
    "australia_iptv.json",
    "france_iptv.json",
    "spain_iptv.json",
    "uk_iptv.json",
    "us_iptv.json",
    "singapore_pw.json",
    "canada_pw.json",
    "china_pw.json",
    "youtube_gold.json",
    "youtube_premium.json",
]


def select_file(input_dir="data/input"):
    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    if not files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")

    order = {name: idx for idx, name in enumerate(ORDERED_INPUT_FILES)}
    files.sort(key=lambda name: (order.get(name, len(order)), name))

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
