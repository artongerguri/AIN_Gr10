import json
import sys

from models.channel import Channel
from models.instance_data import InstanceData
from models.program import Program
from models.priority_block import PriorityBlock
from models.time_preference import TimePreference


class Parser:
    def __init__(self, file_path):
        self.file_path = file_path

    def     parse(self):
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="ignore") as file:
                data = json.load(file)

            channels = []
            unique_program_id = 1

            for ch in data.get("channels", []):
                programs = []
                for p in ch.get("programs", []):
                    # Handle duplicate program IDs by appending channel ID if needed
                    # This fixes issues with datasets like usa_tv_input.json
                    raw_id = p["program_id"]
                    unique_id_str = f"{raw_id}_{ch['channel_id']}"
                    
                    program = Program(
                        raw_id, # Keep original ID for display/logic if needed
                        p["start"],
                        p["end"],
                        p["genre"],
                        p["score"],
                        unique_id_str # Use combined ID for uniqueness
                    )
                    programs.append(program)
                    unique_program_id += 1

                channel_name = ch.get("channel_name", f"Channel_{ch['channel_id']}")
                channels.append(Channel(ch["channel_id"], channel_name, programs))

            priority_blocks = [
                PriorityBlock(
                    pb["start"],
                    pb["end"],
                    pb["allowed_channels"]
                ) for pb in data.get("priority_blocks", [])
            ]

            time_preferences = [
                TimePreference(
                    tp["start"],
                    tp["end"],
                    tp["preferred_genre"],
                    tp["bonus"]
                ) for tp in data.get("time_preferences", [])
            ]

            instance = InstanceData(
                opening_time=data["opening_time"],
                closing_time=data["closing_time"],
                min_duration=data["min_duration"],
                max_consecutive_genre=data["max_consecutive_genre"],
                channels_count=data["channels_count"],
                switch_penalty=data["switch_penalty"],
                termination_penalty=data["termination_penalty"],
                priority_blocks=priority_blocks,
                time_preferences=time_preferences,
                channels=channels
            )

            return instance

        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            sys.exit(1)
        except PermissionError:
            print(f"Permission denied when accessing: {self.file_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
            sys.exit(1)
        except KeyError as e:
            print(f"Missing required field in JSON: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)
