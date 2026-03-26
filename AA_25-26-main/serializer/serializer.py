import json
from pathlib import Path

from models.solution import Solution


class SolutionSerializer:
    """
    Serializer of a schedule list (Schedule objects) in JSON.
    """
    # Saving input file name, algorithm name and creating output directory
    def __init__(self, input_file_path: str, algorithm_name: str):
        self.input_file_path = Path(input_file_path)
        self.algorithm_name = algorithm_name
        self.output_dir = Path("data/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def serialize(self, solution: Solution):
        # Output file name creating based on input file, algorithm name and score
        base_name = self.input_file_path.stem.replace("_input", "")
        score = int(solution.total_score)

        output_file = f"{base_name}_output_{self.algorithm_name}_{score}.json"
        output_path = self.output_dir / output_file
        """
        Takes a list of Schedule objects dhe saves as JSON.
        """
        schedules = []
        for schedule in solution.scheduled_programs:
            # every Schedule returns to dict
            schedules.append({
                "program_id": schedule.program_id,
                "channel_id": schedule.channel_id,
                "start": schedule.start,
                "end": schedule.end,
            })

        data = {
            "scheduled_programs": schedules
        }

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"[INFO] Result saved to the file: {output_path}")
        except Exception as e:
            print(f"[ERROR] Serialization failed: {e}")
