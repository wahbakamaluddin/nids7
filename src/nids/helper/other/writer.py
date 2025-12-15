import csv
from datetime import datetime
from pathlib import Path

class CSVWriter:
    def __init__(self, output_path) -> None:
        
        self.output_path = Path(output_path)

        date_str = datetime.now().strftime("%Y-%m-%d")
        base_name = f"flows_{date_str}"
        
        counter = 0
        while True:
            if counter == 0:
                filename = f"{base_name}.csv"
            else:
                filename = f"{base_name}_{counter}.csv"
            
            full_path = self.output_path / filename
            if not full_path.exists():
                break
            counter += 1

        self.output_path = full_path
        self.file = open(self.output_path, "w")
        self.line = 0
        self.writer = csv.writer(self.file)

    def write(self, data: dict) -> None:
        if self.line == 0:
            self.writer.writerow(data.keys())

        self.writer.writerow(data.values())
        self.file.flush()
        self.line += 1

    def __del__(self):
        self.file.close()
