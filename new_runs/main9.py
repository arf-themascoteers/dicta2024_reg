from task_runner import TaskRunner
import os

os.chdir("..")

if __name__ == '__main__':
    tag = "spa2"
    tasks = {
        "algorithms" : ["spa2"],
        "datasets": [
            "lucas"
        ],
        "target_sizes": [8, 16, 32, 64, 128, 256, 512]
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=False)
    summary, details = ev.evaluate()
