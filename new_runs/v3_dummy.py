from task_runner import TaskRunner
import os

os.chdir("..")

if __name__ == '__main__':
    tag = "dummy3"
    tasks = {
        "algorithms" : ["v3_dummy"],
        "datasets": [
            "lucas"
        ],
        "target_sizes" : [512, 256, 128, 64, 32, 16, 8]
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=False)
    summary, details = ev.evaluate()
