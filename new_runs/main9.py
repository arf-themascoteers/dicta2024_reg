from task_runner import TaskRunner
import os

os.chdir("..")

if __name__ == '__main__':
    tag = "mc2"
    tasks = {
        "algorithms" : ["bsnet","v0","v9"],
        "datasets": [
            "lucas_min"
        ],
        "target_sizes" : [30]
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=False)
    summary, details = ev.evaluate()
