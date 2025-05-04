from task_runner import TaskRunner
import os

os.chdir("..")

if __name__ == '__main__':
    tag = "bsdr"
    tasks = {
        "algorithms" : ["bsdr"],
        "datasets": [
            "lucas_min"
        ],
        "target_sizes" : [8]
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=True)
    summary, details = ev.evaluate()
