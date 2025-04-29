from task_runner import TaskRunner
import os

os.chdir("..")

if __name__ == '__main__':
    tag = "v9"
    tasks = {
        "algorithms" : ["v9"],
        "datasets": [
            "lucas"
        ],
        "target_sizes" : list(range(30,4,-1))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=True)
    summary, details = ev.evaluate()
