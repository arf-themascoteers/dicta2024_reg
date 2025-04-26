from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "v1"
    tasks = {
        "algorithms" : ["v1"],
        "datasets": ["lucas_min"],
        "target_sizes" : [512]
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=False, verbose=True)
    summary, details = ev.evaluate()
