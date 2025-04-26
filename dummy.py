from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "dummy"
    tasks = {
        "algorithms" : ["v0"],
        "datasets" : ["indian_pines"],
        "target_sizes" : list(range(30,4,-1))
    }
    ev = TaskRunner(tasks,1,tag,skip_all_bands=True)
    summary, details = ev.evaluate()
