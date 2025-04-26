from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "all_w_m"
    tasks = {
        "algorithms" : ["v0", "v1", "v2", "v6", "v9", "bsnet", "pcal"],
        "datasets": ["lucas_min"],
        "target_sizes" : [512, 256, 128, 64, 32, 16, 8]
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=False, verbose=False)
    summary, details = ev.evaluate()
