from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "v6"
    tasks = {
        #"algorithms" : ["v0", "v1", "v2", "v6", "v9", "bsnet", "pcal"],
        "algorithms" : ["v6"],
        "datasets": ["lucas"],
        "target_sizes" : [512, 256, 128, 64, 32, 16, 8]
        #"target_sizes" : [32]
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=True)
    summary, details = ev.evaluate()
