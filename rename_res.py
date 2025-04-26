import os
import shutil

source = "v5"
dest = "v4"

s = f"saved_results/{source}"
t = f"saved_results/{dest}"

os.makedirs(t, exist_ok=True)

for f in os.listdir(s):
    path = os.path.join(s, f)
    t_path = os.path.join(t,f.replace(source, dest))
    shutil.copy(path, t_path)
    with open(t_path, 'r') as file:
        data = file.read()
    data = data.replace(source, dest)
    with open(t_path, 'w') as file:
        file.write(data)