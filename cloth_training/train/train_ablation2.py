import subprocess

# List of script paths to execute
script_paths = [
    'train/train_transformer_CES.py',
    'train/train_transformer_MSE.py',
]

# Loop through the list of script paths and execute each one
for script_path in script_paths:
    try:
        print(f"Executing script: {script_path}")
        subprocess.run(['python', script_path], check=True)
        print(f"Script {script_path} executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error executing script {script_path}: {e}")