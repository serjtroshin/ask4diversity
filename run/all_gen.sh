model=${1:-"qwen4b"}  # default model
python_path=${2:-"python"}  # default python interpreter
bash_flag=${3:-""}  # default empty (no --bash)

echo "Running generation for model: ${model} with python path: ${python_path}"
if [ -n "${bash_flag}" ]; then
    echo "Using bash mode (no SLURM)"
fi
bash run/parallel.sh gen ${model} ${python_path} ${bash_flag}
bash run/sequential.sh gen ${model} ${python_path} ${bash_flag}
bash run/iteration.sh gen ${model} ${python_path} ${bash_flag}