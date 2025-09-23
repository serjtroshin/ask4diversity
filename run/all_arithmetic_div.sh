model=${1:-"qwen4b"}  # default model
python_path=${2:-"python"}  # default python interpreter
bash_flag=${3:-""}  # default empty (no --bash)

echo "Running arithmetic diversity for model: ${model} with python path: ${python_path}"
if [ -n "${bash_flag}" ]; then
    echo "Using bash mode (no SLURM)"
fi
bash run/parallel.sh arithmetic_div ${model} ${python_path} ${bash_flag}
bash run/sequential.sh arithmetic_div ${model} ${python_path} ${bash_flag}
bash run/iteration.sh arithmetic_div ${model} ${python_path} ${bash_flag}
