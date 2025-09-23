type="parallel"
args="--name final_run_v2 --limit null --override --type ${type}"
# take model as second argument, default to qwen4b
model=${2:-"qwen4b"}  # default model
# take python_path as third argument, default to python
python_path=${3:-"python"}  # default python interpreter
# take bash flag as fourth argument (optional)
bash_flag=${4:-""}  # default empty (no --bash)

# if first arg is "gen", run_generation; if "arithmetic_div", run_arithmetic_diversity; else run_diversity
if [ "$1" == "gen" ]; then
    script="run_generation"
    ${python_path} run/run_slurm.py --script ${script} --model ${model} ${args} --python_path ${python_path} ${bash_flag}
elif [ "$1" == "arithmetic_div" ]; then
    script="run_arithmetic_diversity"
    for override_n_iterations in 3 5; do
        ${python_path} run/run_slurm.py --script ${script} --model ${model} ${args} --override_n_iterations ${override_n_iterations}  --python_path ${python_path} ${bash_flag}
    done
else
    script="run_diversity"
    for override_n_iterations in 3 5; do
        ${python_path} run/run_slurm.py --script ${script} --model ${model} ${args} --override_n_iterations ${override_n_iterations}  --python_path ${python_path} ${bash_flag}
    done
fi
