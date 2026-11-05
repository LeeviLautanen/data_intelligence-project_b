import subprocess
import sys


def run(*args):
    subprocess.run([sys.executable, *args], cwd="PsySET", check=True)


run(
    "generate_steer_vecs.py",
    "--model_index=13",
    "--use_quantization",
    "--dataset_source=goemotions_instruct_full_binary",
    "--target_tokens=all",
)

run("generate_responses.py", "--name=baseline")
run(
    "generate_responses.py",
    "--name=layers_18_19_c5",
    "--steer_layers=18,19",
    "--steer_coeff=5",
)
run(
    "generate_responses.py",
    "--name=layers_9_10_c15",
    "--steer_layers=9,10",
    "--steer_coeff=15",
)
run(
    "generate_responses.py",
    "--name=layers_18_19_c15",
    "--steer_layers=18,19",
    "--steer_coeff=15",
)
run(
    "generate_responses.py",
    "--name=layers_27_28_c15",
    "--steer_layers=27,28",
    "--steer_coeff=15",
)

run("evaluate_responses.py")
