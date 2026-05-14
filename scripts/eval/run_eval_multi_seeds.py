import os
import lighteval
import torch
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.models.model_input import GenerationParameters
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.utils import EnvConfig
from datetime import datetime
import argparse
import json
from fsspec import url_to_fs

__version__ = f"2.0_lighteval@{lighteval.__version__}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--task", type=str, default="aime24")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--max_model_length", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--data_parallel_size", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--custom_tasks_directory", type=str, default=None)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--launcher_type", type=str, default="VLLM")
    return parser.parse_args()


def main():
    start = datetime.now()
    args = parse_args()
    fs, output_dir = url_to_fs(args.output_dir)

    max_model_length = args.max_model_length
    if args.max_model_length is None:
        print("max_model_length not set. Setting it to max_new_tokens.")
        max_model_length = args.max_new_tokens
    elif args.max_model_length == -1:
        print("max_model_length is -1. Setting it to None.")
        max_model_length = None

    folder = args.model.replace("/", "_")
    fname = f"{args.seed}-{args.temperature}-{args.top_p}-{args.task}-{args.max_new_tokens}"
    if max_model_length != args.max_new_tokens:
        fname += f"-{max_model_length}"
    if not args.use_chat_template:
        fname += "-nochat"
    fpath = os.path.join(output_dir, folder, f"{fname}.json")
    if fs.exists(fpath) and not args.overwrite:
        print(f"File {fpath} already exists. Skipping.")
        return

    system_prompt = None
    if args.system_prompt is not None and os.path.exists(args.system_prompt):
        with open(args.system_prompt, "r") as f:
            system_prompt = f.read()

    env_config = EnvConfig()

    evaluation_tracker = EvaluationTracker(
        output_dir=args.output_dir,
        save_details=True,
        push_to_hub=False,
        push_to_tensorboard=False,
        public=False,
        hub_results_org=None,
    )
    assert args.launcher_type == "VLLM", "Only VLLM is supported for now"
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        env_config=env_config,
        job_id=0,
        dataset_loading_processes=1,
        custom_tasks_directory=args.custom_tasks_directory,
        override_batch_size=-1,  # Cannot override batch size when using VLLM
        num_fewshot_seeds=1,
        max_samples=None,
        use_chat_template=args.use_chat_template,
        system_prompt=system_prompt,
        load_responses_from_details_date_id=None,
    )

    model_config = VLLMModelConfig(
        pretrained=args.model,
        dtype=args.dtype,
        seed=args.seed,
        data_parallel_size=args.data_parallel_size,
        use_chat_template=args.use_chat_template,
        max_model_length=max_model_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        generation_parameters=GenerationParameters(
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        ),
    )

    pipeline = Pipeline(
        tasks=f"custom|{args.task}|0|0",
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
        metric_options={},
    )

    pipeline.evaluate()
    pipeline.show_results()
    results = pipeline.get_results()
    pipeline.save_and_push_results()

    data = {
        "start_time": start.isoformat(),
        "end_time": datetime.now().isoformat(),
        "total_evaluation_time_seconds": (datetime.now() - start).total_seconds(),
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "task": args.task,
        "max_new_tokens": args.max_new_tokens,
        "max_model_length": max_model_length,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "seed": args.seed,
        "system_prompt": system_prompt,
        "use_chat_template": args.use_chat_template,
        "results": results["results"]["all"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "version": __version__,
        "launcher_type": args.launcher_type,
        "device_name": torch.cuda.get_device_name(),
        "lighteval_config": results["config_general"],
    }

    print(json.dumps(data, indent=2))
    fs.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    with fs.open(fpath, "w") as f:
        f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()