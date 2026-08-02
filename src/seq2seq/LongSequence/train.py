import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse, json
import time


from t5_continual import T5ContinualLearner

    
start_time = time.time()


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _to_scalar(value):
    if isinstance(value, dict):
        return None
    try:
        arr = np.asarray(value, dtype=float)
        return float(np.mean(arr)) if arr.size else None
    except Exception:
        try:
            return float(value)
        except Exception:
            return None


def print_results_snapshot(results_dict, task_order=None, tag="final"):
    task_order = task_order or [k for k in results_dict.keys() if k != "test"]
    test_scores = {}
    test_obj = results_dict.get("test", {})
    if isinstance(test_obj, dict):
        nested_steps = [k for k, v in test_obj.items() if isinstance(v, dict)]
        if nested_steps:
            last_step = sorted(nested_steps)[-1]
            test_obj = test_obj[last_step]
        for task, value in test_obj.items():
            scalar = _to_scalar(value)
            if scalar is not None:
                test_scores[str(task)] = scalar

    validation = {}
    for task in task_order:
        if task in results_dict and task != "test":
            history = results_dict[task]
            if isinstance(history, (list, tuple, np.ndarray)):
                vals = [_to_scalar(v) for v in history]
                vals = [v for v in vals if v is not None]
                validation[task] = {
                    "history": vals,
                    "final": vals[-1] if vals else None,
                    "best": max(vals) if vals else None,
                }
            else:
                scalar = _to_scalar(history)
                validation[task] = {
                    "history": [scalar] if scalar is not None else [],
                    "final": scalar,
                    "best": scalar,
                }

    payload = {
        "tag": tag,
        "tasks": list(task_order),
        "test": test_scores,
        "test_average": float(np.mean(list(test_scores.values()))) if test_scores else None,
        "validation": validation,
    }

    print("[SABER_RESULTS_JSON_BEGIN]", flush=True)
    print(json.dumps(_jsonable(payload), sort_keys=True), flush=True)
    print("[SABER_RESULTS_JSON_END]", flush=True)
    if test_scores:
        print("[SABER_RESULTS_TEST_TABLE]", flush=True)
        for task in task_order:
            if task in test_scores:
                print(f"{task}\t{test_scores[task]:.6f}", flush=True)
        print(f"AVERAGE\t{payload['test_average']:.6f}", flush=True)
    print("[SABER_RESULTS_VAL_TABLE]", flush=True)
    for task in task_order:
        if task in validation:
            row = validation[task]
            print(f"{task}\tfinal={row['final']:.6f}\tbest={row['best']:.6f}", flush=True)

def main(args):
    save_path = os.path.join(args.save_dir, args.save_name)
    os.makedirs(save_path, exist_ok=True)
    task_list = args.task_list

    model_name = args.model_name
    cache_dir = args.cache_dir
    continual_learner = T5ContinualLearner(model_name,
                                           cache_dir,
                                           task_list,
                                           batch_size=args.batch_size,
                                           select_k_per_class=args.select_k_per_class,
                                           pre_processed=args.pre_processed,
                                           prefix_len=args.prefix_len,
                                           freeze_weights=args.freeze_weights==1,
                                           freeze_except=args.freeze_except,
                                           lr=args.lr,
                                           seq_len=args.seq_len,

                                           prefix_MLP=args.prefix_MLP,
                                           prefix_path=args.prefix_path if args.prefix_path!='' else None,
                                           mlp_layer_norm=args.mlp_layer_norm==1,
                                           bottleneck_size=args.bottleneck_size,
                                           get_test_subset=args.get_test_subset==1,
                                           memory_perc=args.memory_perc,
                                           seed=args.seed,
                                           )
    # set selection/correlation method
    try:
        continual_learner.selection_method = args.selection_method
    except Exception:
        pass
    resume_state = None
    if args.resume_from_checkpoint:
        resume_state = continual_learner.load_training_checkpoint(args.resume_from_checkpoint)
    if args.get_test_subset==0:
        print("Not creating test subset")

    if args.multitask == 1:
        print('Multi task learning')
        results_dict = continual_learner.multi_task_training(num_epochs=args.num_epochs, save_path=save_path)
        np.save(os.path.join(save_path, 'results_dict.npy'), results_dict)
        print_results_snapshot(results_dict, task_order=task_list, tag=f"{args.save_name}:final")

    else:
        if args.num_epochs<=50:
            eval_every_N = 1
        elif args.num_epochs>50 and args.num_epochs<=200:
            eval_every_N = 5
        elif args.num_epochs>200:
            eval_every_N = 10

        results_dict = continual_learner.train_continual(continual_learner.task_list,
                                                        epochs=args.num_epochs,
                                                        save_path=save_path,
                                                        progressive=args.progressive==1,
                                                        eval_every_N=eval_every_N,
                                                        test_eval_after_every_task=args.test_eval_after_every_task==1,
                                                        data_replay_freq=args.data_replay_freq,
                                                        resume_state=resume_state,
                                                        )
        np.save(os.path.join(save_path, 'results_dict.npy'), results_dict)
        np.save(os.path.join(save_path, 'prompts.npy'), continual_learner.previous_prompts.detach().cpu().numpy())
        print_results_snapshot(results_dict, task_order=task_list, tag=f"{args.save_name}:final")
        
        print(f"Results saved to {save_path}")
        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time:.2f} seconds")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='NLP training script in PyTorch'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        help='base directory of all models / features (should not be changed)',
        default='/data/home/arazdai/T5_prompts/T5_continual/' #'/scratch/hdd001/home/anastasia/CL/'
    )

    parser.add_argument(
        '--save_name',
        type=str,
        help='folder name to save',
        required=True
    )

    parser.add_argument(
        '--task_list',
        nargs='+',
        help='List of tasks for training',
        required=True
    )

    parser.add_argument(
        '--model_name',
        type=str,
        help='Name of the model used for training',
        default="t5-base"
    )

    parser.add_argument(
        '--cache_dir',
        type=str,
        help='Name of the cache directory used for training',
        default='/'
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        help='Number of epochs to train model',
        default=5
    )

    parser.add_argument(
        '--multitask',
        type=int,
        help='Whether to perform multi-task training',
        default=0
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size',
        default=8
    )

    parser.add_argument(
        '--seq_len',
        type=int,
        help='Length of a single repeat (in #tokens)',
        default=512
    )

    parser.add_argument(
        '--prefix_len',
        type=int,
        help='Length of prompt (in #tokens)',
        default=10
    )

    parser.add_argument(
        '--prefix_path',
        type=str,
        help='path to a pre-trained progressive prefix (for superGLUE experiments)',
        default=''
    )


    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate',
        default=0.3
    )


    parser.add_argument(
        '--memory_perc',
        type=float,
        help='Memory perc',
        default=0.01
    )

    parser.add_argument(
        '--data_replay_freq',
        type=float,
        help='Replay data every X iterations',
        default=-1
    )

    parser.add_argument(
        '--select_k_per_class',
        type=int,
        help='Select k examples from each class (default -1, i.e. no changes to the original dataset)',
        default=-1
    )

    parser.add_argument(
        '--pre_processed',
        type=int,
        help='Load pre-processed dataset',
        default=0
    )

    parser.add_argument(
        '--test_eval_after_every_task',
        type=int,
        help='Whether to re-evaluate test accuracy after every task (0 - False, 1 - True)',
        default=0
    )

    parser.add_argument(
        '--progressive',
        type=int,
        help='Whether to concatenate prompts in a progressive way (0 - False, 1 - True)',
        default=1
    )

    parser.add_argument(
        '--freeze_weights',
        type=int,
        help='Whether to freeze model weigts (except word emb)',
        default=0
    )

    parser.add_argument(
        '--freeze_except',
        type=str,
        help='If freeze_weights==1, freeze all weights except those that contain this keyword',
        default='xxxxxxx' # freeze all
    )

    parser.add_argument(
        '--get_test_subset',
        type=int,
        help='Whether to create a separate test split',
        default=1
    )

    parser.add_argument(
        '--early_stopping',
        type=int,
        help='If early_stopping==1, do early stopping based on val accuracy',
        default=1 # freeze all
    )

    parser.add_argument(
        '--prefix_MLP',
        type=str,
        help='Type of MLP reparametrization (if None - use Lester original implementation)',
        default='None' # freeze all
    )

    parser.add_argument(
        '--mlp_layer_norm',
        type=int,
        help='Do layer norm in MLP',
        default=1 # use layer norm
    )

    parser.add_argument(
        '--bottleneck_size',
        type=int,
        help='MLP bottleneck size',
        default=800
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for full determinism',
        default=42
    )
    parser.add_argument(
        '--selection_method',
        type=str,
        default='proj_cos',
        choices=['proj_cos', 'wasserstein'],
        help="Criterion for selecting previous prompts."
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default='',
        help='Path to a prompt checkpoint saved under save_dir/save_name/checkpoints/latest.pt'
    )
    main(parser.parse_args())
