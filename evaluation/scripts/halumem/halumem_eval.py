import argparse
import copy
import json
import os
import sys
import time

from concurrent.futures import ProcessPoolExecutor, as_completed

from eval_tool import (
    evaluation_for_memory_accuracy,
    evaluation_for_memory_integrity,
    evaluation_for_question,
    evaluation_for_update_memory,
)
from tqdm import tqdm


def process_user(idx, user_data, max_workers=10):
    uuid = user_data["uuid"]
    user_name = user_data["user_name"]

    eval_results = {
        "memory_integrity_records": [],
        "memory_accuracy_records": [],
        "memory_update_records": [],
        "question_answering_records": [],
    }

    memory_integrity_inputs = []
    memory_accuracy_inputs = []
    memory_update_inputs = []
    question_answering_inputs = []

    for sid, session in enumerate(user_data["sessions"]):
        if session.get("is_generated_qa_session", False):
            continue
        if "memory_points" not in session or "extracted_memories" not in session:
            continue

        golden_memories = session["memory_points"]
        extract_memories = session["extracted_memories"]
        extract_memories_str = "\n".join(extract_memories)

        for memory in golden_memories:
            if memory["is_update"] == "True" and memory.get("memories_from_system", []):
                new_update_memory = copy.deepcopy(memory)
                new_update_memory["uuid"] = uuid
                new_update_memory["session_id"] = sid
                memory_update_inputs.append(new_update_memory)
            else:
                new_memory = copy.deepcopy(memory)
                new_memory["uuid"] = uuid
                new_memory["session_id"] = sid
                memory_integrity_inputs.append((new_memory, extract_memories_str))

        dialogue = session["dialogue"]
        dialogue_str = []
        for turn in dialogue:
            dialogue_str.append(f"[{turn['timestamp']}]{turn['role']}: {turn['content']}")
            if turn["role"] == "assistant":
                dialogue_str.append("")
        dialogue_str = "\n".join(dialogue_str)

        golden_memories_str = "\n".join(
            [m["memory_content"] for m in golden_memories if m["memory_source"] != "interference"]
        )

        for memory in extract_memories:
            new_memory = {
                "uuid": uuid,
                "session_id": sid,
                "memory_content": memory,
            }
            memory_accuracy_inputs.append((dialogue_str, golden_memories_str, new_memory))

        if "questions" in session:
            for qa in session["questions"]:
                new_qa = copy.deepcopy(qa)
                new_qa["uuid"] = uuid
                new_qa["session_id"] = sid
                question_answering_inputs.append(new_qa)

    # Memory Integrity Evaluation
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for memory, extract_memories_str in memory_integrity_inputs:
            if extract_memories_str.strip() == "":
                memory["memory_integrity_score"] = 0
                eval_results["memory_integrity_records"].append(memory)
                continue
            future = executor.submit(
                evaluation_for_memory_integrity,
                extract_memories_str,
                memory["memory_content"],
            )
            futures[future] = memory

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Memory Integrity ([{idx}]{user_name})",
        ):
            memory = futures[future]
            try:
                result = future.result()
                score = int(result.get("score"))
            except Exception:
                score = None
            memory["memory_integrity_score"] = score
            eval_results["memory_integrity_records"].append(memory)

    # Memory Accuracy Evaluation
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for dialogue_str, golden_memories_str, memory in memory_accuracy_inputs:
            future = executor.submit(
                evaluation_for_memory_accuracy,
                dialogue_str,
                golden_memories_str,
                memory["memory_content"],
            )
            futures[future] = memory

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Memory Accuracy ([{idx}]{user_name})",
        ):
            memory = futures[future]
            try:
                result = future.result()
                score = int(result.get("accuracy_score"))
                is_included = result.get("is_included_in_golden_memories", "false")
            except Exception:
                score = None
                is_included = "false"
            memory["memory_accuracy_score"] = score
            memory["is_included_in_golden_memories"] = is_included
            eval_results["memory_accuracy_records"].append(memory)

    # Memory Update Evaluation
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for update_memory in memory_update_inputs:
            future = executor.submit(
                evaluation_for_update_memory,
                "\n".join(update_memory.get("memories_from_system", [])),
                update_memory["memory_content"],
                "\n".join(update_memory.get("original_memories", [])),
            )
            futures[future] = update_memory

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Memory Update ([{idx}]{user_name})",
        ):
            update_memory = futures[future]
            try:
                result = future.result()
                update_type = result.get("evaluation_result")
            except Exception:
                update_type = None
            update_memory["memory_update_type"] = update_type
            eval_results["memory_update_records"].append(update_memory)

    # Question-Answering Evaluation
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for qa in question_answering_inputs:
            if "system_response" not in qa:
                continue
            future = executor.submit(
                evaluation_for_question,
                qa["question"],
                qa["answer"],
                "\n".join([i["memory_content"] for i in qa.get("evidence", [])]),
                qa["system_response"],
            )
            futures[future] = qa

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"QA Eval ([{idx}]{user_name})",
        ):
            qa = futures[future]
            try:
                result = future.result()
                result_type = result.get("evaluation_result")
            except Exception:
                result_type = None
            qa["result_type"] = result_type
            eval_results["question_answering_records"].append(qa)

    return eval_results


def iter_jsonl(file_path):
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(lib, version, max_workers, eval_workers):
    save_path = f"results/halumem/{lib}-{version}/"
    data_path = os.path.join(save_path, "search_results.jsonl")
    output_file = os.path.join(save_path, "eval_results.jsonl")

    if not os.path.exists(data_path):
        print(f"Error: search results not found at {data_path}")
        sys.exit(1)

    tmp_dir = os.path.join(save_path, "tmp_eval")
    os.makedirs(tmp_dir, exist_ok=True)

    start_time = time.time()

    for idx, user_data in enumerate(iter_jsonl(data_path), 1):
        uuid = user_data["uuid"]
        tmp_file = os.path.join(tmp_dir, f"{uuid}.json")

        if os.path.exists(tmp_file):
            print(f"Skipping user {uuid} ({idx}) - cached result found.")
        else:
            print(f"Processing user {uuid} ({idx})...")
            user_result = process_user(idx, user_data, max_workers=eval_workers)

            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(user_result, f, ensure_ascii=False, indent=4)

            elapsed = time.time() - start_time
            print(f"Finished user {uuid} ({idx}), elapsed {elapsed:.2f}s.")

    with open(output_file, "w", encoding="utf-8") as f_out:
        for fname in sorted(os.listdir(tmp_dir)):
            if fname.endswith(".json"):
                fpath = os.path.join(tmp_dir, fname)
                try:
                    with open(fpath, encoding="utf-8") as f_in:
                        data = json.load(f_in)
                        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Skipped {fname}: {e}")

    elapsed = time.time() - start_time
    print(f"All done in {elapsed:.2f}s. Results: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HaluMem LLM-as-Judge Evaluation")
    parser.add_argument("--lib", type=str, default="memos-api")
    parser.add_argument("--version", type=str, default="default")
    parser.add_argument("--workers", type=int, default=2, help="User-level parallelism")
    parser.add_argument("--eval_workers", type=int, default=10, help="Per-user eval parallelism")
    args = parser.parse_args()
    main(
        lib=args.lib,
        version=args.version,
        max_workers=args.workers,
        eval_workers=args.eval_workers,
    )
