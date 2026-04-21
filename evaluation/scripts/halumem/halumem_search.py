import argparse
import copy
import json
import os
import sys
import time
import traceback

from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llms import llm_request
from prompts import PROMPT_MEMOS
from utils.client import MemosApiClient


load_dotenv()

MEMOS_CONTEXT_TEMPLATE = """Memories for user {user_id}:

    {memories}
"""


def memos_search(client, query, user_id, top_k):
    start = time.time()
    results = client.search(query=query, user_id=user_id, top_k=top_k)
    context = (
        "\n".join([i["memory"] for i in results["text_mem"][0]["memories"]])
        + f"\n{results.get('pref_string', '')}"
    )
    memories = [i["memory"] for i in results["text_mem"][0]["memories"]]
    pref_memories = []
    if results.get("pref_mem"):
        pref_memories = [i["memory"] for i in results["pref_mem"][0].get("memories", [])]
    context = MEMOS_CONTEXT_TEMPLATE.format(user_id=user_id, memories=context)
    duration_ms = (time.time() - start) * 1000
    return context, memories + pref_memories, duration_ms


def process_user(user_data, top_k, pref_top_k, save_path):
    client = MemosApiClient()
    user_name = user_data["user_name"]
    uuid = user_data["uuid"]

    tmp_dir = os.path.join(save_path, "tmp_search")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{uuid}.json")

    if os.path.exists(tmp_file):
        print(f"Skipping user {user_name} - cached result found.")
        return {"uuid": uuid, "status": "cached", "path": tmp_file}

    try:
        for session in tqdm(user_data["sessions"], desc=f"Searching {user_name}", leave=False):
            if session.get("is_generated_qa_session", False):
                continue
            if "memory_points" not in session:
                continue

            for memory in session["memory_points"]:
                if memory["is_update"] == "False" or not memory.get("original_memories"):
                    continue
                _, memories_from_system, dur = memos_search(
                    client, memory["memory_content"], user_name, top_k=10
                )
                memory["memories_from_system"] = memories_from_system

            if "questions" not in session:
                continue

            new_questions = []
            for qa in session["questions"]:
                context, _, duration_ms = memos_search(
                    client, qa["question"], user_name, top_k=top_k
                )
                new_qa = copy.deepcopy(qa)
                new_qa["context"] = context
                new_qa["search_duration_ms"] = duration_ms

                prompt = PROMPT_MEMOS.format(context=context, question=qa["question"])
                start_time = time.time()
                response = llm_request(prompt)
                new_qa["system_response"] = response
                new_qa["response_duration_ms"] = (time.time() - start_time) * 1000

                new_questions.append(new_qa)

            session["questions"] = new_questions

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(user_data, f, ensure_ascii=False)

        print(f"Saved search results for {user_name}")
        return {"uuid": uuid, "status": "ok", "path": tmp_file}

    except Exception as e:
        error_path = os.path.join(tmp_dir, f"{uuid}_error.log")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"Error in user {user_name}: {e}")
        return {"uuid": uuid, "status": "error", "path": error_path}


def iter_jsonl(file_path):
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(lib, version, top_k, pref_top_k, max_workers):
    save_path = f"results/halumem/{lib}-{version}/"
    ingestion_file = os.path.join(save_path, "ingestion_results.jsonl")
    output_file = os.path.join(save_path, "search_results.jsonl")
    tmp_dir = os.path.join(save_path, "tmp_search")
    os.makedirs(tmp_dir, exist_ok=True)

    if not os.path.exists(ingestion_file):
        print(f"Error: ingestion results not found at {ingestion_file}")
        sys.exit(1)

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for _idx, user_data in enumerate(iter_jsonl(ingestion_file), 1):
            future = executor.submit(process_user, user_data, top_k, pref_top_k, save_path)
            futures[future] = user_data["uuid"]
        total_users = _idx

        for i, future in enumerate(as_completed(futures), 1):
            uid = futures[future]
            try:
                result = future.result()
                print(f"[{i}/{total_users}] Finished {uid} ({result['status']})")
            except Exception as e:
                print(f"[{i}/{total_users}] Error processing {uid}: {e}")
                traceback.print_exc()

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
    parser = argparse.ArgumentParser(description="HaluMem Search & QA via memos-api")
    parser.add_argument("--lib", type=str, default="memos-api")
    parser.add_argument("--version", type=str, default="default")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--pref_top_k", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    main(
        lib=args.lib,
        version=args.version,
        top_k=args.top_k,
        pref_top_k=args.pref_top_k,
        max_workers=args.workers,
    )
