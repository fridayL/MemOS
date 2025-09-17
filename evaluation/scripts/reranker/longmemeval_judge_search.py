#!/usr/bin/env python3
"""
LongMemEval Search Results Evaluation Script (Threading Version)

This script evaluates search results by using OpenAI's model to judge whether
retrieved memories contain the correct answer, then calculates various metrics
including top-N accuracy, recall, MAP, and NDCG.

Features:
- Multi-threaded API requests for faster evaluation
- Maintains original order of results
- Rate limiting and error handling
- Progress tracking
"""

import json
import logging
import os
import time
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from openai import OpenAI
import argparse
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('longmemeval_judge.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_CACHED_DIR = "./eval_cache_data"
BASE_EVAL_DIR = "./eval_result"


@dataclass
class JudgmentTask:
    """Represents a single judgment task with its index for ordering."""
    index: int
    question: str
    answer: str
    memory: str
    search_index: int  # Index of the search this belongs to
    memory_index: int  # Index of the memory within the search


class ThreadedSearchResultsEvaluator:
    """Evaluates search results using OpenAI model for relevance judgment with threading."""
    
    def __init__(self, 
                 openai_api_key: str = None, 
                 model: str = "gpt-4o-mini",
                 max_workers: int = 10,
                 rate_limit_delay: float = 0.1,
                 base_url: str = None):
        """
        Initialize the evaluator.
        
        Args:
            openai_api_key: OpenAI API key. If None, will try to get from environment.
            model: OpenAI model to use for evaluation.
            max_workers: Maximum number of worker threads.
            rate_limit_delay: Delay between requests to avoid rate limiting.
            base_url: OpenAI API base URL.
        """
        self.client = OpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),  
            base_url=base_url or os.getenv("OPENAI_API_BASE")
        )
        self.model = model
        self.judgment_cache = {}
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.cache_lock = threading.Lock()  # 保护缓存的线程锁
        
    def judge_relevance(self, question: str, answer: str, memory: str) -> bool:
        """
        Judge whether a memory is relevant to answering a question.
        
        Args:
            question: The question being asked
            answer: The expected answer
            memory: The memory content to evaluate
            
        Returns:
            bool: True if memory is relevant, False otherwise
        """
        cache_key = f"{question}|{answer}|{memory}"
        
        # 检查缓存（线程安全）
        with self.cache_lock:
            if cache_key in self.judgment_cache:
                return self.judgment_cache[cache_key]
            
        prompt = f"""You are evaluating whether a memory contains information that would help answer a specific question.

Question: {question}
Expected Answer: {answer}
Memory Content: {memory}

Your task is to determine if the memory content contains information that supports or leads to the expected answer. The memory doesn't need to contain the exact answer, but should contain relevant information that would help someone arrive at the correct answer.

Respond with only "true" or "false" (lowercase, no quotes).

Examples:
- If the question is "What degree did I graduate with?" and the answer is "Business Administration", a memory about "completed my Business Administration degree" would be true.
- If the question is "What is my cat's name?" and the answer is "Whiskers", a memory about "my cat Whiskers loves to play" would be true.
- If the question is "Where did I go to college?" and the answer is "Harvard", a memory about "my grocery shopping list" would be false.

Response:"""

        try:
            # 添加速率限制延迟
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            
            judgment = response.choices[0].message.content.strip().lower()
            result = False
            if "true" in judgment:
                result = True
            else:
                result = False
            
            # 更新缓存（线程安全）
            with self.cache_lock:
                self.judgment_cache[cache_key] = result
            
            logger.info(f"Question: {question[:50]}... | Answer: {answer} | Judgment: {result}| orignal_judge: {judgment}")
            return result
            
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {e}")
            # 重试一次
            try:
                time.sleep(1.0)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=10
                )
                
                judgment = response.choices[0].message.content.strip().lower()
                result = judgment == "true"
                
                # 更新缓存（线程安全）
                with self.cache_lock:
                    self.judgment_cache[cache_key] = result
                return result
            except Exception as retry_e:
                logger.error(f"Retry failed for judgment: {retry_e}")
                return False

    def process_judgment_task(self, task: JudgmentTask) -> Tuple[int, int, int, bool]:
        """
        Process a single judgment task.
        
        Args:
            task: JudgmentTask to process
            
        Returns:
            Tuple of (index, search_index, memory_index, judgment_result)
        """
        result = self.judge_relevance(task.question, task.answer, task.memory)
        return (task.index, task.search_index, task.memory_index, result)

    def evaluate_all_searches_threaded(self, successful_searches: List[Dict], return_raw_results: bool = False) -> List[Dict]:
        """
        Evaluate all searches using multiple threads while maintaining order.
        
        Args:
            successful_searches: List of search data dictionaries
            
        Returns:
            List of evaluation results in original order
        """
        # 创建所有判断任务
        tasks = []
        task_index = 0
        
        for search_idx, search_data in enumerate(successful_searches):
            # 提取问题和答案
            question = search_data.get("question", search_data.get("query", ""))
            answer = search_data.get("answer", search_data.get("golden_answer", ""))
            
            # 尝试不同的路径来获取memories
            memories = []
            
            # 根据实际JSON结构获取memories
            try:
                memories = search_data["search_result"]["data"]["text_mem"][0]["memories"]
                logger.debug(f"Found memories: {len(memories)} items")
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Failed to access memories for search {search_idx}: {e}")
                logger.error(f"Available keys in search_data: {list(search_data.keys())}")
                if "search_result" in search_data:
                    logger.error(f"Available keys in search_result: {list(search_data['search_result'].keys())}")
                memories = []
            
            if not memories:
                logger.warning(f"No memories found for search {search_idx}, available keys: {list(search_data.keys())}")
                continue
            
            for memory_idx, memory_item in enumerate(memories):
                # 从JSON结构看，每个memory_item都有"memory"字段
                memory_content = memory_item.get("memory", "")
                
                if not memory_content:
                    logger.warning(f"Empty memory content at search {search_idx}, memory {memory_idx}")
                    logger.warning(f"Memory item keys: {list(memory_item.keys()) if isinstance(memory_item, dict) else type(memory_item)}")
                    continue
                task = JudgmentTask(
                    index=task_index,
                    question=question,
                    answer=answer,
                    memory=memory_content,
                    search_index=search_idx,
                    memory_index=memory_idx
                )
                tasks.append(task)
                task_index += 1
        
        logger.info(f"Created {len(tasks)} judgment tasks across {len(successful_searches)} searches")
        
        # 使用线程池处理所有任务
        logger.info("Processing judgment tasks with threading...")
        results = []
        
        # 使用 ThreadPoolExecutor 进行多线程处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(self.process_judgment_task, task): task for task in tasks}
            
            # 使用 tqdm 显示进度条
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing judgments"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    # 如果任务失败，添加一个默认结果
                    task = future_to_task[future]
                    results.append((task.index, task.search_index, task.memory_index, False))
        
        # 按原始索引排序以保持顺序
        results.sort(key=lambda x: x[0])
        
        # 按搜索分组并构建详细的评估结果
        evaluation_results = []
        current_search_idx = -1
        current_memories_with_judgments = []
        
        for _, search_idx, memory_idx, judgment in results:
            if search_idx != current_search_idx:
                # 保存之前的搜索结果（如果有的话）
                if current_search_idx >= 0:
                    search_data = successful_searches[current_search_idx]
                    relevance_scores = [item["judgment"] for item in current_memories_with_judgments]
                    
                    # 计算各个top-N的hit rate
                    top_n_hit_rates = {}
                    for n in [1, 3, 5, 10, 20]:
                        top_n_scores = relevance_scores[:n] if len(relevance_scores) >= n else relevance_scores
                        top_n_hit_rates[f"hit@{n}"] = 1 if sum(top_n_scores) > 0 else 0
                    
                    evaluation_results.append({
                        "question_id": search_data.get("question_id", f"q_{current_search_idx}"),
                        "question_type": search_data.get("question_type", "unknown"),
                        "question": search_data["question"],
                        "answer": search_data["answer"],
                        "question_date": search_data.get("question_date", ""),
                        "original_search_result": search_data.get("search_result", {}),
                        "memories_with_judgments": current_memories_with_judgments,
                        "relevance_scores": relevance_scores,
                        "total_memories": len(relevance_scores),
                        "relevant_memories": sum(relevance_scores),
                        "top_n_hit_rates": top_n_hit_rates,
                        "overall_hit_rate": 1 if sum(relevance_scores) > 0 else 0
                    })
                
                # 开始新的搜索
                current_search_idx = search_idx
                current_memories_with_judgments = []
            
            # 获取对应的memory内容
            search_data = successful_searches[search_idx]
            memories = search_data["search_result"]["data"]["text_mem"][0]["memories"]
            memory_item = memories[memory_idx]
            
            # 添加详细的判断结果到当前搜索
            current_memories_with_judgments.append({
                "memory_index": memory_idx,
                "memory_id": memory_item.get("id", ""),
                "memory_content": memory_item.get("memory", ""),
                "memory_metadata": memory_item.get("metadata", {}),
                "judgment": 1 if judgment else 0,
                "judgment_bool": judgment
            })
        
        # 不要忘记最后一个搜索
        if current_search_idx >= 0:
            search_data = successful_searches[current_search_idx]
            relevance_scores = [item["judgment"] for item in current_memories_with_judgments]
            
            # 计算各个top-N的hit rate
            top_n_hit_rates = {}
            for n in [1, 3, 5, 10, 20]:
                top_n_scores = relevance_scores[:n] if len(relevance_scores) >= n else relevance_scores
                top_n_hit_rates[f"hit@{n}"] = 1 if sum(top_n_scores) > 0 else 0
            
            evaluation_results.append({
                "question_id": search_data.get("question_id", f"q_{current_search_idx}"),
                "question_type": search_data.get("question_type", "unknown"),
                "question": search_data["question"],
                "answer": search_data["answer"],
                "question_date": search_data.get("question_date", ""),
                "original_search_result": search_data.get("search_result", {}),
                "memories_with_judgments": current_memories_with_judgments,
                "relevance_scores": relevance_scores,
                "total_memories": len(relevance_scores),
                "relevant_memories": sum(relevance_scores),
                "top_n_hit_rates": top_n_hit_rates,
                "overall_hit_rate": 1 if sum(relevance_scores) > 0 else 0
            })
        
        if return_raw_results:
            return evaluation_results, results
        else:
            return evaluation_results

    def _build_evaluation_results_from_tuples(self, successful_searches: List[Dict], results: List[Tuple]) -> List[Dict]:
        """
        从results元组构建评估结果，用于从中间结果加载时使用。
        """
        # 按搜索分组并构建详细的评估结果
        evaluation_results = []
        current_search_idx = -1
        current_memories_with_judgments = []
        
        for _, search_idx, memory_idx, judgment in results:
            if search_idx != current_search_idx:
                # 保存之前的搜索结果（如果有的话）
                if current_search_idx >= 0:
                    search_data = successful_searches[current_search_idx]
                    relevance_scores = [item["judgment"] for item in current_memories_with_judgments]
                    
                    # 计算各个top-N的hit rate
                    top_n_hit_rates = {}
                    for n in [1, 3, 5, 10, 20]:
                        top_n_scores = relevance_scores[:n] if len(relevance_scores) >= n else relevance_scores
                        top_n_hit_rates[f"hit@{n}"] = 1 if sum(top_n_scores) > 0 else 0
                    
                    evaluation_results.append({
                        "question_id": search_data.get("question_id", f"q_{current_search_idx}"),
                        "question_type": search_data.get("question_type", "unknown"),
                        "question": search_data["question"],
                        "answer": search_data["answer"],
                        "question_date": search_data.get("question_date", ""),
                        "original_search_result": search_data.get("search_result", {}),
                        "memories_with_judgments": current_memories_with_judgments,
                        "relevance_scores": relevance_scores,
                        "total_memories": len(relevance_scores),
                        "relevant_memories": sum(relevance_scores),
                        "top_n_hit_rates": top_n_hit_rates,
                        "overall_hit_rate": 1 if sum(relevance_scores) > 0 else 0
                    })
                
                # 开始新的搜索
                current_search_idx = search_idx
                current_memories_with_judgments = []
            
            # 获取对应的memory内容
            search_data = successful_searches[search_idx]
            memories = search_data["search_result"]["data"]["text_mem"][0]["memories"]
            memory_item = memories[memory_idx]
            
            # 添加详细的判断结果到当前搜索
            current_memories_with_judgments.append({
                "memory_index": memory_idx,
                "memory_id": memory_item.get("id", ""),
                "memory_content": memory_item.get("memory", ""),
                "memory_metadata": memory_item.get("metadata", {}),
                "judgment": 1 if judgment else 0,
                "judgment_bool": judgment
            })
        
        # 不要忘记最后一个搜索
        if current_search_idx >= 0:
            search_data = successful_searches[current_search_idx]
            relevance_scores = [item["judgment"] for item in current_memories_with_judgments]
            
            # 计算各个top-N的hit rate
            top_n_hit_rates = {}
            for n in [1, 3, 5, 10, 20]:
                top_n_scores = relevance_scores[:n] if len(relevance_scores) >= n else relevance_scores
                top_n_hit_rates[f"hit@{n}"] = 1 if sum(top_n_scores) > 0 else 0
            
            evaluation_results.append({
                "question_id": search_data.get("question_id", f"q_{current_search_idx}"),
                "question_type": search_data.get("question_type", "unknown"),
                "question": search_data["question"],
                "answer": search_data["answer"],
                "question_date": search_data.get("question_date", ""),
                "original_search_result": search_data.get("search_result", {}),
                "memories_with_judgments": current_memories_with_judgments,
                "relevance_scores": relevance_scores,
                "total_memories": len(relevance_scores),
                "relevant_memories": sum(relevance_scores),
                "top_n_hit_rates": top_n_hit_rates,
                "overall_hit_rate": 1 if sum(relevance_scores) > 0 else 0
            })
        
        return evaluation_results

    def calculate_metrics(self, evaluation_results: List[Dict]) -> Dict:
        """
        Calculate various evaluation metrics.
        
        Args:
            evaluation_results: List of evaluation results
            
        Returns:
            Dictionary containing calculated metrics
        """
        metrics = {}
        
        # Calculate top-N accuracy, recall, and hit rate for different N values
        n_values = [1, 3, 5, 10, 20]
        
        for n in n_values:
            correct_at_n = 0  # 在top-N中至少有一个相关结果的查询数
            hit_at_n = 0      # 在top-N中包含正确答案的查询数（新指标）
            total_relevant_at_n = 0
            total_queries = len(evaluation_results)
            
            for result in evaluation_results:
                relevance_scores = result["relevance_scores"]
                top_n_scores = relevance_scores[:n] if len(relevance_scores) >= n else relevance_scores
                
                # Accuracy: whether there's at least one relevant result in top-N
                if sum(top_n_scores) > 0:
                    correct_at_n += 1
                
                # Hit Rate: 如果top-N中有任何相关结果，则认为召回了正确答案
                # 这里的逻辑是：只要top-N中有相关内容，就认为能够回答问题
                if sum(top_n_scores) > 0:
                    hit_at_n += 1
                
                # For recall calculation
                total_relevant_at_n += sum(top_n_scores)
            
            # Top-N accuracy (与hit rate相同，但保留原有命名)
            accuracy_at_n = correct_at_n / total_queries if total_queries > 0 else 0
            metrics[f"accuracy@{n}"] = accuracy_at_n
            
            # Hit Rate: 在top-N中包含正确答案的比例
            hit_rate_at_n = hit_at_n / total_queries if total_queries > 0 else 0
            metrics[f"hit_rate@{n}"] = hit_rate_at_n
            
            # Recall Rate: 如果top-N中有相关结果，则召回率为100%，否则为0%
            recall_at_n = hit_at_n / total_queries if total_queries > 0 else 0
            metrics[f"recall@{n}"] = recall_at_n
            
            # Average relevant items in top-N (原有指标)
            avg_relevant_at_n = total_relevant_at_n / total_queries if total_queries > 0 else 0
            metrics[f"avg_relevant@{n}"] = avg_relevant_at_n
        
        # Calculate MAP (Mean Average Precision)
        map_scores = []
        for result in evaluation_results:
            relevance_scores = result["relevance_scores"]
            if sum(relevance_scores) == 0:  # No relevant documents
                map_scores.append(0.0)
                continue
                
            precision_at_k = []
            relevant_count = 0
            
            for k, score in enumerate(relevance_scores, 1):
                if score == 1:
                    relevant_count += 1
                    precision_at_k.append(relevant_count / k)
            
            if precision_at_k:
                map_scores.append(sum(precision_at_k) / len(precision_at_k))
            else:
                map_scores.append(0.0)
        
        metrics["MAP"] = sum(map_scores) / len(map_scores) if map_scores else 0
        
        # Calculate NDCG (Normalized Discounted Cumulative Gain)
        ndcg_scores = []
        for result in evaluation_results:
            relevance_scores = result["relevance_scores"]
            
            # DCG calculation
            dcg = 0
            for i, score in enumerate(relevance_scores):
                if i == 0:
                    dcg += score
                else:
                    dcg += score / np.log2(i + 1)
            
            # IDCG calculation (ideal ranking)
            ideal_scores = sorted(relevance_scores, reverse=True)
            idcg = 0
            for i, score in enumerate(ideal_scores):
                if i == 0:
                    idcg += score
                else:
                    idcg += score / np.log2(i + 1)
            
            # NDCG
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            else:
                ndcg_scores.append(0.0)
        
        metrics["NDCG"] = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
        
        # Additional statistics
        total_queries = len(evaluation_results)
        total_memories = sum(result["total_memories"] for result in evaluation_results)
        total_relevant = sum(result["relevant_memories"] for result in evaluation_results)
        
        metrics["total_queries"] = total_queries
        metrics["avg_memories_per_query"] = total_memories / total_queries if total_queries > 0 else 0
        metrics["avg_relevant_per_query"] = total_relevant / total_queries if total_queries > 0 else 0
        metrics["overall_relevance_rate"] = total_relevant / total_memories if total_memories > 0 else 0
        
        return metrics
    
    def evaluate_file(self, file_path: str, intermediate_file: str = None, save_intermediate: bool = False) -> Tuple[List[Dict], Dict]:
        """
        Evaluate all search results in a file using threading.
        
        Args:
            file_path: Path to the JSON file containing search results
            intermediate_file: Path to intermediate judgment results (if exists, skip GPT requests)
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Tuple of (evaluation_results, metrics)
        """
        logger.info(f"Loading search results from: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 调试：打印文件结构
        logger.info(f"File keys: {list(data.keys())}")
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    logger.info(f"Key '{key}' contains {len(value)} items")
                    if len(value) > 0:
                        logger.info(f"First item keys: {list(value[0].keys()) if isinstance(value[0], dict) else type(value[0])}")
        
        # 获取搜索结果数据
        successful_searches = []
        
        # 主要路径: search_results 字段
        if "search_results" in data:
            all_results = data["search_results"]
            # 只处理成功的搜索结果
            successful_searches = [result for result in all_results if result.get("status") == "success"]
            logger.info(f"Found search_results field with {len(all_results)} total items, {len(successful_searches)} successful")
        
        # 备用路径: successful_searches 字段
        elif "successful_searches" in data:
            successful_searches = data["successful_searches"]
            logger.info(f"Found successful_searches field with {len(successful_searches)} items")
        
        # 其他可能的字段名
        else:
            for key in ["results", "data"]:
                if key in data and isinstance(data[key], list):
                    successful_searches = data[key]
                    logger.info(f"Found {key} field with {len(successful_searches)} items")
                    break
        
        logger.info(f"Found {len(successful_searches)} successful searches to evaluate")
        
        # 检查是否使用中间结果
        if intermediate_file and os.path.exists(intermediate_file):
            logger.info("Using existing intermediate judgment results...")
            successful_searches, results = self.load_intermediate_results(intermediate_file)
            
            # 直接从中间结果构建评估结果
            evaluation_results = self._build_evaluation_results_from_tuples(successful_searches, results)
        else:
            logger.info("Performing GPT judgment requests...")
            # 执行GPT判断
            if save_intermediate:
                # 如果需要保存中间结果，获取原始结果
                evaluation_results, results = self.evaluate_all_searches_threaded(successful_searches, return_raw_results=True)
                
                # 生成带时间戳的中间文件名
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                intermediate_output = f"intermediate_judgments_{base_name}_{timestamp}.json"
                
                logger.info("Saving intermediate results...")
                self.save_intermediate_results(successful_searches, results, intermediate_output)
            else:
                # 正常执行，不保存中间结果
                evaluation_results = self.evaluate_all_searches_threaded(successful_searches)
        
        logger.info("Calculating metrics...")
        metrics = self.calculate_metrics(evaluation_results)
        
        return evaluation_results, metrics
    
    def save_intermediate_results(self, successful_searches: List[Dict], results: List[Tuple], output_path: str):
        """
        Save intermediate judgment results to avoid re-requesting GPT.
        
        Args:
            successful_searches: Original search data
            results: List of (index, search_index, memory_index, judgment_result) tuples
            output_path: Path to save intermediate results
        """
        # 按搜索分组整理中间结果
        intermediate_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_used": self.model,
            "total_searches": len(successful_searches),
            "total_judgments": len(results),
            "searches_with_judgments": []
        }
        
        # 按搜索索引分组
        results_by_search = {}
        for _, search_idx, memory_idx, judgment in results:
            if search_idx not in results_by_search:
                results_by_search[search_idx] = []
            results_by_search[search_idx].append((memory_idx, judgment))
        
        # 构建详细的中间结果
        for search_idx, search_data in enumerate(successful_searches):
            if search_idx in results_by_search:
                memories = search_data["search_result"]["data"]["text_mem"][0]["memories"]
                memory_judgments = []
                
                # 按memory_idx排序
                search_results = sorted(results_by_search[search_idx], key=lambda x: x[0])
                
                for memory_idx, judgment in search_results:
                    memory_item = memories[memory_idx]
                    memory_judgments.append({
                        "memory_index": memory_idx,
                        "memory_id": memory_item.get("id", ""),
                        "memory_content": memory_item.get("memory", ""),
                        "judgment": judgment
                    })
                
                intermediate_data["searches_with_judgments"].append({
                    "search_index": search_idx,
                    "question_id": search_data.get("question_id", f"q_{search_idx}"),
                    "question": search_data["question"],
                    "answer": search_data["answer"],
                    "memory_judgments": memory_judgments
                })
        
        with open(os.path.join(BASE_CACHED_DIR, output_path), 'w', encoding='utf-8') as f:
            json.dump(intermediate_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Intermediate results saved to: {output_path}")
    
    def load_intermediate_results(self, intermediate_path: str) -> Tuple[List[Dict], List[Tuple]]:
        """
        Load intermediate judgment results from file.
        
        Args:
            intermediate_path: Path to intermediate results file
            
        Returns:
            Tuple of (reconstructed_searches, results_tuples)
        """
        logger.info(f"Loading intermediate results from: {intermediate_path}")
        
        with open(intermediate_path, 'r', encoding='utf-8') as f:
            intermediate_data = json.load(f)
        
        logger.info(f"Loaded {intermediate_data['total_judgments']} judgments from {intermediate_data['total_searches']} searches")
        
        # 重构搜索数据和结果元组
        reconstructed_searches = []
        results_tuples = []
        task_index = 0
        
        for search_item in intermediate_data["searches_with_judgments"]:
            search_idx = len(reconstructed_searches)
            
            # 重构搜索数据（简化版，只包含必要信息）
            search_data = {
                "question_id": search_item["question_id"],
                "question": search_item["question"],
                "answer": search_item["answer"],
                "search_result": {
                    "data": {
                        "text_mem": [{
                            "memories": []
                        }]
                    }
                }
            }
            
            # 重构memories和结果元组
            for judgment_item in search_item["memory_judgments"]:
                memory_idx = judgment_item["memory_index"]
                
                # 添加memory到搜索数据
                memory_data = {
                    "id": judgment_item["memory_id"],
                    "memory": judgment_item["memory_content"]
                }
                search_data["search_result"]["data"]["text_mem"][0]["memories"].append(memory_data)
                
                # 添加结果元组
                results_tuples.append((task_index, search_idx, memory_idx, judgment_item["judgment"]))
                task_index += 1
            
            reconstructed_searches.append(search_data)
        
        return reconstructed_searches, results_tuples
    
    def save_results(self, evaluation_results: List[Dict], metrics: Dict, output_path: str):
        """
        Save detailed evaluation results and metrics to files.
        
        Args:
            evaluation_results: List of detailed evaluation results
            metrics: Calculated metrics
            output_path: Base path to save the results
        """
        # 准备完整的结果数据
        complete_results = {
            "evaluation_summary": {
                "total_questions": len(evaluation_results),
                "total_memories": sum(result["total_memories"] for result in evaluation_results),
                "total_relevant_memories": sum(result["relevant_memories"] for result in evaluation_results),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_used": self.model,
                "max_workers": self.max_workers,
                "rate_limit_delay": self.rate_limit_delay
            },
            "metrics": metrics,
            "detailed_results": evaluation_results
        }
        
        # 保存完整的详细结果
        output_path = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + output_path
        with open(os.path.join(BASE_EVAL_DIR, output_path), 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2, ensure_ascii=False)
        
        # 同时保存一个简化版本，只包含统计信息
        summary_path = output_path.replace('.json', '_summary.json')
        summary_results = {
            "evaluation_summary": complete_results["evaluation_summary"],
            "metrics": metrics,
            "questions_summary": [
                {
                    "question_id": result["question_id"],
                    "question": result["question"],
                    "answer": result["answer"],
                    "total_memories": result["total_memories"],
                    "relevant_memories": result["relevant_memories"],
                    "relevance_rate": result["relevant_memories"] / result["total_memories"] if result["total_memories"] > 0 else 0
                }
                for result in evaluation_results
            ]
        }
        
        with open(os.path.join(BASE_EVAL_DIR, summary_path), 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed results saved to: {output_path}")
        logger.info(f"Summary results saved to: {summary_path}")
    
    def print_metrics(self, metrics: Dict, evaluation_results: List[Dict] = None):
        """Print metrics in a formatted way."""
        print("\n" + "="*70)
        print("EVALUATION METRICS (THREADING VERSION)")
        print("="*70)
        
        print(f"Total Queries: {metrics['total_queries']}")
        print(f"Average Memories per Query: {metrics['avg_memories_per_query']:.2f}")
        print(f"Average Relevant per Query: {metrics['avg_relevant_per_query']:.2f}")
        print(f"Overall Relevance Rate: {metrics['overall_relevance_rate']:.4f}")
        
        print("\nTop-N Accuracy (至少包含一个相关结果的查询比例):")
        for n in [1, 3, 5, 10, 20]:
            if f"accuracy@{n}" in metrics:
                print(f"  Accuracy@{n}: {metrics[f'accuracy@{n}']:.4f}")
        
        print("\nTop-N Hit Rate (包含正确答案的查询比例):")
        for n in [1, 3, 5, 10, 20]:
            if f"hit_rate@{n}" in metrics:
                print(f"  Hit Rate@{n}: {metrics[f'hit_rate@{n}']:.4f}")
        
        print("\nTop-N Recall Rate (二元召回率: 有相关结果=100%, 无相关结果=0%):")
        for n in [1, 3, 5, 10, 20]:
            if f"recall@{n}" in metrics:
                print(f"  Recall@{n}: {metrics[f'recall@{n}']:.4f}")
        
        print("\nAverage Relevant Items in Top-N:")
        for n in [1, 3, 5, 10, 20]:
            if f"avg_relevant@{n}" in metrics:
                print(f"  Avg Relevant@{n}: {metrics[f'avg_relevant@{n}']:.4f}")
        
        print(f"\nMAP (Mean Average Precision): {metrics['MAP']:.4f}")
        print(f"NDCG (Normalized Discounted Cumulative Gain): {metrics['NDCG']:.4f}")
        
        # 显示一些详细的统计信息
        if evaluation_results:
            print(f"\nDetailed Statistics:")
            zero_relevant = sum(1 for r in evaluation_results if r['relevant_memories'] == 0)
            all_relevant = sum(1 for r in evaluation_results if r['relevant_memories'] == r['total_memories'])
            print(f"  Questions with 0 relevant memories: {zero_relevant} ({zero_relevant/len(evaluation_results)*100:.1f}%)")
            print(f"  Questions with all memories relevant: {all_relevant} ({all_relevant/len(evaluation_results)*100:.1f}%)")
            
            # 显示相关性分布
            relevance_rates = [r['relevant_memories']/r['total_memories'] if r['total_memories'] > 0 else 0 for r in evaluation_results]
            print(f"  Average relevance rate per question: {sum(relevance_rates)/len(relevance_rates):.4f}")
            print(f"  Min relevance rate: {min(relevance_rates):.4f}")
            print(f"  Max relevance rate: {max(relevance_rates):.4f}")
        
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate search results using OpenAI model (Threading)")
    parser.add_argument("input_file", help="Path to the JSON file containing search results or intermediate judgment results")
    parser.add_argument("--output", "-o", help="Output file path for results", 
                       default="evaluation_results.json")
    parser.add_argument("--model", "-m", help="OpenAI model to use", 
                       default="gpt-4o-mini")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--base-url", default="http://openai.com/v1", help="OpenAI API base URL (or set OPENAI_API_BASE env var)")
    parser.add_argument("--max-workers", "-w", type=int, default=20,
                       help="Maximum number of worker threads (default: 20)")
    parser.add_argument("--rate-limit-delay", "-d", type=float, default=0.1,
                       help="Delay between requests in seconds (default: 0.1)")
    parser.add_argument("--intermediate-file", "-i", help="Path to intermediate judgment results file (if exists, skip GPT requests)")
    parser.add_argument("--save-intermediate", action="store_true", 
                       help="Save intermediate judgment results for future use")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return 1
    
    # Initialize evaluator
    try:
        evaluator = ThreadedSearchResultsEvaluator(
            openai_api_key=args.api_key,
            model=args.model,
            max_workers=args.max_workers,
            rate_limit_delay=args.rate_limit_delay,
            base_url=args.base_url
        )
        
        logger.info(f"Using threaded evaluation with max_workers={args.max_workers}, "
                   f"rate_limit_delay={args.rate_limit_delay}s")
        
        # Run evaluation
        start_time = time.time()
        evaluation_results, metrics = evaluator.evaluate_file(
            args.input_file, 
            intermediate_file=args.intermediate_file,
            save_intermediate=args.save_intermediate
        )
        end_time = time.time()
        
        logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")
        
        # Print metrics
        evaluator.print_metrics(metrics, evaluation_results)
        
        # Save results
        evaluator.save_results(evaluation_results, metrics, args.output)
        
        logger.info("Evaluation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())