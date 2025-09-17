import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import logging
from tqdm import tqdm
from datetime import datetime
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#url = "http://192.168.0.51:8001"
url = "http://127.0.0.1:8001"

# 线程安全的计数器
class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
    
    @property
    def value(self):
        with self._lock:
            return self._value

# 全局计数器
success_counter = ThreadSafeCounter()
failed_counter = ThreadSafeCounter()

def _search(user_id, mem_cube_id, query, max_retries=3, retry_delay=1):
    """搜索记忆，带重试机制"""
    for attempt in range(max_retries):
        try:
            _url = f"{url}/product/search"
            payload = json.dumps({
                "user_id": user_id,
                "mem_cube_id": mem_cube_id,
                "query": query,
                "top_k": 50
            })
            headers = {
                'Content-Type': 'application/json'
            }
            response = requests.request("POST", _url, headers=headers, data=payload, timeout=600)
            result = json.loads(response.text)
            return {"status": "success", "result": result, "response_code": response.status_code}
        except Exception as e:
            logger.warning(f"搜索 {user_id} 失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # 指数退避
            else:
                logger.error(f"搜索 {user_id} 最终失败: {str(e)}")
                return {"status": "error", "error": str(e), "response_code": None}

def create_users_from_existing_data(datalines, prefix):
    """从现有数据构造用户信息列表（不实际创建用户，因为用户已经存在）"""
    user_infos = []
    
    logger.info(f"从现有数据构建 {len(datalines)} 个用户信息...")
    
    for data in datalines:
        question_id = data["question_id"]
        user_name = f"{prefix}_{question_id}"
        user_infos.append({
            "user_id": user_name,
            "mem_cube_id": user_name,  # 假设mem_cube_id与user_id相同
            "query": data["question"],
            "data": data
        })
    
    logger.info(f"用户信息构建完成: {len(user_infos)} 个用户")
    return user_infos

def process_user_search(user_info):
    """处理单个用户的搜索请求"""
    try:
        user_id = user_info["user_id"]
        mem_cube_id = user_info["mem_cube_id"]
        data = user_info["data"]
        
        # 从数据中获取查询问题和其他字段
        query = data.get("question")
        question_id = data.get("question_id")
        question_type = data.get("question_type")
        answer = data.get("answer")
        question_date = data.get("question_date")
        
        logger.info(f"开始为用户 {user_id} 执行搜索")
        
        # 执行搜索
        search_result = _search(user_id=user_id, mem_cube_id=mem_cube_id, query=query)
        
        # 构建基础返回信息
        base_result = {
            "user_id": user_id,
            "question_id": question_id,
            "question_type": question_type,
            "question": query,
            "answer": answer,
            "question_date": question_date,
        }
        
        if search_result["status"] == "success":
            success_counter.increment()
            logger.info(f"用户 {user_id} 搜索成功")
            return {
                "status": "success",
                **base_result,
                "search_result": search_result["result"],
                "response_code": search_result["response_code"]
            }
        else:
            failed_counter.increment()
            logger.error(f"用户 {user_id} 搜索失败: {search_result['error']}")
            return {
                "status": "error",
                **base_result,
                "error": search_result["error"],
                "response_code": search_result["response_code"]
            }
            
    except Exception as e:
        failed_counter.increment()
        error_msg = f"为用户 {user_id} 执行搜索时发生异常: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "exception", 
            "user_id": user_id, 
            "question_id": data.get("question_id") if 'data' in locals() else None,
            "error": error_msg
        }

def process_searches_concurrently_ordered(user_infos, max_workers=5):
    """
    并发处理搜索请求，但按原始顺序返回结果
    
    Args:
        user_infos: 用户信息列表
        max_workers: 最大并发数
    """
    results = [None] * len(user_infos)  # 预分配结果数组
    total_tasks = len(user_infos)
    
    if total_tasks == 0:
        logger.warning("没有用户信息，跳过搜索处理")
        return []
    
    logger.info(f"开始并发处理 {total_tasks} 个用户的搜索请求，最大并发数: {max_workers}")
    
    # 创建搜索处理进度条
    search_progress = tqdm(
        total=total_tasks,
        desc="搜索处理",
        position=0,
        ncols=100
    )
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务，保存索引信息
        future_to_index = {
            executor.submit(process_user_search, user_info): idx 
            for idx, user_info in enumerate(user_infos)
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_index):
            try:
                result = future.result()
                index = future_to_index[future]
                results[index] = result  # 按原始索引存储结果
                
                # 更新进度条
                completed = sum(1 for r in results if r is not None)
                search_progress.update(1)
                search_progress.set_postfix({
                    "成功": success_counter.value,
                    "失败": failed_counter.value,
                    "完成率": f"{completed/total_tasks*100:.1f}%"
                })
                
            except Exception as e:
                failed_counter.increment()
                logger.error(f"搜索处理任务执行异常: {str(e)}")
                index = future_to_index[future]
                results[index] = {
                    "status": "exception", 
                    "user_id": user_infos[index].get("user_id", "unknown"),
                    "error": str(e)
                }
                search_progress.update(1)
    
    search_progress.close()
    return results

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='LongMemEval 搜索工具')
    parser.add_argument('--prefix', 
                       type=str, 
                       default='2025091302-testneo4j',
                       help='用户名前缀')
    parser.add_argument('--workers', 
                       type=int, 
                       default=10,
                       help='并发数量')
    parser.add_argument('--data-file', 
                       type=str, 
                       default='/Users/chunyuli/Documents/osworkspace/MemOS/tmp/data/longmemeval_s.json',
                       help='数据文件路径')
    parser.add_argument('--top',
                       type=int,
                       default=100,
                       help='top数量')
    
    args = parser.parse_args()
    
    # 加载数据
    try:
        with open(args.data_file, 'r', encoding='utf-8') as f:
            datalines = json.load(f)[:args.top]
        logger.info(f"成功加载数据文件: {args.data_file}, 共 {len(datalines)} 条数据")
    except Exception as e:
        logger.error(f"加载数据文件失败: {e}")
        return
    
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info(f"开始执行搜索，前缀: {args.prefix}, 并发数: {args.workers}")
    logger.info("=" * 60)
    
    # 构建用户信息（因为用户已经创建过了，所以不需要调用实际的createuser接口）
    logger.info("=" * 50)
    logger.info("构建用户信息")
    logger.info("=" * 50)
    user_infos = create_users_from_existing_data(datalines, args.prefix)
    
    # 并发处理搜索请求，按顺序返回结果
    logger.info("=" * 50)
    logger.info("并发处理搜索请求，按顺序返回结果")
    logger.info("=" * 50)
    search_results = process_searches_concurrently_ordered(user_infos, max_workers=args.workers)
    
    end_time = time.time()
    
    # 统计最终结果
    successful_searches = [r for r in search_results if r.get("status") == "success"]
    failed_searches = [r for r in search_results if r.get("status") in ["error", "exception"]]
    
    # 打印详细统计
    logger.info("=" * 50)
    logger.info("搜索完成！最终统计：")
    logger.info("=" * 50)
    logger.info(f"总耗时: {end_time - start_time:.2f} 秒")
    logger.info(f"总数据条数: {len(datalines)}")
    logger.info(f"搜索处理 - 成功: {len(successful_searches)}, 失败: {len(failed_searches)}")
    if len(datalines) > 0:
        logger.info(f"整体成功率: {len(successful_searches)/len(datalines)*100:.1f}%")
    
    # 保存详细结果
    final_results = {
        "config": {
            "prefix": args.prefix,
            "workers": args.workers,
            "data_file": args.data_file,
            "top": args.top
        },
        "summary": {
            "total_data": len(datalines),
            "searches_success": len(successful_searches),
            "searches_failed": len(failed_searches),
            "total_time": end_time - start_time,
            "success_rate": len(successful_searches)/len(datalines)*100 if len(datalines) > 0 else 0
        },
        "search_results": search_results
    }
    
    # 生成结果文件名
    suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_filename = f"./eval_cache_data/search_results_{args.prefix}_{suffix}.json"
    with open(result_filename, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"详细结果已保存到 {result_filename}")
    
    # 额外保存一个简化的结果文件，只包含搜索结果
    simplified_results = {
        "summary": final_results["summary"],
        "successful_searches": [
            {
                "user_id": r["user_id"],
                "question_id": r["question_id"],
                "question_type": r["question_type"],
                "question": r["question"],
                "answer": r["answer"],
                "question_date": r["question_date"],
                "search_result": r["search_result"]
            }
            for r in successful_searches
        ]
    }
    
    simplified_filename = f"./eval_cache_data/search_results_simplified_{args.prefix}_{suffix}.json"
    with open(simplified_filename, "w", encoding="utf-8") as f:
        json.dump(simplified_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"简化结果已保存到 {simplified_filename}")

# 主执行部分
if __name__ == "__main__":
    main()
