import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import logging
from tqdm import tqdm
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# url = "http://192.168.0.51:8001"

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

def _create_user(user_name, max_retries=3, retry_delay=1):
    """创建用户，带重试机制"""
    for attempt in range(max_retries):
        try:
            _url = f"{url}/product/users/register"
            payload = json.dumps({
                "user_name": user_name,
                "user_id": user_name,
                "mem_cube_id": user_name
            })
            headers = {
                'Content-Type': 'application/json'
            }
            response = requests.request("POST", _url, headers=headers, data=payload, timeout=600)
            return json.loads(response.text)
        except Exception as e:
            logger.warning(f"创建用户 {user_name} 失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # 指数退避
            else:
                logger.error(f"创建用户 {user_name} 最终失败: {str(e)}")
                return {"code": 500, "error": str(e)}

def _add(user_name, sessions, haystack_dates, max_retries=3, retry_delay=1):
    """添加会话数据，带重试机制，按照13轮一个batch，3轮重叠的方式添加"""
    success_add = []
    
    # 为当前用户的会话添加创建进度条
    session_progress = tqdm(
        zip(sessions, haystack_dates), 
        total=len(sessions),
        desc=f"添加 {user_name} 的会话",
        leave=False,  # 完成后不保留进度条
        position=1,   # 设置进度条位置，避免与主进度条冲突
        ncols=100     # 设置进度条宽度
    )
    
    for session, chat_time in session_progress:
        session_added = False
        for attempt in range(max_retries):

            try:
                update_session = []
                for d in session:
                    d.update({"chat_time": chat_time})
                    update_session.append(d)
                _url = f"{url}/product/add"
                payload = json.dumps({
                    "user_id": user_name,
                    "mem_cube_id": user_name,
                    "messages": update_session
                })
                headers = {
                    'Content-Type': 'application/json'
                }
                response = requests.request("POST", _url, headers=headers, data=payload, timeout=600)
                success_add.append(user_name)
                session_added = True
                break  # 成功则跳出重试循环
            except Exception as e:
                logger.warning(f"添加会话数据 {user_name} 失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"添加会话数据 {user_name} 最终失败: {str(e)}")
        
        # 更新进度条描述，显示当前会话的处理状态
        if session_added:
            session_progress.set_postfix({"状态": "成功", "已添加": len(success_add)})
        else:
            session_progress.set_postfix({"状态": "失败", "已添加": len(success_add)})
    
    session_progress.close()
    return success_add

def create_users_sequentially(datalines, prefix):
    """串行创建所有用户"""
    created_users = []
    failed_users = []
    
    logger.info(f"开始串行创建 {len(datalines)} 个用户...")
    
    # 创建用户创建进度条
    user_creation_progress = tqdm(
        datalines,
        desc="创建用户",
        position=0,
        ncols=100
    )
    
    for data in user_creation_progress:
        try:
            question_id = data["question_id"]
            user_name = f"{prefix}_{question_id}"
            
            # 创建用户
            response_users = _create_user(user_name=user_name)
            if response_users["code"] == 200:
                created_users.append({
                    "user_name": user_name,
                    "data": data
                })
                user_creation_progress.set_postfix({
                    "成功": len(created_users),
                    "失败": len(failed_users)
                })
            else:
                failed_users.append({
                    "user_name": user_name,
                    "error": response_users,
                    "data": data
                })
                logger.error(f"创建用户 {user_name} 失败: {response_users}")
                user_creation_progress.set_postfix({
                    "成功": len(created_users),
                    "失败": len(failed_users)
                })
                
        except Exception as e:
            failed_users.append({
                "user_name": f"{prefix}_{data.get('question_id', 'unknown')}",
                "error": str(e),
                "data": data
            })
            logger.error(f"创建用户时发生异常: {str(e)}")
    
    user_creation_progress.close()
    
    logger.info(f"用户创建完成: 成功 {len(created_users)}, 失败 {len(failed_users)}")
    return created_users, failed_users

def process_user_sessions(user_info):
    """处理单个用户的会话数据（用于并行执行）"""
    try:
        user_name = user_info["user_name"]
        data = user_info["data"]
        haystack_dates = data["haystack_dates"]
        haystack_sessions = data["haystack_sessions"]
        
        logger.info(f"开始为用户 {user_name} 添加会话数据")
        
        # 添加会话数据
        success_add = _add(user_name=user_name, sessions=haystack_sessions, haystack_dates=haystack_dates)
        success_counter.increment()
        logger.info(f"用户 {user_name} 会话数据添加成功")
        return {"status": "success", "user_name": user_name, "added_sessions": len(success_add)}
            
    except Exception as e:
        failed_counter.increment()
        error_msg = f"为用户 {user_name} 添加会话数据时发生异常: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "user_name": user_name, "error": error_msg}

def process_sessions_concurrently(created_users, max_workers=5):
    """
    并发处理会话数据
    
    Args:
        created_users: 已创建的用户列表
        max_workers: 最大并发数
    """
    results = []
    total_tasks = len(created_users)
    
    if total_tasks == 0:
        logger.warning("没有成功创建的用户，跳过会话数据处理")
        return results
    
    logger.info(f"开始并发处理 {total_tasks} 个用户的会话数据，最大并发数: {max_workers}")
    
    # 创建会话处理进度条
    session_progress = tqdm(
        total=total_tasks,
        desc="会话数据处理",
        position=0,
        ncols=100
    )
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_user = {
            executor.submit(process_user_sessions, user_info): user_info 
            for user_info in created_users
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_user):
            try:
                result = future.result()
                results.append(result)
                
                # 更新进度条
                completed = len(results)
                session_progress.update(1)
                session_progress.set_postfix({
                    "成功": success_counter.value,
                    "失败": failed_counter.value,
                    "完成率": f"{completed/total_tasks*100:.1f}%"
                })
                
            except Exception as e:
                failed_counter.increment()
                logger.error(f"会话处理任务执行异常: {str(e)}")
                results.append({"status": "exception", "error": str(e)})
                session_progress.update(1)
    
    session_progress.close()
    return results

def create_users_from_existing_data(datalines, prefix):
    """从现有数据创建用户信息列表（不实际创建用户）"""
    created_users = []
    
    logger.info(f"从现有数据构建 {len(datalines)} 个用户信息...")
    
    for data in datalines:
        question_id = data["question_id"]
        user_name = f"{prefix}_{question_id}"
        created_users.append({
            "user_name": user_name,
            "data": data
        })
    
    logger.info(f"用户信息构建完成: {len(created_users)} 个用户")
    return created_users

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='LongMemEval 数据处理工具')
    parser.add_argument('--mode', 
                       choices=['create_only', 'add_only', 'both'], 
                       default='both',
                       help='执行模式: create_only(仅创建用户), add_only(仅添加会话), both(两步都执行)')
    parser.add_argument('--prefix', 
                       type=str, 
                       default='2025091302-testneo4j',
                       help='用户名前缀')
    parser.add_argument('--workers', 
                       type=int, 
                       default=20,
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
    created_users = []
    failed_users = []
    session_results = []
    
    logger.info("=" * 60)
    logger.info(f"开始执行，模式: {args.mode}, 前缀: {args.prefix}, 并发数: {args.workers}")
    logger.info("=" * 60)
    
    # 根据模式执行不同的操作
    if args.mode in ['create_only', 'both']:
        # 第一步：串行创建所有用户
        logger.info("=" * 50)
        logger.info("第一阶段：串行创建用户")
        logger.info("=" * 50)
        created_users, failed_users = create_users_sequentially(datalines, args.prefix)
        
        if args.mode == 'create_only':
            logger.info("=" * 50)
            logger.info("仅创建用户模式完成！")
            logger.info("=" * 50)
            logger.info(f"用户创建 - 成功: {len(created_users)}, 失败: {len(failed_users)}")
    
    if args.mode in ['add_only', 'both']:
        # 如果是仅添加模式，需要从数据构建用户列表
        if args.mode == 'add_only':
            logger.info("=" * 50)
            logger.info("仅添加会话模式：构建用户信息")
            logger.info("=" * 50)
            created_users = create_users_from_existing_data(datalines, args.prefix)
        
        # 第二步：并发处理会话数据
        logger.info("=" * 50)
        logger.info("第二阶段：并发处理会话数据")
        logger.info("=" * 50)
        session_results = process_sessions_concurrently(created_users, max_workers=args.workers)
    
    end_time = time.time()
    
    # 统计最终结果
    successful_sessions = [r for r in session_results if r.get("status") == "success"]
    failed_sessions = [r for r in session_results if r.get("status") in ["error", "exception"]]
    
    # 打印详细统计
    logger.info("=" * 50)
    logger.info("处理完成！最终统计：")
    logger.info("=" * 50)
    logger.info(f"执行模式: {args.mode}")
    logger.info(f"总耗时: {end_time - start_time:.2f} 秒")
    logger.info(f"总数据条数: {len(datalines)}")
    
    if args.mode in ['create_only', 'both']:
        logger.info(f"用户创建 - 成功: {len(created_users)}, 失败: {len(failed_users)}")
    
    if args.mode in ['add_only', 'both']:
        logger.info(f"会话处理 - 成功: {len(successful_sessions)}, 失败: {len(failed_sessions)}")
        if len(datalines) > 0:
            logger.info(f"整体成功率: {len(successful_sessions)/len(datalines)*100:.1f}%")
    
    # 保存详细结果
    final_results = {
        "config": {
            "mode": args.mode,
            "prefix": args.prefix,
            "workers": args.workers,
            "data_file": args.data_file
        },
        "summary": {
            "total_data": len(datalines),
            "users_created": len(created_users) if args.mode in ['create_only', 'both'] else 0,
            "users_failed": len(failed_users) if args.mode in ['create_only', 'both'] else 0,
            "sessions_success": len(successful_sessions),
            "sessions_failed": len(failed_sessions),
            "total_time": end_time - start_time,
            "success_rate": len(successful_sessions)/len(datalines)*100 if len(datalines) > 0 else 0
        },
        "failed_users": failed_users if args.mode in ['create_only', 'both'] else [],
        "session_results": session_results
    }
    
    # 根据模式生成不同的结果文件名
    result_filename = f"processing_results_{args.mode}_{args.prefix}.json"
    with open(result_filename, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"详细结果已保存到 {result_filename}")

# 主执行部分
if __name__ == "__main__":
    main()