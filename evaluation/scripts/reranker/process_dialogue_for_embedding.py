import json
import requests
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Any
import math
import argparse
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI API配置
base_url = "http://openai.com/v1"
base_model = "bge-m3"

class EmbeddingEvaluator:
    def __init__(self, embedding_url: str = "http://openai.com/v1/v1/embeddings", batch_size: int = 50):
        self.embedding_url = embedding_url
        self.headers = {'Content-Type': 'application/json'}
        self.batch_size = batch_size
    
    def format_single_turn(self, user_msg: str, assistant_msg: str, chat_time: str) -> str:
        """format single turn"""
        return f"chat_time:{chat_time}\nuser：{user_msg}\nassistant:{assistant_msg}"
    
    def format_multi_turn(self, messages: List[Dict], chat_time: str) -> str:
        """format multi turn"""
        formatted = f"chat_time:{chat_time}\n"
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            formatted += f"{role}：{content}\n"
        return formatted.strip()
    
    def create_documents_single_turn(self, data: Dict) -> Tuple[List[str], int]:
        """create single turn documents
        Args:
            data: Dict, the data of a single turn
        Returns:
            documents: List[str], the documents of a single turn
            correct_answer_index: int, the index of the correct answer
        """
        documents = []
        correct_answer_index = -1
        answer_session_ids = set(data.get('answer_session_ids', []))
        assert len(data['haystack_sessions']) == len(data['haystack_session_ids']) == len(data['haystack_dates'])
        for i, (session, session_id, date) in enumerate(zip(
            data['haystack_sessions'], 
            data['haystack_session_ids'], 
            data['haystack_dates']
        )):
            if not session:
                continue
            # 处理每个session中的对话
            j = 0
            while j < len(session):
                if j + 1 < len(session):
                    user_msg = session[j].get('content', '')
                    assistant_msg = session[j + 1].get('content', '')
                    
                    # 检查是否包含正确答案
                    user_answer = session[j].get('has_answer', False)
                    assistant_answer = session[j + 1].get('has_answer', False)
                    has_answer = user_answer or assistant_answer
                    
                    doc = self.format_single_turn(user_msg, assistant_msg, date)
                    documents.append(doc)

                    if has_answer and (session_id in answer_session_ids):
                        correct_answer_index = len(documents) - 1
                    
                    j += 2
                else:
                    j += 1
        
        return documents, correct_answer_index
    
    def create_documents_multi_turn(self, data: Dict, n_turns: int = 3) -> Tuple[List[str], int]:
        """create multi turn documents
        Args:
            data: Dict, the data of a multi turn
            n_turns: int, the number of turns
        Returns:
            documents: List[str], the documents of a multi turn
            correct_answer_index: int, the index of the correct answer
        """
        documents = []
        correct_answer_index = -1
        answer_session_ids = set(data.get('answer_session_ids', []))
        
        for session, session_id, date in zip(
            data['haystack_sessions'], 
            data['haystack_session_ids'], 
            data['haystack_dates']
        ):
            # 按n_turns切分session
            for start_idx in range(0, len(session), n_turns):
                end_idx = min(start_idx + n_turns, len(session))
                messages = session[start_idx:end_idx]
                
                if not messages:
                    continue
                
                # 检查这个片段是否包含正确答案
                has_answer = any(msg.get('has_answer', False) for msg in messages)
                
                doc = self.format_multi_turn(messages, date)
                documents.append(doc)
                
                if has_answer and session_id in answer_session_ids:
                    correct_answer_index = len(documents) - 1
        
        return documents, correct_answer_index
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """调用OpenAI embedding API获取文本嵌入
        Args:
            texts: List[str], 文本列表
        Returns:
            List[List[float]], 嵌入向量列表
        """
        payload = {
            "model": base_model,
            "input": texts
        }
        
        try:
            response = requests.post(
                self.embedding_url, 
                headers=self.headers, 
                data=json.dumps(payload),
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # 提取嵌入向量
            embeddings = []
            for item in result.get('data', []):
                embeddings.append(item['embedding'])
            
            return embeddings
        except Exception as e:
            print(f"Embedding API调用失败: {e}")
            return []
    
    def get_embeddings_batched(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """分批调用embedding API并合并结果
        Args:
            texts: List[str], 文本列表
            batch_size: int, 每批处理的文本数量
        Returns:
            List[List[float]], 合并后的嵌入向量列表
        """
        if len(texts) <= batch_size:
            return self.get_embeddings(texts)
        
        all_embeddings = []
        total_batches = math.ceil(len(texts) / batch_size)
        
        print(f"文本总数: {len(texts)}, 分为 {total_batches} 批处理，每批 {batch_size} 个文本")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            print(f"处理第 {batch_idx + 1}/{total_batches} 批 (文本 {start_idx}-{end_idx-1})")
            
            # 调用embedding API
            batch_embeddings = self.get_embeddings(batch_texts)
            
            if not batch_embeddings:
                print(f"第 {batch_idx + 1} 批处理失败，跳过")
                continue
            
            all_embeddings.extend(batch_embeddings)
        
        if not all_embeddings:
            print("所有批次都处理失败")
            return []
        
        print(f"分批处理完成，共处理 {len(all_embeddings)} 个文本")
        return all_embeddings
    
    def rank_by_similarity(self, query: str, documents: List[str], use_batched: bool = True) -> List[Dict]:
        """基于余弦相似度对文档进行排序
        Args:
            query: str, 查询文本
            documents: List[str], 文档列表
            use_batched: bool, 是否使用分批处理
        Returns:
            List[Dict], 排序结果，格式类似reranker API
        """
        # 准备所有文本（查询 + 文档）
        all_texts = [query] + documents
        
        # 获取嵌入向量
        if use_batched:
            embeddings = self.get_embeddings_batched(all_texts, self.batch_size)
        else:
            embeddings = self.get_embeddings(all_texts)
        
        if not embeddings or len(embeddings) != len(all_texts):
            print(f"嵌入向量获取失败，期望 {len(all_texts)} 个，实际获得 {len(embeddings)} 个")
            return []
        
        # 分离查询和文档的嵌入向量
        query_embedding = np.array(embeddings[0]).reshape(1, -1)
        doc_embeddings = np.array(embeddings[1:])
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # 创建结果列表
        results = []
        for i, similarity in enumerate(similarities):
            results.append({
                'index': i,
                'relevance_score': float(similarity),
                'document': documents[i]
            })
        
        # 按相似度降序排序
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return results
    
    def calculate_metrics(self, rankings: List[List[int]], correct_indices: List[int]) -> Dict[str, float]:
        """计算排序评估指标"""
        metrics = {}
        
        # Hit@K 计算
        for k in [1,3,5,10,20]:
            hit_k = 0
            for ranking, correct_idx in zip(rankings, correct_indices):
                if correct_idx != -1 and correct_idx in ranking[:k]:
                    hit_k += 1
            metrics[f'Hit@{k}'] = hit_k / len(rankings) if rankings else 0
        
        # MRR 计算
        mrr_sum = 0
        valid_queries = 0
        for ranking, correct_idx in zip(rankings, correct_indices):
            if correct_idx != -1:
                valid_queries += 1
                try:
                    rank = ranking.index(correct_idx) + 1
                    mrr_sum += 1.0 / rank
                except ValueError:
                    pass  # 正确答案不在ranking中
        
        metrics['MRR'] = mrr_sum / valid_queries if valid_queries > 0 else 0
        
        # MAP 计算
        ap_sum = 0
        for ranking, correct_idx in zip(rankings, correct_indices):
            if correct_idx != -1:
                # 对于ranking任务，每个query只有一个相关文档
                try:
                    rank = ranking.index(correct_idx) + 1
                    ap_sum += 1.0 / rank
                except ValueError:
                    pass
        
        metrics['MAP'] = ap_sum / len(rankings) if rankings else 0
        
        # NDCG@K 计算
        for k in [1, 5, 10, 20]:
            ndcg_sum = 0
            for ranking, correct_idx in zip(rankings, correct_indices):
                if correct_idx != -1:
                    dcg = 0
                    idcg = 1  # 理想情况下相关文档排在第一位
                    
                    for i, doc_idx in enumerate(ranking[:k]):
                        if doc_idx == correct_idx:
                            dcg = 1.0 / math.log2(i + 2)
                            break
                    
                    ndcg_sum += dcg / idcg if idcg > 0 else 0
            
            metrics[f'NDCG@{k}'] = ndcg_sum / len(rankings) if rankings else 0
        
        # Precision@K 和 Recall@K
        for k in [1, 5, 10, 20]:
            precision_sum = 0
            recall_sum = 0
            
            for ranking, correct_idx in zip(rankings, correct_indices):
                if correct_idx != -1:
                    top_k = ranking[:k]
                    if correct_idx in top_k:
                        precision_sum += 1.0 / len(top_k)
                        recall_sum += 1.0  # 只有一个相关文档
            
            metrics[f'Precision@{k}'] = precision_sum / len(rankings) if rankings else 0
            metrics[f'Recall@{k}'] = recall_sum / len(rankings) if rankings else 0
        
        return metrics
    
    def evaluate_single_question(self, data: Dict, mode: str = 'single', n_turns: int = 3, use_batched: bool = True) -> Dict:
        """评估单个问题"""
        query = data['question']
        
        if mode == 'single':
            documents, correct_idx = self.create_documents_single_turn(data)
        else:
            documents, correct_idx = self.create_documents_multi_turn(data, n_turns)
        
        if not documents:
            return {'error': 'No documents generated'}
        if correct_idx == -1:
            return {'error': 'No correct answer index generated'}
        
        print(f"Query: {query}")
        print(f"Documents count: {len(documents)}")
        print(f"Correct answer index: {correct_idx}")
        
        # 基于embedding相似度排序
        similarity_results = self.rank_by_similarity(query, documents, use_batched)
        
        if not similarity_results:
            return {'error': 'Embedding similarity calculation failed'}
        
        # 提取排序后的文档索引
        ranking = [result['index'] for result in similarity_results]
        
        return {
            'query': query,
            'documents_count': len(documents),
            'correct_index': correct_idx,
            'ranking': ranking,
            'similarity_results': similarity_results
        }
    
    def evaluate_dataset(self, dataset: List[Dict], mode: str = 'single', n_turns: int = 3, use_batched: bool = True) -> Dict:
        """评估整个数据集"""
        all_rankings = []
        all_correct_indices = []
        failed_queries = []
        
        for i, data in enumerate(dataset):
            print(f"\n处理第 {i+1}/{len(dataset)} 个问题...")
            
            result = self.evaluate_single_question(data, mode, n_turns, use_batched)
            
            if 'error' in result:
                print(f"问题 {i+1} 处理失败: {result['error']}")
                failed_queries.append(i)
                continue
            
            all_rankings.append(result['ranking'])
            all_correct_indices.append(result['correct_index'])
        
        # 计算评估指标
        metrics = self.calculate_metrics(all_rankings, all_correct_indices)
        
        return {
            'metrics': metrics,
            'total_queries': len(dataset),
            'successful_queries': len(all_rankings),
            'failed_queries': failed_queries,
            'mode': mode,
            'n_turns': n_turns if mode == 'multi' else None,
            'use_batched': use_batched,
            'batch_size': self.batch_size if use_batched else None
        }

# 使用示例
def main():
    # 加载数据
    with open('/Users/chunyuli/Documents/osworkspace/MemOS/tmp/data/longmemeval_s.json', 'r', encoding='utf-8') as f:
        data = json.load(f)[:100]
    
    # 如果数据是单个样本，转换为列表
    if isinstance(data, dict):
        dataset = [data]
    else:
        dataset = data
    
    # 创建评估器，设置批次大小为10
    evaluator = EmbeddingEvaluator(batch_size=10)
    
    # 评估单轮对话模式 - 使用分批处理
    print("评估单轮对话模式 (使用分批处理)...")
    single_results = evaluator.evaluate_dataset(dataset, mode='single', use_batched=True)
    
    print("\n单轮对话模式结果 (分批处理):")
    for metric, value in single_results['metrics'].items():
        print(f"{metric}: {value:.4f}")
    print(f"批次大小: {single_results['batch_size']}")
    
    # 评估多轮对话模式 - 使用分批处理
    print("\n评估多轮对话模式 (使用分批处理)...")
    multi_results = evaluator.evaluate_dataset(dataset, mode='multi', n_turns=3, use_batched=True)
    
    print("\n多轮对话模式结果 (分批处理):")
    for metric, value in multi_results['metrics'].items():
        print(f"{metric}: {value:.4f}")
    print(f"批次大小: {multi_results['batch_size']}")
    
    # 保存结果
    results = {
        'single_turn_batched': single_results,
        'multi_turn_batched': multi_results,
        'evaluation_config': {
            'batch_size': evaluator.batch_size,
            'embedding_url': evaluator.embedding_url,
            'model': base_model,
            'base_url': base_url
        }
    }
    prefix = datetime.now().strftime('%Y%m%d_%H%M%S') + f"_{base_model}_embedding_batched"
    with open(f'evaluation_results_{prefix}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估完成，结果已保存到 evaluation_results_{prefix}.json")

if __name__ == "__main__":
    main() 