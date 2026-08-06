"""Deterministic multi-agent negotiation protocol.

Model and hardware clients are injected by callers so the protocol remains
independently testable and reusable across execution backends.
"""
import json
import copy
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Agent:
    """Agent实体"""
    agent_id: int
    weight: float
    solution: str = ""  # 当前方案（代码）
    history_consistency: float = 0.0  # 历史累计一致性得分 Γ_i
    is_alive: bool = True
    
    def __repr__(self):
        return f"Agent(id={self.agent_id}, weight={self.weight:.3f}, alive={self.is_alive})"


@dataclass
class VoteResult:
    """投票结果"""
    agent_id: int
    top_k: List[int]  # 推荐的Agent ID序列（禁止投自己）


class MultiAgentNegotiationFramework:
    """
    多Agent协商框架
    
    核心机制：
    1. 确定性单调淘汰机制（淘汰期）
    2. 终局单票坍缩与图论精准破局（终局期）
    3. 零超时参数，绝对收敛保证
    """
    
    def __init__(self, N: int, K: int, alpha: float = 0.1, gamma: float = 0.3):
        if N < 2:
            raise ValueError("N must be at least 2")
        if not 1 <= K <= N:
            raise ValueError("K must satisfy 1 <= K <= N")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must satisfy 0 < gamma <= 1")
        self.N = N  # 初始Agent总数
        self.K = K  # 核心监控阈值
        self.alpha = alpha  # 权重更新学习率
        self.gamma = gamma  # 终局权重吸收率
        
        # 初始化Agent
        self.agents: List[Agent] = [
            Agent(agent_id=i, weight=1.0/N)
            for i in range(N)
        ]
        
        self.round = 0
        self.history: List[Dict] = []
        
    @property
    def alive_agents(self) -> List[Agent]:
        """获取所有存活Agent"""
        return [a for a in self.agents if a.is_alive]
    
    @property
    def N_alive(self) -> int:
        """当前存活Agent数"""
        return len(self.alive_agents)
    
    @property
    def is_elimination_phase(self) -> bool:
        """是否在淘汰期"""
        return self.N_alive > self.K
    
    def generate_initial_solutions(self, llm_func) -> None:
        """步骤1：环境初始化与方案生成"""
        print(f"\n[步骤1] 初始化 {self.N} 个Agent并生成初始方案")
        for agent in self.alive_agents:
            agent.solution = llm_func(agent.agent_id, "initial")
            print(f"  Agent {agent.agent_id} 生成初始方案")
    
    def calculate_weighted_scores(self, votes: List[VoteResult]) -> Dict[int, float]:
        """
        计算加权投票得分
        
        Score(x)(t) = Σ W_i(t) × (I(x ∈ S_i) / Pos(x, S_i))
        """
        scores = {agent.agent_id: 0.0 for agent in self.alive_agents}
        
        for vote in votes:
            voter = next(a for a in self.alive_agents if a.agent_id == vote.agent_id)
            for pos, candidate_id in enumerate(vote.top_k, 1):
                if candidate_id in scores:
                    scores[candidate_id] += voter.weight / pos
        
        return scores
    
    def get_global_ranking(self, scores: Dict[int, float]) -> List[int]:
        """获取全局方案排序 S^global"""
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    def calculate_consistency_score(self, agent: Agent, vote: VoteResult, 
                                   global_top_k: List[int]) -> float:
        """
        计算单轮一致性得分 C_i(t)
        
        C_i(t) = Σ [1/Pos(x, S_i) × 1/Pos(x, S^global)] for x ∈ (S_i ∩ S^global)
        """
        score = 0.0
        global_set = set(global_top_k[:self.K])
        
        for pos_i, candidate_id in enumerate(vote.top_k, 1):
            if candidate_id in global_set:
                pos_global = global_top_k.index(candidate_id) + 1
                score += (1.0 / pos_i) * (1.0 / pos_global)
        
        return score
    
    def update_weights_elimination_phase(self, consistency_scores: Dict[int, float]) -> None:
        """
        淘汰期权重更新
        
        W_i^(t+1) = W_i^(t) × (1 + α × C_i(t))
        """
        for agent in self.alive_agents:
            c_i = consistency_scores.get(agent.agent_id, 0.0)
            agent.weight *= (1 + self.alpha * c_i)
        
        # 归一化
        total_weight = sum(a.weight for a in self.alive_agents)
        for agent in self.alive_agents:
            agent.weight /= total_weight
    
    def eliminate_one_agent(self) -> Optional[Agent]:
        """
        确定性单调淘汰机制
        
        1. 淘汰权重最低的Agent
        2. 若并列，淘汰 Γ_i 最小者
        3. 若仍并列，淘汰ID最大者
        """
        alive = self.alive_agents
        
        # 排序：权重升序 → 历史得分升序 → ID降序
        alive.sort(key=lambda a: (a.weight, a.history_consistency, -a.agent_id))
        
        eliminated = alive[0]
        eliminated.is_alive = False
        print(f"  [淘汰] Agent {eliminated.agent_id} (weight={eliminated.weight:.4f}, Γ={eliminated.history_consistency:.4f})")
        
        return eliminated
    
    def update_weights_final_phase(self, votes: List[VoteResult]) -> None:
        """
        终局期权重吸收
        
        W_j^* = W_j(t) + γ × Σ W_i(t) for i ∈ V_j
        """
        # 收集投票给每个Agent的权重
        received_weights = {agent.agent_id: 0.0 for agent in self.alive_agents}
        
        for vote in votes:
            if vote.top_k:  # Best-1投票
                target_id = vote.top_k[0]
                voter = next(a for a in self.alive_agents if a.agent_id == vote.agent_id)
                received_weights[target_id] += voter.weight
        
        # 更新权重
        for agent in self.alive_agents:
            agent.weight += self.gamma * received_weights[agent.agent_id]
        
        # 归一化
        total_weight = sum(a.weight for a in self.alive_agents)
        for agent in self.alive_agents:
            agent.weight /= total_weight
    
    def detect_cycles_and_eliminate(self, votes: List[VoteResult] = None) -> Optional[Agent]:
        """
        图论精准斩断：检测环并淘汰环内 Γ_i 最低的Agent
        
        参数:
            votes: 本轮投票结果（必须传入）
        """
        # 构建投票图
        alive_ids = {a.agent_id for a in self.alive_agents}
        votes_map = {}  # agent_id -> voted_agent_id
        
        # 从传入的投票记录中获取
        if votes:
            for vote in votes:
                if vote.agent_id in alive_ids and vote.top_k:
                    target = vote.top_k[0]
                    if target in alive_ids:  # 确保投票目标也存活
                        votes_map[vote.agent_id] = target
        else:
            # 兜底：从history获取
            if not self.history:
                return None
            last_round = self.history[-1]
            if 'votes' not in last_round:
                return None
            for vote in last_round['votes']:
                if vote.agent_id in alive_ids and vote.top_k:
                    target = vote.top_k[0]
                    if target in alive_ids:
                        votes_map[vote.agent_id] = target
        
        if not votes_map:
            return None
        
        # DFS检测环
        visited = set()
        in_cycle = set()
        
        def find_cycle(start_id):
            path = []
            path_set = set()
            current = start_id
            while current in votes_map:
                if current in path_set:
                    # 找到环
                    cycle_start = path.index(current)
                    return set(path[cycle_start:])
                if current in visited:
                    return set()
                visited.add(current)
                path.append(current)
                path_set.add(current)
                current = votes_map[current]
            return set()
        
        for agent_id in alive_ids:
            if agent_id not in visited:
                cycle = find_cycle(agent_id)
                in_cycle.update(cycle)
        
        if not in_cycle:
            # 如果没有检测到环，淘汰历史得分最低的
            print(f"  [无环] 淘汰历史得分最低者")
            alive = self.alive_agents
            alive.sort(key=lambda a: (a.history_consistency, -a.agent_id))
            eliminated = alive[0]
        else:
            # 在环内淘汰 Γ_i 最低的Agent
            print(f"  [检测到环] 环内Agent: {in_cycle}")
            cycle_agents = [a for a in self.alive_agents if a.agent_id in in_cycle]
            cycle_agents.sort(key=lambda a: (a.history_consistency, -a.agent_id))
            eliminated = cycle_agents[0]
        
        eliminated.is_alive = False
        print(f"  [图论斩断] Agent {eliminated.agent_id} (in_cycle={eliminated.agent_id in in_cycle}, Γ={eliminated.history_consistency:.4f})")
        
        return eliminated
    
    def check_termination(self) -> Optional[Agent]:
        """
        步骤3：状态判定与即时胜出拦截
        
        1. 检查是否有Agent权重 > 50%
        2. 检查是否仅剩1个Agent
        """
        # 检查权重胜出
        for agent in self.alive_agents:
            if agent.weight > 0.5:
                print(f"  [胜出] Agent {agent.agent_id} 权重 {agent.weight:.4f} > 50%")
                return agent
        
        # 检查是否仅剩1个
        if self.N_alive == 1:
            winner = self.alive_agents[0]
            print(f"  [胜出] Agent {winner.agent_id} 是唯一存活者")
            return winner
        
        return None
    
    def run_round(self, llm_refine_func, llm_vote_func) -> Optional[Agent]:
        """
        执行一轮协商
        
        返回胜出的Agent，如果没有则返回None
        """
        self.round += 1
        
        # 捕获本轮开始时的阶段状态（用于历史记录，避免淘汰后 is_elimination_phase 变化导致阶段标签错误）
        _round_start_phase = 'elimination' if self.is_elimination_phase else 'endgame'
        
        # 构建权重分布（设计文档要求广播）
        weight_distribution = {
            a.agent_id: {
                'weight': a.weight,
                'history_consistency': a.history_consistency
            }
            for a in self.alive_agents
        }
        
        print(f"\n{'='*60}")
        print(f"[轮次 {self.round}] 存活Agent: {self.N_alive}, 阶段: {'淘汰期' if self.is_elimination_phase else '终局期'}")
        print(f"{'='*60}")
        
        # 打印当前权重分布
        print("\n[当前权重分布]")
        for agent in self.alive_agents:
            print(f"  Agent {agent.agent_id}: W={agent.weight:.4f}, Γ={agent.history_consistency:.4f}")
        
        # 步骤2：方案反思迭代与交叉投票
        print("\n[步骤2.1] 方案反思与迭代")
        agent_solutions = {}  # 记录每个Agent的新方案（保存完整内容）
        for agent in self.alive_agents:
            old_solution = agent.solution
            agent.solution = llm_refine_func(agent, self.alive_agents, weight_distribution)
            agent_solutions[agent.agent_id] = {
                'old_solution': old_solution if old_solution else None,
                'new_solution': agent.solution if agent.solution else None,
                'old_solution_preview': old_solution[:200] if old_solution else None,
                'new_solution_preview': agent.solution[:200] if agent.solution else None,
            }
        
        print("\n[步骤2.2] 交叉投票")
        if self.is_elimination_phase:
            # 淘汰期：Top-K投票
            votes = []
            vote_details = {}  # 记录投票详情
            for agent in self.alive_agents:
                top_k = llm_vote_func(agent, self.alive_agents, top_k=self.K, weight_distribution=weight_distribution)
                votes.append(VoteResult(agent_id=agent.agent_id, top_k=top_k))
                vote_details[agent.agent_id] = {
                    'voted_for': top_k,
                    'vote_type': f'Top-{self.K}'
                }
        else:
            # 终局期：Best-1投票
            votes = []
            vote_details = {}
            for agent in self.alive_agents:
                best_1 = llm_vote_func(agent, self.alive_agents, top_k=1, weight_distribution=weight_distribution)
                votes.append(VoteResult(agent_id=agent.agent_id, top_k=best_1[:1]))
                vote_details[agent.agent_id] = {
                    'voted_for': best_1[:1],
                    'vote_type': 'Best-1'
                }
        
        # 计算得分和全局排序
        scores = self.calculate_weighted_scores(votes)
        global_ranking = self.get_global_ranking(scores)
        
        # 打印投票结果和得分
        print("\n[投票结果]")
        for agent, vote in zip(self.alive_agents, votes):
            print(f"  Agent {agent.agent_id} 投票给: {vote.top_k} ({vote_details[agent.agent_id]['vote_type']})")
        
        print(f"\n[加权得分]")
        for agent_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  Agent {agent_id}: {score:.4f}")
        
        print(f"\n[全局排序]: {global_ranking}")
        
        # 计算一致性得分
        consistency_scores = {}
        for agent, vote in zip(self.alive_agents, votes):
            c_i = self.calculate_consistency_score(agent, vote, global_ranking)
            consistency_scores[agent.agent_id] = c_i
            agent.history_consistency += c_i
            print(f"  Agent {agent.agent_id} 一致性得分: C={c_i:.4f}, Γ={agent.history_consistency:.4f}")
        
        # 步骤3：检查终止
        winner = self.check_termination()
        if winner:
            # 记录最终历史
            self.history.append({
                'round': self.round,
                'phase': '终局期' if not self.is_elimination_phase else '淘汰期',
                'N_alive_before': self.N_alive,
                'agent_solutions': agent_solutions,
                'votes': [
                    {
                        'agent_id': v.agent_id,
                        'voted_for': v.top_k,
                        'vote_type': vote_details[v.agent_id]['vote_type']
                    }
                    for v in votes
                ],
                'scores': scores,
                'consistency_scores': consistency_scores,
                'weights_after': {a.agent_id: a.weight for a in self.alive_agents},
                'action': f'Agent {winner.agent_id} 胜出',
                'winner': winner.agent_id
            })
            return winner
        
        # 步骤4：权重更新与状态分流
        action_taken = ""
        eliminated_agent = None
        
        if self.is_elimination_phase:
            print("\n[步骤4] 淘汰期 - 权重更新与淘汰")
            weights_before = {a.agent_id: a.weight for a in self.alive_agents}
            print("  权重更新前:")
            for agent_id, w in weights_before.items():
                print(f"    Agent {agent_id}: {w:.4f}")
            
            self.update_weights_elimination_phase(consistency_scores)
            
            print("  权重更新后:")
            for agent in self.alive_agents:
                print(f"    Agent {agent.agent_id}: {agent.weight:.4f} (Δ={agent.weight - weights_before.get(agent.agent_id, 0):.4f})")
            
            # 淘汰期权重更新后即时胜出检查（对齐设计文档步骤3）
            winner = self.check_termination()
            if winner:
                self.history.append({
                    'round': self.round,
                    'phase': '淘汰期',
                    'N_alive_before': self.N_alive,
                    'agent_solutions': agent_solutions,
                    'votes': [
                        {
                            'agent_id': v.agent_id,
                            'voted_for': v.top_k,
                            'vote_type': vote_details[v.agent_id]['vote_type']
                        }
                        for v in votes
                    ],
                    'scores': scores,
                    'consistency_scores': consistency_scores,
                    'weights_before': weights_before,
                    'weights_after': {a.agent_id: a.weight for a in self.alive_agents},
                    'action': f'Agent {winner.agent_id} 权重胜出（淘汰期权重更新后突破50%）',
                    'winner': winner.agent_id
                })
                print(f"  [行动] Agent {winner.agent_id} 权重突破50%，直接胜出")
                return winner
            
            eliminated = self.eliminate_one_agent()
            eliminated_agent = eliminated.agent_id if eliminated else None
            action_taken = f"淘汰 Agent {eliminated_agent}"
            print(f"  [行动] {action_taken}")
        else:
            print("\n[步骤4] 终局期 - 权重吸收")
            weights_before = {a.agent_id: a.weight for a in self.alive_agents}
            print("  权重吸收前:")
            for agent_id, w in weights_before.items():
                print(f"    Agent {agent_id}: {w:.4f}")
            
            self.update_weights_final_phase(votes)
            
            print("  权重吸收后:")
            for agent in self.alive_agents:
                print(f"    Agent {agent.agent_id}: {agent.weight:.4f} (Δ={agent.weight - weights_before.get(agent.agent_id, 0):.4f})")
            
            # 检查是否有权重 > 50%
            winner = self.check_termination()
            if winner:
                self.history.append({
                    'round': self.round,
                    'phase': '终局期',
                    'N_alive_before': self.N_alive,
                    'agent_solutions': agent_solutions,
                    'votes': [
                        {
                            'agent_id': v.agent_id,
                            'voted_for': v.top_k,
                            'vote_type': vote_details[v.agent_id]['vote_type']
                        }
                        for v in votes
                    ],
                    'scores': scores,
                    'consistency_scores': consistency_scores,
                    'weights_before': weights_before,
                    'weights_after': {a.agent_id: a.weight for a in self.alive_agents},
                    'action': f'Agent {winner.agent_id} 权重胜出',
                    'winner': winner.agent_id
                })
                return winner
            
            # 图论精准斩断
            print("\n[步骤4] 检测死锁环")
            eliminated = self.detect_cycles_and_eliminate(votes)
            eliminated_agent = eliminated.agent_id if eliminated else None
            action_taken = f"图论斩断 Agent {eliminated_agent}"
        
        # 记录历史
        self.history.append({
            'round': self.round,
            'phase': '淘汰期' if _round_start_phase == 'elimination' else '终局期',
            'N_alive_before': self.N_alive + (1 if eliminated_agent is not None else 0),
            'N_alive_after': self.N_alive,
            'agent_solutions': agent_solutions,
            'votes': [
                {
                    'agent_id': v.agent_id,
                    'voted_for': v.top_k,
                    'vote_type': vote_details[v.agent_id]['vote_type']
                }
                for v in votes
            ],
            'scores': scores,
            'consistency_scores': consistency_scores,
            'weights_before': weights_before if 'weights_before' in dir() else {a.agent_id: a.weight for a in self.alive_agents},
            'weights_after': {a.agent_id: a.weight for a in self.alive_agents},
            'action': action_taken,
            'eliminated_agent': eliminated_agent
        })
        
        return None
    
    def run(self, llm_initial_func, llm_refine_func, llm_vote_func) -> Agent:
        """
        运行完整协商流程
        
        参数：
            llm_initial_func: (agent_id, phase) -> solution
            llm_refine_func: (agent, alive_agents) -> refined_solution
            llm_vote_func: (agent, alive_agents, top_k) -> List[agent_id]
        """
        print(f"\n{'#'*60}")
        print(f"# 多Agent协商框架启动")
        print(f"# N={self.N}, K={self.K}")
        print(f"{'#'*60}")
        
        # 步骤1：初始化
        self.generate_initial_solutions(llm_initial_func)
        
        # 严格遵循框架设计：无外部超时兜底，收敛性由算法内在保证
        # PDF设计文档明确指出："不再依赖任何外部超时兜底"
        # 收敛性证明：每轮必然淘汰1个Agent，最多N-1轮从N递减到1
        while True:
            winner = self.run_round(llm_refine_func, llm_vote_func)
            if winner:
                print(f"\n{'#'*60}")
                print(f"# 协商完成！胜出者: Agent {winner.agent_id}")
                print(f"# 总轮数: {self.round}")
                print(f"# 总LLM调用: 见实验统计")
                print(f"{'#'*60}")
                return winner
            # 安全断言（理论上不可达，仅防御编程错误）
            assert self.round < self.N, \
                f"框架收敛性被破坏：已运行{self.round}轮仍未终止，请检查算法实现"
