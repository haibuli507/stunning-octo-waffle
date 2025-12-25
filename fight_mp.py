#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Gao Fang
@contact: gaofang@cetc.com.cn
@software: PyCharm
@file: fight_mp.py
@time: 2018/3/9 0009 16:41
@desc: execution battle between two agents - 集成战术策略版本 (Side1专用) - 适配3000x3000地图
"""
import argparse
import os
import time
import sys
import math
import random
import json
from datetime import datetime

# 添加环境路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'environment'))

from interface import Environment
from common.agent_process import AgentCtrl


class TacticalStrategy:
    """战术策略类 - 实现阵型协同与战术决策 (只用于Side1) - 适配3000x3000地图"""

    def __init__(self, side_name="Side1", map_size=(3000, 3000)):
        self.side_name = side_name
        self.map_width, self.map_height = map_size
        self.map_center_x = self.map_width // 2
        self.map_center_y = self.map_height // 2

        # 根据地图大小调整战术参数
        # 原始参数基于1000x1000地图，现在放大3倍
        scale_factor = 3  # 3000/1000

        self.strategy_state = 'unknown'  # 当前策略状态
        self.roles_assigned = {}  # 角色分配
        self.step_count = 0
        self.enemy_info_history = []  # 敌方信息历史记录

        # 战术参数 - 根据地图大小按比例缩放
        self.support_distance_threshold = 300 * scale_factor  # 支援距离阈值
        self.attack_distance_threshold = 180 * scale_factor  # 攻击距离阈值（雷达最大距离）
        self.retreat_distance_threshold = 150 * scale_factor  # 撤退距离阈值
        self.formation_distance = 100 * scale_factor  # 编队距离
        self.last_strategy = 'hold'  # 上次的策略
        self.strategy_change_count = 0  # 策略切换次数

        # 新参数：地图边界缓冲区
        self.map_boundary_buffer = 100 * scale_factor

    def analyze_situation(self, our_fighters, our_detectors, enemy_passive_list, step_cnt):
        """分析战场态势（基于文档的策略）"""
        # 获取我方战斗机数量
        our_fighter_count = len(our_fighters)

        # 从被动侦测列表获取敌方战斗机数量
        enemy_fighter_count = 0
        enemy_detector_count = 0
        enemy_positions = []

        for enemy in enemy_passive_list:
            if enemy.get('type', 1) == 1:  # 攻击单元
                enemy_fighter_count += 1
                enemy_positions.append((enemy.get('pos_x', 0), enemy.get('pos_y', 0)))
            elif enemy.get('type', 0) == 0:  # 探测单元
                enemy_detector_count += 1
                enemy_positions.append((enemy.get('pos_x', 0), enemy.get('pos_y', 0)))

        # 计算数量对比 N = 我方战斗数量 - 敌方战斗数量
        N = our_fighter_count - enemy_fighter_count

        # 计算支援可达性 S（简化：检查是否有战斗机在后方）
        S = 0
        if our_fighter_count >= 4:  # 如果有4架以上战斗机，假设有支援
            S = 1

        # 判断敌方进攻性 E（简化：根据历史位置判断）
        E = 0  # 默认敌方不进攻
        if len(self.enemy_info_history) >= 2:
            # 简单判断：如果敌方在接近，则认为是进攻
            if len(enemy_positions) > 0 and len(self.enemy_info_history[-1]) > 0:
                # 计算平均距离变化
                prev_positions = self.enemy_info_history[-1]
                if len(prev_positions) == len(enemy_positions):
                    total_dist_change = 0
                    for i in range(min(len(prev_positions), len(enemy_positions))):
                        # 使用地图中心作为参考点
                        prev_dist = self.calculate_distance(prev_positions[i][0], prev_positions[i][1],
                                                            self.map_center_x, self.map_center_y)
                        curr_dist = self.calculate_distance(enemy_positions[i][0], enemy_positions[i][1],
                                                            self.map_center_x, self.map_center_y)
                        total_dist_change += (curr_dist - prev_dist)

                    if total_dist_change < -150:  # 如果敌方总体向中心靠近，阈值放大3倍
                        E = 1  # 敌方进攻

        # 保存当前敌方位置历史
        self.enemy_info_history.append(enemy_positions)
        if len(self.enemy_info_history) > 10:  # 只保留最近10步
            self.enemy_info_history.pop(0)

        # 根据文档规则选择策略
        situation = {
            'our_fighter_count': our_fighter_count,
            'enemy_fighter_count': enemy_fighter_count,
            'our_detector_count': len(our_detectors),
            'enemy_detector_count': enemy_detector_count,
            'N': N,  # 数量对比
            'S': S,  # 支援可达性
            'E': E,  # 敌方进攻性
            'strategy': 'hold',
            'tactic': '保持阵型',
            'step': step_cnt,
            'map_size': (self.map_width, self.map_height)
        }

        # 根据文档的决策逻辑
        if N < 0:  # 情况一：我方数量劣势
            if S == 1:  # 有支援
                situation['strategy'] = 'retreat_ambush'
                situation['tactic'] = '佯装撤退，引诱包围'
            else:  # 无支援
                situation['strategy'] = 'retreat_fast'
                situation['tactic'] = '立即撤退，请求支援'
        else:  # 情况二：我方数量均势或优势
            if E == 1:  # 敌方没有撤退意图
                situation['strategy'] = 'pressure_flank'
                situation['tactic'] = '放慢进攻，形成包围'
            else:  # 敌方全力撤退
                situation['strategy'] = 'attack_fast'
                situation['tactic'] = '加快歼灭，力求全歼'

        # 记录策略变化
        if self.last_strategy != situation['strategy']:
            self.strategy_change_count += 1
            self.last_strategy = situation['strategy']

        return situation

    def assign_roles(self, fighter_count, strategy):
        """分配战斗角色（基于文档的三类角色）"""
        roles = {}

        if strategy in ['retreat_ambush', 'retreat_fast']:
            # 情况一：弹性撤退，诱敌深入
            if fighter_count >= 3:
                roles[0] = 'bait'  # 诱饵单元：撤退速度减半，保持雷达照射
                roles[1] = 'assault'  # 突击单元：全速撤退后横向机动
                roles[2] = 'assault'  # 突击单元：从两侧迂回
                for i in range(3, min(fighter_count, 10)):
                    roles[i] = 'support'  # 支援单元：全速向战场机动
            else:
                for i in range(fighter_count):
                    roles[i] = 'bait'  # 战斗机不足，全部作为诱饵

        elif strategy == 'pressure_flank':
            # 情况二：正面施压，侧翼锁喉
            if fighter_count >= 3:
                roles[0] = 'bait'  # 诱饵单元：主动前出，正面接触
                if fighter_count >= 4:
                    roles[1] = 'bait'  # 第二个诱饵单元
                start_idx = 2 if fighter_count >= 4 else 1
                for i in range(start_idx, min(fighter_count, 10)):
                    roles[i] = 'assault'  # 突击单元：大范围迂回，切断撤退路线
            else:
                for i in range(fighter_count):
                    roles[i] = 'bait'

        elif strategy == 'attack_fast':
            # 全力进攻情况
            if fighter_count >= 3:
                # 留下1-2架战斗机掩护探测器
                roles[0] = 'guard'  # 掩护单元：保护探测器
                if fighter_count >= 4:
                    roles[1] = 'guard'
                start_idx = 2 if fighter_count >= 4 else 1
                for i in range(start_idx, min(fighter_count, 10)):
                    roles[i] = 'attack'  # 攻击单元：全力进攻
            else:
                for i in range(fighter_count):
                    roles[i] = 'attack'

        else:  # hold策略
            for i in range(fighter_count):
                roles[i] = 'hold'

        return roles

    def calculate_distance(self, x1, y1, x2, y2):
        """计算两点间的距离"""
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_tactical_detector_action(self, detector_obs, strategy, step_cnt):
        """生成战术探测器动作"""
        if not detector_obs.get('alive', False):
            return [0, 0]  # 死亡单位

        current_course = detector_obs.get('course', 0)
        pos_x = detector_obs.get('pos_x', 0)
        pos_y = detector_obs.get('pos_y', 0)

        if strategy in ['retreat_ambush', 'retreat_fast']:
            # 撤退策略：探测器向友军方向撤退
            # 假设友军在战场中心
            dx = self.map_center_x - pos_x
            dy = self.map_center_y - pos_y
            retreat_course = math.degrees(math.atan2(dy, dx)) % 360
            return [int(retreat_course), 1]  # 雷达开启

        elif strategy == 'pressure_flank':
            # 侧翼策略：保持前进，雷达全开
            forward_course = (current_course + random.randint(-5, 5)) % 360
            return [int(forward_course), 1]

        elif strategy == 'attack_fast':
            # 攻击策略：向前推进，雷达开启
            forward_course = (current_course + random.randint(-10, 10)) % 360
            return [int(forward_course), 1]

        else:  # hold
            # 保持策略：小幅调整航向
            adjusted_course = (current_course + random.randint(-5, 5)) % 360
            return [int(adjusted_course), 1]

    def get_tactical_fighter_action(self, fighter_obs, role, fighter_idx, total_fighters,
                                    strategy, enemy_passive_list, step_cnt):
        """生成战术战斗机动作"""
        if not fighter_obs.get('alive', False):
            return [0, 0, 0, 0]  # 死亡单位

        pos_x = fighter_obs.get('pos_x', 0)
        pos_y = fighter_obs.get('pos_y', 0)
        current_course = fighter_obs.get('course', 0)

        # 基础动作：[航向, 雷达频点, 干扰频点, 导弹攻击]
        action = [current_course, 0, 0, 0]

        # 根据文档中的具体战术生成动作
        if role == 'bait':
            if strategy in ['retreat_ambush', 'retreat_fast']:
                # 诱饵单元：撤退速度减半，保持雷达照射
                retreat_course = (current_course + 180) % 360
                # 模拟减速：航向不变，但实际速度由系统控制
                action[0] = retreat_course
                action[1] = 1  # 雷达开启，让敌人看到
                action[2] = 0  # 不干扰
                action[3] = 0  # 不发射导弹（诱饵）

            elif strategy == 'pressure_flank':
                # 诱饵单元：正面接触和周旋，粘住敌人
                action[0] = current_course
                action[1] = 1  # 雷达开启
                # 如果敌人接近，发射导弹牵制
                if enemy_passive_list:
                    closest_enemy = enemy_passive_list[0]
                    dist = self.calculate_distance(pos_x, pos_y,
                                                   closest_enemy.get('pos_x', 0),
                                                   closest_enemy.get('pos_y', 0))
                    if dist < self.attack_distance_threshold:
                        action[3] = random.randint(1, 12)  # 远程导弹攻击

        elif role == 'assault':
            if strategy in ['retreat_ambush', 'retreat_fast']:
                # 突击单元：全速撤退至后方，然后横向机动
                retreat_course = (current_course + 180) % 360
                # 根据战斗机编号决定横向机动方向
                if fighter_idx % 2 == 0:
                    flank_course = (retreat_course + 90) % 360  # 向右
                else:
                    flank_course = (retreat_course - 90) % 360  # 向左
                action[0] = flank_course
                action[1] = 1  # 雷达开启

            elif strategy == 'pressure_flank':
                # 突击单元：大范围急速迂回，切断敌军撤退路线
                # 计算向敌人侧翼移动的方向
                if enemy_passive_list and len(enemy_passive_list) > 0:
                    enemy = enemy_passive_list[0]
                    enemy_x = enemy.get('pos_x', 0)
                    enemy_y = enemy.get('pos_y', 0)

                    # 计算敌人侧翼位置（距离放大3倍）
                    flank_distance = 600  # 原200放大3倍
                    if fighter_idx % 2 == 0:
                        # 右翼包抄
                        target_x = enemy_x + flank_distance
                        target_y = enemy_y
                    else:
                        # 左翼包抄
                        target_x = enemy_x - flank_distance
                        target_y = enemy_y

                    dx = target_x - pos_x
                    dy = target_y - pos_y
                    flank_course = math.degrees(math.atan2(dy, dx)) % 360
                    action[0] = int(flank_course)
                else:
                    # 没有敌人信息，向前方侧翼移动
                    action[0] = (current_course + 45 * (1 if fighter_idx % 2 == 0 else -1)) % 360

                action[1] = 1  # 雷达开启
                # 如果接近敌人，发射导弹
                if enemy_passive_list:
                    closest_enemy = enemy_passive_list[0]
                    dist = self.calculate_distance(pos_x, pos_y,
                                                   closest_enemy.get('pos_x', 0),
                                                   closest_enemy.get('pos_y', 0))
                    if dist < self.attack_distance_threshold:
                        action[3] = random.randint(1, 12)  # 远程导弹攻击

        elif role == 'support':
            # 支援单元：向战场中心移动，准备致命一击
            dx = self.map_center_x - pos_x
            dy = self.map_center_y - pos_y
            support_course = math.degrees(math.atan2(dy, dx)) % 360
            action[0] = int(support_course)
            action[1] = 1  # 雷达开启
            # 支援单元不主动攻击，保持距离

        elif role == 'attack':
            # 攻击单元：全力进攻
            if enemy_passive_list:
                closest_enemy = enemy_passive_list[0]
                enemy_x = closest_enemy.get('pos_x', 0)
                enemy_y = closest_enemy.get('pos_y', 0)
                dx = enemy_x - pos_x
                dy = enemy_y - pos_y
                attack_course = math.degrees(math.atan2(dy, dx)) % 360
                action[0] = int(attack_course)
            action[1] = 1  # 雷达开启
            # 如果敌人在攻击范围内，发射导弹
            if enemy_passive_list:
                closest_enemy = enemy_passive_list[0]
                dist = self.calculate_distance(pos_x, pos_y,
                                               closest_enemy.get('pos_x', 0),
                                               closest_enemy.get('pos_y', 0))
                if dist < self.attack_distance_threshold:
                    action[3] = random.randint(1, 12)  # 远程导弹攻击

        elif role == 'guard':
            # 掩护单元：保护探测器，留在后方
            # 向探测器方向移动或保持位置
            guard_course = (current_course + random.randint(-30, 30)) % 360
            action[0] = guard_course
            action[1] = 1  # 雷达开启
            # 只对接近的敌人攻击
            if enemy_passive_list:
                closest_enemy = enemy_passive_list[0]
                dist = self.calculate_distance(pos_x, pos_y,
                                               closest_enemy.get('pos_x', 0),
                                               closest_enemy.get('pos_y', 0))
                if dist < self.attack_distance_threshold // 2:  # 近距离才攻击
                    action[3] = random.randint(1, 12)

        elif role == 'hold':
            # 保持位置
            action[0] = current_course
            action[1] = 1  # 雷达开启

        return action


def generate_tactical_actions(obs_dict, step_cnt, detector_num, fighter_num, map_size=(3000, 3000)):
    """生成战术动作 - 用于Side1 - 适配3000x3000地图"""
    tactical = TacticalStrategy("Side1", map_size)

    # 解析观测数据
    fighter_obs_list = obs_dict.get('fighter_obs_list', [])
    detector_obs_list = obs_dict.get('detector_obs_list', [])
    joint_obs_dict = obs_dict.get('joint_obs_dict', {})

    # 只处理存活的单位
    alive_fighters = [f for f in fighter_obs_list if f.get('alive', False)]
    alive_detectors = [d for d in detector_obs_list if d.get('alive', False)]

    # 获取敌方信息
    enemy_passive_list = joint_obs_dict.get('passive_detection_enemy_list', [])

    # 分析态势
    situation = tactical.analyze_situation(alive_fighters, alive_detectors, enemy_passive_list, step_cnt)

    # 分配角色
    roles = tactical.assign_roles(len(alive_fighters), situation['strategy'])

    # 生成探测器动作
    detector_actions = []
    for i in range(detector_num):
        if i < len(detector_obs_list):
            detector_obs = detector_obs_list[i]
            action = tactical.get_tactical_detector_action(detector_obs, situation['strategy'], step_cnt)
        else:
            action = [0, 0]  # 超出索引，单位不存在
        detector_actions.append(action)

    # 生成战斗机动作
    fighter_actions = []
    for i in range(fighter_num):
        if i < len(fighter_obs_list):
            fighter_obs = fighter_obs_list[i]
            role = roles.get(i, 'hold')
            action = tactical.get_tactical_fighter_action(
                fighter_obs, role, i, len(alive_fighters),
                situation['strategy'], enemy_passive_list, step_cnt
            )
        else:
            action = [0, 0, 0, 0]  # 超出索引，单位不存在
        fighter_actions.append(action)

    # 输出战术信息（每50步）
    if step_cnt % 50 == 0 or step_cnt <= 10:
        print(f"Side1战术 - 步{step_cnt}: {situation['tactic']}")
        print(f"  数量对比(N): {situation['N']}, 支援(S): {situation['S']}, 敌方进攻(E): {situation['E']}")
        print(f"  策略: {situation['strategy']}, 存活: {len(alive_fighters)}战斗机")
        print(f"  地图尺寸: {map_size[0]}x{map_size[1]}")
        if step_cnt <= 10:  # 前10步显示角色分配
            print(f"  角色分配: {dict(list(roles.items())[:5])}")

    return {
        'detector_action': detector_actions,
        'fighter_action': fighter_actions,
        'situation': situation  # 返回态势信息用于记录
    }


def run_100_matches_tournament():
    """运行100场对局锦标赛 - 适配3000x3000地图"""
    print("=" * 80)
    print("开始100场对局锦标赛 - 测试战术策略效果 (3000x3000地图)")
    print("=" * 80)

    # 配置参数 - 使用新地图
    map_name = "3000_3000_2_10_vs_2_10"  # 修改为新的地图名称
    side1_agent = "tactical"  # Side1使用战术策略
    side2_agent = "fix_rule"  # Side2使用规则智能体
    max_step = 300  # 增加最大步数以适应更大的地图
    total_matches = 100

    # 统计结果
    results = {
        'side1_wins': 0,
        'side2_wins': 0,
        'draws': 0,
        'match_details': [],
        'side1_strategy_usage': {},
        'average_steps': 0,
        'side1_avg_alive': 0,
        'side2_avg_alive': 0,
        'side1_avg_damage': 0,
        'side2_avg_damage': 0
    }

    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"tournament_results_{timestamp}_3000x3000"
    os.makedirs(log_dir, exist_ok=True)

    print(f"配置: {side1_agent} vs {side2_agent}, 地图: {map_name}, 最大步数: {max_step}")
    print(f"总对局数: {total_matches}")
    print("-" * 80)

    # 文件路径 - 使用新地图
    map_path = f'maps/{map_name}.map'
    agent2_path = f'agent/{side2_agent}/agent.py'

    if not os.path.exists(map_path):
        print(f'错误: 地图文件不存在: {map_path}')
        return

    if not os.path.exists(agent2_path):
        print(f'错误: 智能体2文件不存在: {agent2_path}')
        return

    # 初始化环境
    print("初始化环境中...")
    try:
        env = Environment(map_path, 'raw', 'raw', max_step=max_step, render=False,
                          random_pos=True, log=False, external_render=False,
                          side1_name=side1_agent, side2_name=side2_agent)
    except Exception as e:
        print(f'环境初始化错误: {e}')
        return

    # 获取地图信息
    size_x, size_y = env.get_map_size()
    side1_detector_num, side1_fighter_num, side2_detector_num, side2_fighter_num = env.get_unit_num()

    print(f"地图尺寸: {size_x}x{size_y}")
    print(f"Side1: {side1_detector_num}探测器, {side1_fighter_num}战斗机 (使用战术策略)")
    print(f"Side2: {side2_detector_num}探测器, {side2_fighter_num}战斗机 (使用规则智能体)")
    print("-" * 80)

    # 创建Side2智能体
    agent2 = AgentCtrl(side2_agent, size_x, size_y, side2_detector_num, side2_fighter_num, -1)

    if not agent2.agent_init():
        print('错误: 智能体2初始化失败!')
        return
    else:
        print('智能体2初始化成功!')

    total_steps = 0

    # 运行100场对局
    for match_num in range(1, total_matches + 1):
        print(f"\n第 {match_num}/{total_matches} 场对局开始...")

        # 重置环境
        env.reset()
        step_cnt = 0
        match_result = None
        side1_strategy_usage = {}

        while True:
            step_cnt += 1

            # 获取观测
            side1_obs_dict, side2_obs_dict = env.get_obs()

            # Side1: 使用战术策略生成动作 - 传递地图尺寸
            try:
                tactical_action = generate_tactical_actions(
                    side1_obs_dict, step_cnt,
                    side1_detector_num, side1_fighter_num,
                    map_size=(size_x, size_y)  # 传递地图尺寸
                )
                side1_detector_action = tactical_action['detector_action']
                side1_fighter_action = tactical_action['fighter_action']
                side1_result = 0

                # 记录策略使用情况
                situation = tactical_action.get('situation', {})
                strategy = situation.get('strategy', 'unknown')
                if strategy in side1_strategy_usage:
                    side1_strategy_usage[strategy] += 1
                else:
                    side1_strategy_usage[strategy] = 1

            except Exception as e:
                print(f'Side1 战术策略错误: {e}')
                side1_result = 1
                side1_detector_action = None
                side1_fighter_action = None

            # Side2: 使用规则智能体
            agent2_action, agent2_result = agent2.get_action(side2_obs_dict, step_cnt)
            if agent2_result == 0:
                side2_detector_action = agent2_action['detector_action']
                side2_fighter_action = agent2_action['fighter_action']
            else:
                side2_detector_action = None
                side2_fighter_action = None

            # 执行动作
            if side1_result == 0 and agent2_result == 0:
                env.step(side1_detector_action, side1_fighter_action,
                         side2_detector_action, side2_fighter_action)
            elif side1_result != 0 and agent2_result != 0:
                env.set_surrender(2)  # 双方都崩溃，平局
            elif side1_result != 0:
                env.set_surrender(0)  # Side1崩溃，投降
            else:
                env.set_surrender(1)  # Side2崩溃，投降

            # 检查是否结束
            if env.get_done() or step_cnt >= max_step:
                # 获取最终状态
                side1_obs_raw, side2_obs_raw = env.get_obs_raw()
                side1_detector_obs_raw_list = side1_obs_raw['detector_obs_list']
                side1_fighter_obs_raw_list = side1_obs_raw['fighter_obs_list']
                side2_detector_obs_raw_list = side2_obs_raw['detector_obs_list']
                side2_fighter_obs_raw_list = side2_obs_raw['fighter_obs_list']

                # 获取奖励
                o_detector_reward, o_fighter_reward, o_game_reward, e_detector_reward, e_fighter_reward, e_game_reward = env.get_reward()

                # 统计存活单位
                side1_alive_units = 0
                side2_alive_units = 0

                for y in range(side1_detector_num):
                    if y < len(side1_detector_obs_raw_list) and side1_detector_obs_raw_list[y]['alive']:
                        side1_alive_units += 1
                for y in range(side1_fighter_num):
                    if y < len(side1_fighter_obs_raw_list) and side1_fighter_obs_raw_list[y]['alive']:
                        side1_alive_units += 1

                for y in range(side2_detector_num):
                    if y < len(side2_detector_obs_raw_list) and side2_detector_obs_raw_list[y]['alive']:
                        side2_alive_units += 1
                for y in range(side2_fighter_num):
                    if y < len(side2_fighter_obs_raw_list) and side2_fighter_obs_raw_list[y]['alive']:
                        side2_alive_units += 1

                # 判断胜负（基于奖励和存活单位）
                if o_game_reward > e_game_reward:
                    winner = "Side1"
                    results['side1_wins'] += 1
                elif o_game_reward < e_game_reward:
                    winner = "Side2"
                    results['side2_wins'] += 1
                else:
                    # 奖励相等，根据存活单位判断
                    if side1_alive_units > side2_alive_units:
                        winner = "Side1"
                        results['side1_wins'] += 1
                    elif side2_alive_units > side1_alive_units:
                        winner = "Side2"
                        results['side2_wins'] += 1
                    else:
                        winner = "Draw"
                        results['draws'] += 1

                # 记录对局详情
                match_detail = {
                    'match': match_num,
                    'winner': winner,
                    'steps': step_cnt,
                    'side1_alive': side1_alive_units,
                    'side2_alive': side2_alive_units,
                    'side1_reward': o_game_reward,
                    'side2_reward': e_game_reward,
                    'side1_strategy_usage': side1_strategy_usage,
                    'strategy_changes': len(side1_strategy_usage),
                    'map_size': f"{size_x}x{size_y}"
                }
                results['match_details'].append(match_detail)

                # 更新统计数据
                total_steps += step_cnt
                results['side1_avg_alive'] += side1_alive_units
                results['side2_avg_alive'] += side2_alive_units
                results['side1_avg_damage'] += o_game_reward
                results['side2_avg_damage'] += e_game_reward

                # 更新策略使用统计
                for strategy, count in side1_strategy_usage.items():
                    if strategy in results['side1_strategy_usage']:
                        results['side1_strategy_usage'][strategy] += count
                    else:
                        results['side1_strategy_usage'][strategy] = count

                # 显示对局结果
                print(f"第 {match_num} 场结束: {winner} 胜利 (步数: {step_cnt})")
                print(f"  存活: Side1={side1_alive_units}, Side2={side2_alive_units}")
                print(f"  奖励: Side1={o_game_reward}, Side2={e_game_reward}")
                print(f"  Side1策略使用: {side1_strategy_usage}")

                break

        # 每10场显示一次进度
        if match_num % 10 == 0:
            current_win_rate = results['side1_wins'] / match_num * 100
            print(f"\n进度: {match_num}/{total_matches}, Side1胜率: {current_win_rate:.1f}%")
            print(f"当前战绩: Side1 {results['side1_wins']}胜, Side2 {results['side2_wins']}胜, {results['draws']}平")

    # 清理
    agent2.terminate()

    # 计算平均值
    if total_matches > 0:
        results['average_steps'] = total_steps / total_matches
        results['side1_avg_alive'] /= total_matches
        results['side2_avg_alive'] /= total_matches
        results['side1_avg_damage'] /= total_matches
        results['side2_avg_damage'] /= total_matches

    # 计算胜率
    side1_win_rate = results['side1_wins'] / total_matches * 100 if total_matches > 0 else 0
    side2_win_rate = results['side2_wins'] / total_matches * 100 if total_matches > 0 else 0

    # 显示最终结果
    print("\n" + "=" * 80)
    print("100场对局锦标赛结果 (3000x3000地图)")
    print("=" * 80)
    print(f"总对局数: {total_matches}")
    print(f"地图尺寸: {size_x}x{size_y}")
    print(f"Side1 (战术策略) 胜利: {results['side1_wins']} ({side1_win_rate:.1f}%)")
    print(f"Side2 (规则智能体) 胜利: {results['side2_wins']} ({side2_win_rate:.1f}%)")
    print(f"平局: {results['draws']} ({results['draws'] / total_matches * 100:.1f}%)")
    print(f"平均每场步数: {results['average_steps']:.1f}")
    print(f"Side1 平均存活单位: {results['side1_avg_alive']:.1f}")
    print(f"Side2 平均存活单位: {results['side2_avg_alive']:.1f}")
    print(f"Side1 平均奖励: {results['side1_avg_damage']:.1f}")
    print(f"Side2 平均奖励: {results['side2_avg_damage']:.1f}")

    # 显示策略使用统计
    print("\n战术策略使用统计:")
    total_strategy_uses = sum(results['side1_strategy_usage'].values())
    for strategy, count in results['side1_strategy_usage'].items():
        percentage = count / total_strategy_uses * 100 if total_strategy_uses > 0 else 0
        strategy_name = {
            'retreat_ambush': '佯装撤退引诱',
            'retreat_fast': '立即撤退支援',
            'pressure_flank': '正面施压侧翼',
            'attack_fast': '全力进攻歼灭',
            'hold': '保持阵型'
        }.get(strategy, strategy)
        print(f"  {strategy_name}: {count}次 ({percentage:.1f}%)")

    # 策略有效性分析
    print("\n策略有效性分析:")
    if side1_win_rate > 50:
        print(f"✓ 战术策略有效！胜率({side1_win_rate:.1f}%)高于50%")
    elif side1_win_rate > side2_win_rate:
        print(f"✓ 战术策略略有效果，胜率({side1_win_rate:.1f}%)高于对手({side2_win_rate:.1f}%)")
    else:
        print(f"✗ 战术策略效果不佳，胜率({side1_win_rate:.1f}%)低于对手({side2_win_rate:.1f}%)")

    # 保存结果到文件
    results['side1_win_rate'] = side1_win_rate
    results['side2_win_rate'] = side2_win_rate
    results['total_matches'] = total_matches
    results['timestamp'] = timestamp
    results['map_name'] = map_name
    results['map_size'] = f"{size_x}x{size_y}"
    results['side1_agent'] = side1_agent
    results['side2_agent'] = side2_agent
    results['max_step'] = max_step

    result_file = f"{log_dir}/tournament_results.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n详细结果已保存至: {result_file}")
    print("=" * 80)

    # 生成简要报告
    report_file = f"{log_dir}/summary_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("100场对局锦标赛结果报告 (3000x3000地图)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"地图: {map_name} ({size_x}x{size_y})\n")
        f.write(f"Side1: {side1_agent} (战术策略)\n")
        f.write(f"Side2: {side2_agent} (规则智能体)\n")
        f.write(f"最大步数: {max_step}\n")
        f.write(f"总对局数: {total_matches}\n\n")
        f.write("=" * 80 + "\n")
        f.write("胜率统计\n")
        f.write("=" * 80 + "\n")
        f.write(f"Side1 (战术策略) 胜率: {side1_win_rate:.1f}% ({results['side1_wins']}胜)\n")
        f.write(f"Side2 (规则智能体) 胜率: {side2_win_rate:.1f}% ({results['side2_wins']}胜)\n")
        f.write(f"平局率: {results['draws'] / total_matches * 100:.1f}% ({results['draws']}平)\n\n")
        f.write("=" * 80 + "\n")
        f.write("性能指标\n")
        f.write("=" * 80 + "\n")
        f.write(f"平均每场步数: {results['average_steps']:.1f}\n")
        f.write(f"Side1 平均存活单位: {results['side1_avg_alive']:.1f}\n")
        f.write(f"Side2 平均存活单位: {results['side2_avg_alive']:.1f}\n")
        f.write(f"Side1 平均奖励: {results['side1_avg_damage']:.1f}\n")
        f.write(f"Side2 平均奖励: {results['side2_avg_damage']:.1f}\n\n")
        f.write("=" * 80 + "\n")
        f.write("战术策略使用统计\n")
        f.write("=" * 80 + "\n")
        for strategy, count in results['side1_strategy_usage'].items():
            percentage = count / total_strategy_uses * 100 if total_strategy_uses > 0 else 0
            strategy_name = {
                'retreat_ambush': '佯装撤退引诱',
                'retreat_fast': '立即撤退支援',
                'pressure_flank': '正面施压侧翼',
                'attack_fast': '全力进攻歼灭',
                'hold': '保持阵型'
            }.get(strategy, strategy)
            f.write(f"{strategy_name}: {count}次 ({percentage:.1f}%)\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("结论\n")
        f.write("=" * 80 + "\n")
        if side1_win_rate > 50:
            f.write(f"✓ 战术策略有效！胜率({side1_win_rate:.1f}%)高于50%\n")
        elif side1_win_rate > side2_win_rate:
            f.write(f"✓ 战术策略略有效果，胜率({side1_win_rate:.1f}%)高于对手({side2_win_rate:.1f}%)\n")
        else:
            f.write(f"✗ 战术策略效果不佳，胜率({side1_win_rate:.1f}%)低于对手({side2_win_rate:.1f}%)\n")

    print(f"简要报告已保存至: {report_file}")

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="3000_3000_2_10_vs_2_10", help='地图名称')  # 默认改为新地图
    parser.add_argument("--agent1", type=str, default="fix_rule", help='Side1智能体名称')
    parser.add_argument("--agent2", type=str, default="fix_rule", help='Side2智能体名称')
    parser.add_argument("--round", type=int, default=1, help='对局数')
    parser.add_argument("--max_step", type=int, default=300, help='每轮最大步数')  # 增加默认最大步数
    parser.add_argument("--fps", type=float, default=0, help='显示帧率')
    parser.add_argument("--random_pos", action="store_true", help='初始位置是否随机')
    parser.add_argument("--log", action="store_true", help='是否记录对战过程')
    parser.add_argument("--log_path", type=str, default="default_log", help='日志文件夹名称')
    parser.add_argument("--ext_render", action="store_true", help='启用外部渲染')
    parser.add_argument("--agent1_gpu", type=int, default=-1, help='Side1分配的GPU索引')
    parser.add_argument("--agent2_gpu", type=int, default=-1, help='Side2分配的GPU索引')

    # 新增参数
    parser.add_argument("--tactical_side1", action="store_true", help='Side1使用战术策略（替代智能体）')
    parser.add_argument("--tournament_100", action="store_true", help='运行100场对局锦标赛')

    args = parser.parse_args()

    # 如果指定了锦标赛模式，运行100场对局
    if args.tournament_100:
        run_100_matches_tournament()
        return

    # 否则运行普通对战
    print('地图:', args.map)
    print('Side1 智能体:', '战术策略' if args.tactical_side1 else args.agent1)
    print('Side2 智能体:', args.agent2)
    print('对战轮数:', args.round)
    print('每轮最大步数:', args.max_step)

    if args.tactical_side1:
        print('Side1 使用战术策略: 阵型协同与战术决策 (适配3000x3000地图)')

    side1_win_times = 0
    side2_win_times = 0
    draw_times = 0

    # file path constructing
    map_path = 'maps/' + args.map + '.map'
    agent1_path = 'agent/' + args.agent1 + '/agent.py'
    agent2_path = 'agent/' + args.agent2 + '/agent.py'

    if not os.path.exists(map_path):
        print('错误: 地图文件不存在!')
        exit(-1)

    # 检查智能体文件（如果Side1不使用战术策略）
    if not args.tactical_side1 and not os.path.exists(agent1_path):
        print('错误: 智能体1文件不存在!')
        exit(-1)
    if not os.path.exists(agent2_path):
        print('错误: 智能体2文件不存在!')
        exit(-1)

    # delay calc
    if args.fps == 0:
        step_delay = 0
    else:
        step_delay = 1 / args.fps

    # environment initiation
    if args.log:
        if args.log_path == 'default_log':
            log_flag = args.agent1 + '_vs_' + args.agent2
        else:
            log_flag = args.log_path
    else:
        log_flag = False

    print("初始化环境中...")
    try:
        env = Environment(map_path, 'raw', 'raw', max_step=args.max_step, render=True,
                          random_pos=args.random_pos, log=log_flag, external_render=args.ext_render,
                          side1_name=args.agent1, side2_name=args.agent2)
    except Exception as e:
        print(f'环境初始化错误: {e}')
        exit(-1)

    # get map info
    size_x, size_y = env.get_map_size()
    side1_detector_num, side1_fighter_num, side2_detector_num, side2_fighter_num = env.get_unit_num()

    print(f'地图尺寸: {size_x}x{size_y}')
    print(f'Side1: {side1_detector_num}个探测单元(雷达侦察机), {side1_fighter_num}个攻击单元')
    print(f'Side2: {side2_detector_num}个探测单元(雷达侦察机), {side2_fighter_num}个攻击单元')

    # create agent（如果Side1不使用战术策略）
    if not args.tactical_side1:
        agent1 = AgentCtrl(args.agent1, size_x, size_y, side1_detector_num, side1_fighter_num, args.agent1_gpu)
        if not agent1.agent_init():
            print('错误: 智能体1初始化失败!')
            exit(-1)
        else:
            print('智能体1初始化成功!')
    else:
        agent1 = None
        print('Side1 使用内置战术策略')

    # 总是创建Side2智能体
    agent2 = AgentCtrl(args.agent2, size_x, size_y, side2_detector_num, side2_fighter_num, args.agent2_gpu)
    if not agent2.agent_init():
        print('错误: 智能体2初始化失败!')
        if not args.tactical_side1:
            agent1.terminate()
        exit(-1)
    else:
        print('智能体2初始化成功!')

    # execution
    step_cnt = 0
    round_cnt = 0
    agent1_crash_list = []
    agent2_crash_list = []
    agent1_timeout_list = []
    agent2_timeout_list = []

    print("\n" + "=" * 60)
    print("开始对战...")
    print("=" * 60)

    for x in range(args.round):
        side1_total_reward = 0
        side2_total_reward = 0
        if x != 0:
            env.reset()
        step_cnt = 0
        round_cnt += 1
        print(f"\n第 {round_cnt} 轮对战开始...")
        print("-" * 40)

        while True:
            time.sleep(step_delay)
            step_cnt += 1

            # get obs
            side1_obs_dict, side2_obs_dict = env.get_obs()

            # get action for side1
            if args.tactical_side1:
                # 使用战术策略生成动作 - 传递地图尺寸
                try:
                    tactical_action = generate_tactical_actions(
                        side1_obs_dict, step_cnt,
                        side1_detector_num, side1_fighter_num,
                        map_size=(size_x, size_y)  # 传递地图尺寸
                    )
                    side1_detector_action = tactical_action['detector_action']
                    side1_fighter_action = tactical_action['fighter_action']
                    agent1_result = 0  # 成功
                except Exception as e:
                    print(f'Side1 战术策略错误: {e}')
                    agent1_result = 1  # 崩溃
                    side1_detector_action = None
                    side1_fighter_action = None
            else:
                # 使用原智能体
                agent1_action, agent1_result = agent1.get_action(side1_obs_dict, step_cnt)
                if agent1_result == 0:
                    side1_detector_action = agent1_action['detector_action']
                    side1_fighter_action = agent1_action['fighter_action']
                elif agent1_result == 1:
                    agent1_crash_list.append(round_cnt)
                    print('智能体1崩溃!')
                    side1_detector_action = None
                    side1_fighter_action = None
                elif agent1_result == 2:
                    agent1_timeout_list.append(round_cnt)
                    print('智能体1超时!')
                    side1_detector_action = None
                    side1_fighter_action = None

            # get action for side2
            agent2_action, agent2_result = agent2.get_action(side2_obs_dict, step_cnt)
            if agent2_result == 0:
                side2_detector_action = agent2_action['detector_action']
                side2_fighter_action = agent2_action['fighter_action']
            elif agent2_result == 1:
                agent2_crash_list.append(round_cnt)
                print('智能体2崩溃!')
                side2_detector_action = None
                side2_fighter_action = None
            elif agent2_result == 2:
                agent2_timeout_list.append(round_cnt)
                print('智能体2超时!')
                side2_detector_action = None
                side2_fighter_action = None

            # execution
            if (agent1_result == 0 or args.tactical_side1) and agent2_result == 0:
                env.step(side1_detector_action, side1_fighter_action,
                         side2_detector_action, side2_fighter_action)
            elif (agent1_result != 0 and not args.tactical_side1) and agent2_result != 0:
                env.set_surrender(2)  # 双方都崩溃/超时，平局
            elif (agent1_result != 0 and not args.tactical_side1):
                env.set_surrender(0)  # 智能体1崩溃/超时，投降
            elif agent2_result != 0:
                env.set_surrender(1)  # 智能体2崩溃/超时，投降

            # 检查是否达到最大步数
            if step_cnt >= args.max_step:
                print(f'第 {round_cnt} 轮达到最大步数 {args.max_step}！根据存活单位数判断胜负')

                # 获取存活状态
                side1_obs_raw, side2_obs_raw = env.get_obs_raw()
                side1_detector_obs_raw_list = side1_obs_raw['detector_obs_list']
                side1_fighter_obs_raw_list = side1_obs_raw['fighter_obs_list']
                side2_detector_obs_raw_list = side2_obs_raw['detector_obs_list']
                side2_fighter_obs_raw_list = side2_obs_raw['fighter_obs_list']

                # 计算存活单位数
                side1_alive_units = 0
                side2_alive_units = 0

                for y in range(side1_detector_num):
                    if y < len(side1_detector_obs_raw_list) and side1_detector_obs_raw_list[y]['alive']:
                        side1_alive_units += 1
                for y in range(side1_fighter_num):
                    if y < len(side1_fighter_obs_raw_list) and side1_fighter_obs_raw_list[y]['alive']:
                        side1_alive_units += 1

                for y in range(side2_detector_num):
                    if y < len(side2_detector_obs_raw_list) and side2_detector_obs_raw_list[y]['alive']:
                        side2_alive_units += 1
                for y in range(side2_fighter_num):
                    if y < len(side2_fighter_obs_raw_list) and side2_fighter_obs_raw_list[y]['alive']:
                        side2_alive_units += 1

                # 根据存活单位数判断胜负
                if side1_alive_units > side2_alive_units:
                    print('Side 1 WIN!!! (存活单位更多)')
                    side1_win_times += 1
                elif side2_alive_units > side1_alive_units:
                    print('Side 2 WIN!!! (存活单位更多)')
                    side2_win_times += 1
                else:
                    print('DRAW!!! (存活单位数相等)')
                    draw_times += 1

                print(
                    f'Side1 存活单位: {side1_alive_units} (探测器: {sum(1 for d in side1_detector_obs_raw_list if d["alive"])}, 攻击机: {sum(1 for f in side1_fighter_obs_raw_list if f["alive"])})')
                print(
                    f'Side2 存活单位: {side2_alive_units} (探测器: {sum(1 for d in side2_detector_obs_raw_list if d["alive"])}, 攻击机: {sum(1 for f in side2_fighter_obs_raw_list if f["alive"])})')
                time.sleep(2)
                break

            # 正常结束检查
            if env.get_done():
                print(f'第 {round_cnt} 轮在第 {step_cnt} 步结束!')

                # get final obs
                side1_obs_raw, side2_obs_raw = env.get_obs_raw()
                side1_detector_obs_raw_list = side1_obs_raw['detector_obs_list']
                side1_fighter_obs_raw_list = side1_obs_raw['fighter_obs_list']
                side2_detector_obs_raw_list = side2_obs_raw['detector_obs_list']
                side2_fighter_obs_raw_list = side2_obs_raw['fighter_obs_list']

                # get reward
                o_detector_reward, o_fighter_reward, o_game_reward, e_detector_reward, e_fighter_reward, e_game_reward = env.get_reward()

                # 统计存活单位
                side1_alive_units = 0
                side2_alive_units = 0

                for y in range(side1_detector_num):
                    if y < len(side1_detector_obs_raw_list) and side1_detector_obs_raw_list[y]['alive']:
                        side1_alive_units += 1
                for y in range(side1_fighter_num):
                    if y < len(side1_fighter_obs_raw_list) and side1_fighter_obs_raw_list[y]['alive']:
                        side1_alive_units += 1

                for y in range(side2_detector_num):
                    if y < len(side2_detector_obs_raw_list) and side2_detector_obs_raw_list[y]['alive']:
                        side2_alive_units += 1
                for y in range(side2_fighter_num):
                    if y < len(side2_fighter_obs_raw_list) and side2_fighter_obs_raw_list[y]['alive']:
                        side2_alive_units += 1

                # determine winner
                if o_game_reward > e_game_reward:
                    print('Side 1 WIN!!!')
                    side1_win_times += 1
                elif o_game_reward < e_game_reward:
                    print('Side 2 WIN!!!')
                    side2_win_times += 1
                else:
                    # 如果奖励相等，根据存活单位数判断
                    if side1_alive_units > side2_alive_units:
                        print('Side 1 WIN!!! (奖励相等但存活单位更多)')
                        side1_win_times += 1
                    elif side2_alive_units > side1_alive_units:
                        print('Side 2 WIN!!! (奖励相等但存活单位更多)')
                        side2_win_times += 1
                    else:
                        print('DRAW!!!')
                        draw_times += 1

                print(f'Side1 总奖励: {side1_total_reward}, Side2 总奖励: {side2_total_reward}')
                print(f'Side1 本轮奖励: {o_game_reward}, Side2 本轮奖励: {e_game_reward}')
                print(
                    f'Side1 存活: 探测器 {sum(1 for d in side1_detector_obs_raw_list if d["alive"])}, 攻击机 {sum(1 for f in side1_fighter_obs_raw_list if f["alive"])}')
                print(
                    f'Side2 存活: 探测器 {sum(1 for d in side2_detector_obs_raw_list if d["alive"])}, 攻击机 {sum(1 for f in side2_fighter_obs_raw_list if f["alive"])}')
                time.sleep(2)
                break

            # 每50步打印一次进度
            if step_cnt % 50 == 0:
                print(f'第 {round_cnt} 轮，第 {step_cnt}/{args.max_step} 步')

    # cleanup
    if not args.tactical_side1 and agent1:
        agent1.terminate()
    if agent2:
        agent2.terminate()

    # final results
    print('\n' + '=' * 60)
    print('对战结果:')
    print('=' * 60)
    print(f'总对战轮数: {round_cnt}. Side1 胜利: {side1_win_times}. Side2 胜利: {side2_win_times}. 平局: {draw_times}')

    if round_cnt > 0:
        print(
            f'Side1 胜率: {side1_win_times / round_cnt * 100:.1f}%, Side2 胜率: {side2_win_times / round_cnt * 100:.1f}%')

    if len(agent1_crash_list) != 0:
        print(f'Side1 崩溃 {len(agent1_crash_list)} 次:')
        print(agent1_crash_list)

    if len(agent2_crash_list) != 0:
        print(f'Side2 崩溃 {len(agent2_crash_list)} 次:')
        print(agent2_crash_list)

    if len(agent1_timeout_list) != 0:
        print(f'Side1 超时 {len(agent1_timeout_list)} 次:')
        print(agent1_timeout_list)

    if len(agent2_timeout_list) != 0:
        print(f'Side2 超时 {len(agent2_timeout_list)} 次:')
        print(agent2_timeout_list)

    print('=' * 60)

    # wait for user to continue
    try:
        input("按<回车>键继续...")
    except:
        pass

    exit(0)


if __name__ == "__main__":
    main()