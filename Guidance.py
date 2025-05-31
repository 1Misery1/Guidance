# guidance_mod.py
import heapq
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import List, Dict, Tuple, Optional, Callable
import math
import random
import time  # 引入 time 模块用于计时

# 设置matplotlib中文字体
plt.rcParams["font.family"] = ["SimHei"]


class NavigationApp:
    def __init__(self):
        self.graph = {}  # 地图图结构: {节点: {相邻节点: 权重字典}}
        self.nodes = set()  # 所有节点集合
        self.node_coordinates = {}  # 节点坐标: {节点: (x, y)}
        self.road_segments = []  # 道路线段列表

    def add_node(self, node: str, x: float, y: float) -> None:
        if node not in self.graph:
            self.graph[node] = {}
            self.nodes.add(node)
            self.node_coordinates[node] = (x, y)

    def add_edge(self, node1: str, node2: str, **weights) -> None:
        self.add_node(node1, *self.node_coordinates.get(node1, (0, 0)))
        self.add_node(node2, *self.node_coordinates.get(node2, (0, 0)))

        self.graph[node1][node2] = weights.copy()
        self.graph[node2][node1] = weights.copy()
        self.road_segments.append((node1, node2, weights))

    def haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """计算两点间的欧几里得距离"""  # 注意：函数名用的是haversine，但实际是欧氏距离
        x1, y1 = coord1
        x2, y2 = coord2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def heuristic(self, current: str, goal: str, weight_type: str) -> float:
        """启发函数，根据不同权重类型调整"""
        if current not in self.node_coordinates or goal not in self.node_coordinates:
            return float('inf')  # 节点不存在坐标时返回无穷大
        if current == goal:
            return 0

        coord_current = self.node_coordinates[current]
        coord_goal = self.node_coordinates[goal]

        if weight_type == 'time':
            avg_speed = 50
            dist = self.haversine_distance(coord_current, coord_goal)
            return dist / avg_speed if avg_speed > 0 else float('inf')
        elif weight_type == 'traffic_light':
            n = 0.5
            dist = self.haversine_distance(coord_current, coord_goal)
            return dist * n
        else:  # 默认使用距离
            return self.haversine_distance(coord_current, coord_goal)

    def multi_criteria_weight(self, edge_weights: Dict, criteria: Dict[str, float]) -> float:
        """基于多标准计算综合权重"""
        total = 0
        for weight_type, importance in criteria.items():
            if weight_type in edge_weights:
                total += edge_weights[weight_type] * importance
        return total

    def a_star(self, start: str, goal: str, weight_type: str = "distance",
               criteria: Dict[str, float] = None) -> Tuple[List[str], float, int, float]:  # 返回路径，总权重，扩展节点数，执行时间
        """A*算法实现，支持单权重或多标准权重"""
        start_time = time.time()
        if start not in self.nodes or goal not in self.nodes:
            raise ValueError("起点或终点不在地图中")

        open_set = [(0, start)]  # (f_score, node)
        heapq.heapify(open_set)
        came_from = {}
        g_score = {node: float('inf') for node in self.nodes}
        g_score[start] = 0
        f_score = {node: float('inf') for node in self.nodes}
        f_score[start] = self.heuristic(start, goal, weight_type if not criteria else "distance")  # 启发函数通常基于距离或时间

        expanded_nodes_count = 0

        while open_set:
            _, current = heapq.heappop(open_set)
            expanded_nodes_count += 1

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                execution_time = time.time() - start_time
                return path[::-1], g_score[goal], expanded_nodes_count, execution_time

            for neighbor, weights in self.graph[current].items():
                if criteria:
                    weight = self.multi_criteria_weight(weights, criteria)
                else:
                    weight = weights.get(weight_type, float('inf'))

                if weight == float('inf'): continue

                tentative_g_score = g_score[current] + weight
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    heuristic_val = self.heuristic(neighbor, goal, weight_type if not criteria else "distance")
                    f_score[neighbor] = tentative_g_score + heuristic_val
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        execution_time = time.time() - start_time
        return [], float('inf'), expanded_nodes_count, execution_time

    # --- 新增 Dijkstra 算法 ---
    def dijkstra(self, start: str, goal: str, weight_type: str = "distance") -> Tuple[List[str], float, int, float]:
        """Dijkstra算法实现"""
        start_time_exec = time.time()  # 使用不同的变量名以避免与外部start_time冲突
        if start not in self.nodes or goal not in self.nodes:
            raise ValueError("起点或终点不在地图中")

        open_set = [(0, start)]  # (cost, node)
        heapq.heapify(open_set)
        came_from = {}
        g_score = {node: float('inf') for node in self.nodes}
        g_score[start] = 0
        expanded_nodes_count = 0

        while open_set:
            current_g_score, current_node = heapq.heappop(open_set)
            expanded_nodes_count += 1

            if current_g_score > g_score[current_node]:  # 优化：如果已经有更短路，跳过
                continue

            if current_node == goal:
                path = [current_node]
                while current_node in came_from:
                    current_node = came_from[current_node]
                    path.append(current_node)
                execution_time = time.time() - start_time_exec
                return path[::-1], g_score[goal], expanded_nodes_count, execution_time

            for neighbor, weights in self.graph[current_node].items():
                weight = weights.get(weight_type, float('inf'))
                if weight == float('inf'): continue

                tentative_g_score = g_score[current_node] + weight
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    heapq.heappush(open_set, (g_score[neighbor], neighbor))

        execution_time = time.time() - start_time_exec
        return [], float('inf'), expanded_nodes_count, execution_time

    # --- 新增 贪婪最佳优先搜索 ---
    def greedy_best_first_search(self, start: str, goal: str, weight_type: str = "distance") -> Tuple[
        List[str], float, int, float]:
        """贪婪最佳优先搜索算法实现"""
        start_time_exec = time.time()
        if start not in self.nodes or goal not in self.nodes:
            raise ValueError("起点或终点不在地图中")

        # (heuristic_cost, node)
        # 启发函数使用 weight_type 以便与 A* 的启发函数保持一致性
        open_set = [(self.heuristic(start, goal, weight_type), start)]
        heapq.heapify(open_set)
        came_from = {}  # 用于路径回溯

        # g_score 用于计算最终路径的实际成本，不参与优先队列决策
        path_actual_cost = {node: float('inf') for node in self.nodes}
        path_actual_cost[start] = 0

        # 记录已访问节点以避免简单环路 (对于非最优算法这很重要)
        # 对于GBFS，一旦一个节点被扩展(从open_set中取出)，通常不再重新考虑它，因为它不保证找到最优路径。
        closed_set = set()
        expanded_nodes_count = 0

        while open_set:
            _, current_node = heapq.heappop(open_set)
            expanded_nodes_count += 1

            if current_node in closed_set:  # 如果节点已经被处理过，跳过
                continue
            closed_set.add(current_node)

            if current_node == goal:
                path = [current_node]
                # 计算实际路径成本
                final_cost = 0
                temp_curr = current_node
                path_build_ok = True
                while temp_curr in came_from:
                    prev_node = came_from[temp_curr]
                    edge_weights_map = self.graph.get(prev_node, {}).get(temp_curr)
                    if not edge_weights_map:  # 路径构建中出现问题
                        path_build_ok = False
                        break
                    actual_edge_weight = edge_weights_map.get(weight_type, float('inf'))
                    if actual_edge_weight == float('inf'):
                        path_build_ok = False
                        break
                    final_cost += actual_edge_weight
                    path.append(prev_node)  # 先加入，再反转
                    temp_curr = prev_node

                if not path_build_ok:  # 如果路径成本计算有问题
                    execution_time = time.time() - start_time_exec
                    return [], float('inf'), expanded_nodes_count, execution_time  # 或者返回已构建的部分路径和成本

                execution_time = time.time() - start_time_exec
                return path[::-1], path_actual_cost[goal], expanded_nodes_count, execution_time

            for neighbor, weights in self.graph[current_node].items():
                if neighbor in closed_set:  # 不考虑已经处理过的邻居
                    continue

                # 计算到邻居的实际成本（用于最终路径成本计算，不用于GBFS决策）
                edge_weight = weights.get(weight_type, float('inf'))
                if edge_weight == float('inf'): continue

                tentative_actual_cost = path_actual_cost[current_node] + edge_weight

                # GBFS通常不关心g_score的更新，但为了能回溯路径和计算总成本，我们需要记录
                # 如果通过当前路径到邻居的实际成本更低（或首次到达）
                if tentative_actual_cost < path_actual_cost[neighbor]:
                    came_from[neighbor] = current_node
                    path_actual_cost[neighbor] = tentative_actual_cost

                # 如果came_from[neighbor]不存在，也记录，因为GBFS可能选择非最优路径
                if neighbor not in came_from:
                    came_from[neighbor] = current_node
                    path_actual_cost[neighbor] = tentative_actual_cost  # 记录成本

                h_cost = self.heuristic(neighbor, goal, weight_type)
                # 确保邻居不在closed_set中，并且不再open_set中或在open_set中有更高h_cost (标准GBFS一般不这么复杂)
                # 简单起见，只要不在closed_set就加入open_set
                heapq.heappush(open_set, (h_cost, neighbor))

        execution_time = time.time() - start_time_exec
        return [], float('inf'), expanded_nodes_count, execution_time

    # --- 修改 get_path ---
    def get_path(self, start: str, goal: str, algorithm: str = "a_star",
                 weight_type: str = "distance", criteria: Dict[str, float] = None) -> Dict:
        """获取规划路径，支持多种算法和权重类型"""
        path = []
        total_weight = float('inf')
        expanded_nodes = 0
        exec_time = 0.0

        if algorithm == "a_star":
            path, total_weight, expanded_nodes, exec_time = self.a_star(start, goal, weight_type, criteria)
        elif algorithm == "dijkstra":
            if criteria:
                print("警告: Dijkstra 算法在此实现中不支持多标准权重，将使用单一 weight_type。")
            path, total_weight, expanded_nodes, exec_time = self.dijkstra(start, goal, weight_type)
        elif algorithm == "greedy_bfs":
            if criteria:
                print("警告: 贪婪最佳优先搜索在此实现中不支持多标准权重，将使用单一 weight_type。")
            path, total_weight, expanded_nodes, exec_time = self.greedy_best_first_search(start, goal, weight_type)
        else:
            return {"success": False, "message": f"未知算法: {algorithm}", "path": [], "total_weight": 0,
                    "expanded_nodes": 0, "exec_time": 0.0}

        if not path:
            return {"success": False, "message": "未找到路径", "algorithm": algorithm, "path": [], "total_weight": 0,
                    "expanded_nodes": expanded_nodes, "exec_time": exec_time}

        weight_info = {}
        if path:
            first_node_in_path = path[0]
            if len(path) > 1 and first_node_in_path in self.graph and path[1] in self.graph[first_node_in_path]:
                first_edge_weights_keys = self.graph[first_node_in_path][path[1]].keys()
            elif len(path) == 1 and start == goal:  # 起点终点相同
                # 尝试从图中获取任意一条边的权重类型作为模板
                any_node = next(iter(self.graph), None)
                if any_node and self.graph[any_node]:
                    any_neighbor = next(iter(self.graph[any_node]))
                    first_edge_weights_keys = self.graph[any_node][any_neighbor].keys()
                else:  # 图为空或节点无边
                    first_edge_weights_keys = []
            else:  # 路径只有一个节点但不是起点终点相同，或路径无效
                first_edge_weights_keys = []

            for w_type in first_edge_weights_keys:
                w = 0
                for i in range(len(path) - 1):
                    node1, node2 = path[i], path[i + 1]
                    if node1 in self.graph and node2 in self.graph[node1]:
                        w += self.graph[node1][node2].get(w_type, 0)
                    else:  # 路径片段无效
                        w = float('nan')  # 表示数据有问题
                        break
                weight_info[w_type] = w
            if len(path) == 1 and start == goal:  # 对于起点终点相同的情况，所有权重为0
                for w_type in first_edge_weights_keys:
                    weight_info[w_type] = 0

        return {
            "success": True,
            "algorithm": algorithm,
            "weight_type": weight_type if not criteria else "comprehensive",
            "criteria": criteria,
            "path": path,
            "total_weight": total_weight,  # 这是算法优化目标的总权重
            "weight_info": weight_info,
            "expanded_nodes": expanded_nodes,
            "exec_time": exec_time
        }


class NavigationGUI:
    def __init__(self, root, navigation_app):
        self.root = root
        self.root.title("智能导航系统 (多算法版)")
        self.root.geometry("1200x850")  # 稍微增加高度以容纳新控件

        self.app = navigation_app
        self.selected_start = None
        self.selected_goal = None
        self.current_path = None

        self.create_widgets()
        self.initialize_map()  # 使用与原代码相同的地图数据
        self.draw_map()

    def create_widgets(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False)

        # --- 路径优化目标 ---
        ttk.Label(control_frame, text="路径优化目标:").pack(anchor=tk.W, pady=5)
        self.weight_var = tk.StringVar(value="time")
        ttk.Radiobutton(control_frame, text="最短时间", variable=self.weight_var, value="time").pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="最短距离", variable=self.weight_var, value="distance").pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="最少红绿灯", variable=self.weight_var, value="traffic_light").pack(
            anchor=tk.W)
        ttk.Radiobutton(control_frame, text="综合最优 (仅A*)", variable=self.weight_var, value="comprehensive").pack(
            anchor=tk.W)

        # --- 新增：算法选择 ---
        ttk.Label(control_frame, text="选择搜索算法:").pack(anchor=tk.W, pady=(10, 5))
        self.algorithm_var = tk.StringVar(value="a_star")  # 默认 A*
        ttk.Radiobutton(control_frame, text="A* 搜索", variable=self.algorithm_var, value="a_star").pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Dijkstra 算法", variable=self.algorithm_var, value="dijkstra").pack(
            anchor=tk.W)
        ttk.Radiobutton(control_frame, text="贪婪最佳优先搜索", variable=self.algorithm_var, value="greedy_bfs").pack(
            anchor=tk.W)

        # --- 约束条件 ---
        ttk.Label(control_frame, text="约束条件:").pack(anchor=tk.W, pady=5)
        self.avoid_construction_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="避开施工区域", variable=self.avoid_construction_var).pack(anchor=tk.W)
        ttk.Label(control_frame, text="最小道路宽度:").pack(anchor=tk.W)
        self.min_width_var = tk.DoubleVar(value=0)
        ttk.Scale(control_frame, variable=self.min_width_var, from_=0, to=15, orient=tk.HORIZONTAL, length=200).pack(
            fill=tk.X)

        ttk.Button(control_frame, text="计算路径", command=self.calculate_path).pack(pady=20)

        ttk.Label(control_frame, text="路径信息:").pack(anchor=tk.W, pady=5)
        self.path_info_text = tk.Text(control_frame, height=20, width=35)  # 增加文本框宽度和高度
        self.path_info_text.pack(fill=tk.BOTH, expand=True)

        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_map_click)

    def initialize_map(self):
        # 调整原有节点坐标，使其更分散，避免重叠
        self.app.add_node("A", 0, 20)
        self.app.add_node("B", 10, 20)
        self.app.add_node("C", 25, 20)
        self.app.add_node("D", 35, 20)
        self.app.add_node("E", 45, 20)
        self.app.add_node("F", 55, 20)
        self.app.add_node("G", 0, 10)
        self.app.add_node("H", 15, 10)
        self.app.add_node("I", 10, 15)
        self.app.add_node("J", 30, 10)
        self.app.add_node("K", 20, 15)
        self.app.add_node("M", 40, 5)
        self.app.add_node("N", 37.5, 12.5)
        self.app.add_node("O", 45, 15)
        self.app.add_node("P", 50, 10)

        # 增加新节点
        self.app.add_node("Q", 5, 5)  # 新增节点 Q
        self.app.add_node("R", 15, 0)  # 新增节点 R
        self.app.add_node("S", 30, 0)  # 新增节点 S
        self.app.add_node("T", 50, 0)  # 新增节点 T
        self.app.add_node("U", 60, 10) # 新增节点 U (在F右侧，P右侧)
        self.app.add_node("V", 60, 20) # 新增节点 V (在F右侧，E右侧)
        self.app.add_node("W", 25, 5) # 新增节点 W (在K下方，S左侧)


        # 调整原有边的权重和连接，使其更符合实际情况
        self.app.add_edge("A", "B", distance=10, time=15, traffic_light=1, road_width=8, congestion=1.1, construction=0)
        self.app.add_edge("B", "C", distance=15, time=20, traffic_light=2, road_width=9, congestion=1.3, construction=0)
        self.app.add_edge("C", "D", distance=10, time=12, traffic_light=1, road_width=10, congestion=1.0, construction=0)
        self.app.add_edge("D", "E", distance=10, time=18, traffic_light=1, road_width=7, congestion=1.5, construction=0.3) # 施工
        self.app.add_edge("E", "F", distance=10, time=15, traffic_light=0, road_width=11, congestion=1.2, construction=0)

        self.app.add_edge("A", "G", distance=10, time=12, traffic_light=1, road_width=7, congestion=1.0, construction=0)
        self.app.add_edge("G", "H", distance=15, time=25, traffic_light=2, road_width=6, congestion=1.6, construction=0.5) # 施工
        self.app.add_edge("H", "J", distance=15, time=20, traffic_light=1, road_width=8, congestion=1.2, construction=0)
        self.app.add_edge("H", "I", distance=5, time=8, traffic_light=0, road_width=9, congestion=1.0, construction=0)

        self.app.add_edge("I", "B", distance=8, time=10, traffic_light=1, road_width=7, congestion=1.2, construction=0)
        self.app.add_edge("I", "K", distance=10, time=12, traffic_light=1, road_width=6, congestion=1.1, construction=0)
        self.app.add_edge("J", "K", distance=10, time=10, traffic_light=0, road_width=8, congestion=1.0, construction=0)
        self.app.add_edge("K", "C", distance=8, time=9, traffic_light=0, road_width=9, congestion=1.0, construction=0)

        self.app.add_edge("J", "M", distance=12, time=15, traffic_light=1, road_width=7, congestion=1.3, construction=0)
        self.app.add_edge("J", "N", distance=10, time=10, traffic_light=0, road_width=9, congestion=1.0, construction=0)
        self.app.add_edge("D", "N", distance=8, time=9, traffic_light=0, road_width=8, congestion=1.1, construction=0)
        self.app.add_edge("N", "O", distance=7, time=8, traffic_light=0, road_width=9, congestion=1.0, construction=0)
        self.app.add_edge("M", "O", distance=10, time=12, traffic_light=1, road_width=8, congestion=1.2, construction=0)
        self.app.add_edge("O", "P", distance=8, time=10, traffic_light=0, road_width=9, congestion=1.0, construction=0)
        self.app.add_edge("P", "E", distance=7, time=8, traffic_light=0, road_width=8, congestion=1.0, construction=0)
        self.app.add_edge("P", "F", distance=5, time=6, traffic_light=0, road_width=10, congestion=1.0, construction=0)

        # 增加新路段连接新旧节点
        self.app.add_edge("G", "Q", distance=5, time=7, traffic_light=0, road_width=6, congestion=1.0, construction=0)
        self.app.add_edge("Q", "R", distance=10, time=12, traffic_light=1, road_width=7, congestion=1.2, construction=0)
        self.app.add_edge("R", "H", distance=8, time=10, traffic_light=0, road_width=8, congestion=1.1, construction=0)
        self.app.add_edge("R", "S", distance=15, time=20, traffic_light=2, road_width=6, congestion=1.4, construction=0.2) # 施工
        self.app.add_edge("S", "J", distance=10, time=12, traffic_light=1, road_width=7, congestion=1.2, construction=0)
        self.app.add_edge("S", "W", distance=5, time=6, traffic_light=0, road_width=8, congestion=1.0, construction=0) # S连接W
        self.app.add_edge("K", "W", distance=5, time=6, traffic_light=0, road_width=8, congestion=1.0, construction=0) # K连接W
        self.app.add_edge("W", "M", distance=15, time=18, traffic_light=1, road_width=7, congestion=1.3, construction=0) # W连接M
        self.app.add_edge("S", "T", distance=20, time=25, traffic_light=2, road_width=9, congestion=1.5, construction=0)
        self.app.add_edge("T", "P", distance=10, time=12, traffic_light=1, road_width=8, congestion=1.1, construction=0)
        self.app.add_edge("F", "U", distance=5, time=7, traffic_light=0, road_width=10, congestion=1.0, construction=0)
        self.app.add_edge("U", "V", distance=10, time=12, traffic_light=1, road_width=9, congestion=1.1, construction=0)
        self.app.add_edge("E", "V", distance=5, time=7, traffic_light=0, road_width=8, congestion=1.0, construction=0)

    def draw_map(self):
        self.ax.clear()
        x_coords = [coord[0] for coord in self.app.node_coordinates.values()]
        y_coords = [coord[1] for coord in self.app.node_coordinates.values()]
        if not x_coords or not y_coords:  # 处理空图的情况
            self.ax.set_title("智能导航地图 (无数据)")
            self.figure.canvas.draw()
            return

        x_min, x_max = min(x_coords) - 5, max(x_coords) + 5
        y_min, y_max = min(y_coords) - 5, max(y_coords) + 5
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

        # 移除坐标轴刻度及标签
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")

        for node1, node2, weights in self.app.road_segments:
            if node1 not in self.app.node_coordinates or node2 not in self.app.node_coordinates: continue
            x1, y1 = self.app.node_coordinates[node1]
            x2, y2 = self.app.node_coordinates[node2]
            road_width = weights.get('road_width', 6)
            line_width = road_width / 2
            construction = weights.get('construction', 0)
            congestion = weights.get('congestion', 1.0)
            color = 'grey'  # Default color
            if construction > 0.3:
                color = 'black'
            else:
                if congestion < 1.2:
                    color = 'green'
                elif congestion < 1.5:
                    color = 'yellow'
                else:
                    color = 'red'  # congestion > 1.5 or equal
            linestyle = '--' if construction > 0.3 else '-'
            self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=line_width, linestyle=linestyle, alpha=0.7)

        for node, (x, y) in self.app.node_coordinates.items():
            has_construction = False
            if node in self.app.graph:  # 确保节点在图中
                for neighbor, weights in self.app.graph[node].items():
                    if weights.get('construction', 0) > 0.3:
                        has_construction = True
                        break
            color = 'orange' if has_construction else 'blue'
            self.ax.plot(x, y, 'o', color=color, markersize=8)
            self.ax.annotate(node, (x + 0.5, y + 0.5), fontsize=10)

        if self.selected_start and self.selected_start in self.app.node_coordinates:
            x, y = self.app.node_coordinates[self.selected_start]
            self.ax.plot(x, y, 'o', color='green', markersize=12, alpha=0.8, label='起点')  # Add label for legend
            self.ax.annotate("起点", (x + 1, y + 1), fontsize=12, color='green')
        if self.selected_goal and self.selected_goal in self.app.node_coordinates:
            x, y = self.app.node_coordinates[self.selected_goal]
            self.ax.plot(x, y, 'o', color='purple', markersize=12, alpha=0.8,
                         label='终点')  # Changed color for distinction, add label
            self.ax.annotate("终点", (x + 1, y + 1), fontsize=12, color='purple')

        if self.current_path and self.current_path['success']:
            path = self.current_path['path']
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                if node1 not in self.app.node_coordinates or node2 not in self.app.node_coordinates: continue
                x1, y1 = self.app.node_coordinates[node1]
                x2, y2 = self.app.node_coordinates[node2]
                self.ax.plot([x1, x2], [y1, y2], 'r-', linewidth=3, alpha=0.9)  # Planning path in red

        self.ax.set_title("智能导航地图")
        # self.ax.set_xlabel("X坐标") # 移除
        # self.ax.set_ylabel("Y坐标") # 移除
        self.ax.grid(False) # 通常删除坐标轴后，网格线也一并删除会更简洁

        # Rebuild legend to avoid duplicates and ensure all elements are shown
        handles, labels = self.ax.get_legend_handles_labels()
        # Add static legend items if not already present from plotted data
        static_handles = [
            plt.Line2D([0], [0], color='green', linestyle='-', linewidth=2, label='畅通道路'),
            plt.Line2D([0], [0], color='yellow', linestyle='-', linewidth=2, label='缓行道路'),
            plt.Line2D([0], [0], color='red', linestyle='-', linewidth=2, label='拥堵道路'),
            plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='施工道路'),
            plt.Line2D([0], [0], marker='o', color='blue', label='正常路口', linestyle='None', markersize=8),
            plt.Line2D([0], [0], marker='o', color='orange', label='施工路口附近', linestyle='None', markersize=8),
            plt.Line2D([0], [0], color='red', linestyle='-', linewidth=3, label='规划路径')
        ]
        # Filter out existing labels from static ones to prevent duplicates
        existing_labels = set(labels)
        unique_static_handles = [h for h in static_handles if h.get_label() not in existing_labels]

        self.ax.legend(handles=handles + unique_static_handles, loc='upper right', fontsize='small')
        self.figure.canvas.draw()

    def on_map_click(self, event):
        if event.inaxes != self.ax: return
        x, y = event.xdata, event.ydata
        if x is None or y is None: return  # Clicked outside axes bounds

        closest_node = None
        min_distance = float('inf')
        click_threshold = 2.0  # 增加点击阈值

        for node, (nx, ny) in self.app.node_coordinates.items():
            distance = math.sqrt((nx - x) ** 2 + (ny - y) ** 2)
            if distance < min_distance and distance < click_threshold:
                min_distance = distance
                closest_node = node

        if closest_node:
            if not self.selected_start or (self.selected_start and self.selected_goal):
                self.selected_start = closest_node
                self.selected_goal = None
                print(f"起点设置为: {self.selected_start}")
            elif self.selected_start and not self.selected_goal:
                if closest_node == self.selected_start:  # 如果重复点击起点，则取消选择
                    self.selected_start = None
                    print("起点取消选择")
                else:
                    self.selected_goal = closest_node
                    print(f"终点设置为: {self.selected_goal}")
            self.draw_map()

    def calculate_path(self):
        if not self.selected_start or not self.selected_goal:
            messagebox.showerror("错误", "请先选择起点和终点")
            return

        weight_type_choice = self.weight_var.get()
        selected_algorithm = self.algorithm_var.get()  # 获取选择的算法

        criteria = None
        actual_weight_type_for_algo = weight_type_choice  # 默认算法使用的权重类型

        if weight_type_choice == "comprehensive":
            if selected_algorithm == "a_star":
                criteria = {"time": 0.4, "distance": 0.3, "traffic_light": 0.2, "congestion": 0.1}
                # A*的启发函数在使用comprehensive时，仍需要一个基础类型，比如time或distance
                # 或者在A*内部处理，如果criteria不为None，启发函数固定为某种类型
                actual_weight_type_for_algo = "time"  # 或者 "distance"
            else:
                messagebox.showwarning("提示",
                                       f"{selected_algorithm.upper()} 算法当前不支持“综合最优”。\n将使用“{actual_weight_type_for_algo}”作为优化目标。")
                # 如果不是A*但选了comprehensive，则回退到选定的单一目标，或者一个默认值
                # 这里我们让 actual_weight_type_for_algo 保持为 weight_var 的值 (如果不是comprehensive)
                # 或者强制设为一个默认值，比如 "time"
                if self.weight_var.get() == "comprehensive":  # 如果确实是comprehensive
                    self.weight_var.set("time")  # GUI上改回time
                    actual_weight_type_for_algo = "time"

        # --- 临时图结构，处理约束 (与原代码一致) ---
        temp_graph = {node: edges.copy() for node, edges in self.app.graph.items()}
        original_graph = self.app.graph  # 保存原始图

        # 避开施工
        if self.avoid_construction_var.get():
            roads_to_remove = []
            for node, neighbors_data in temp_graph.items():
                for neighbor, weights in list(neighbors_data.items()):  # 使用list()复制以允许修改
                    if weights.get('construction', 0) > 0.3:
                        roads_to_remove.append((node, neighbor))
            for node1, node2 in roads_to_remove:
                if node1 in temp_graph and node2 in temp_graph[node1]: del temp_graph[node1][node2]
                if node2 in temp_graph and node1 in temp_graph[node2]: del temp_graph[node2][node1]

        # 最小道路宽度
        min_width = self.min_width_var.get()
        if min_width > 0:
            roads_to_remove = []
            for node, neighbors_data in temp_graph.items():
                for neighbor, weights in list(neighbors_data.items()):
                    if weights.get('road_width', float('inf')) < min_width:
                        roads_to_remove.append((node, neighbor))
            for node1, node2 in roads_to_remove:
                if node1 in temp_graph and node2 in temp_graph[node1]: del temp_graph[node1][node2]
                if node2 in temp_graph and node1 in temp_graph[node2]: del temp_graph[node2][node1]

        self.app.graph = temp_graph  # 应用修改后的图

        try:
            self.current_path = self.app.get_path(
                self.selected_start,
                self.selected_goal,
                algorithm=selected_algorithm,  # 传递算法
                weight_type=actual_weight_type_for_algo,  # 传递给算法的实际单一权重类型
                criteria=criteria  # 传递多标准（主要给A*）
            )
            self.draw_map()
            self.display_path_info()
        except Exception as e:
            messagebox.showerror("错误", f"路径计算失败: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印详细错误信息到控制台
        finally:
            self.app.graph = original_graph  # 恢复原始图

    def display_path_info(self):
        self.path_info_text.delete(1.0, tk.END)
        if not self.current_path or not self.current_path['success']:
            msg = "未找到有效路径"
            if self.current_path and "message" in self.current_path:
                msg = self.current_path["message"]
            if self.current_path and "algorithm" in self.current_path:
                msg += f"\n(算法: {self.current_path['algorithm'].upper()})"
            self.path_info_text.insert(tk.END, msg)
            return

        path_data = self.current_path
        path = path_data['path']
        weight_info = path_data['weight_info']

        # 确保weight_info不为空
        avg_road_width_val = 'N/A'
        avg_congestion_val = 'N/A'
        avg_construction_val = 'N/A'
        num_segments = len(path) - 1
        if num_segments > 0:
            if 'road_width' in weight_info and not math.isnan(weight_info['road_width']):
                avg_road_width_val = f"{weight_info['road_width'] / num_segments:.2f} 米"
            if 'congestion' in weight_info and not math.isnan(weight_info['congestion']):
                avg_congestion_val = f"{weight_info['congestion'] / num_segments:.2f}"
            if 'construction' in weight_info and not math.isnan(weight_info['construction']):
                avg_construction_val = f"{weight_info['construction'] / num_segments:.2f}"

        info = f"算法: {path_data['algorithm'].upper()}\n"
        info += f"优化目标: {path_data['weight_type']}\n"
        if path_data['criteria']:
            info += f"综合标准: {path_data['criteria']}\n"
        info += f"路径: {' -> '.join(path)}\n\n"

        info += f"总优化成本 ({path_data['weight_type']}): {path_data['total_weight']:.2f}\n"
        info += "-----------------------------\n"
        info += f"总距离: {weight_info.get('distance', 'N/A') if isinstance(weight_info.get('distance'), str) else (f'{weight_info.get("distance", 0):.2f}' if not math.isnan(weight_info.get('distance', 0)) else 'N/A')} 公里\n"
        info += f"预计时间: {weight_info.get('time', 'N/A') if isinstance(weight_info.get('time'), str) else (f'{weight_info.get("time", 0):.2f}' if not math.isnan(weight_info.get('time', 0)) else 'N/A')} 分钟\n"
        info += f"红绿灯数量: {weight_info.get('traffic_light', 'N/A') if isinstance(weight_info.get('traffic_light'), str) else (f'{weight_info.get("traffic_light", 0):.0f}' if not math.isnan(weight_info.get('traffic_light', 0)) else 'N/A')}\n"
        info += f"平均道路宽度: {avg_road_width_val}\n"
        info += f"平均拥堵系数: {avg_congestion_val}\n"
        info += f"平均施工影响: {avg_construction_val}\n"
        info += "-----------------------------\n"
        info += f"扩展节点数: {path_data.get('expanded_nodes', 'N/A')}\n"
        info += f"执行时间: {path_data.get('exec_time', 'N/A'):.4f} 秒\n"

        self.path_info_text.insert(tk.END, info)


if __name__ == "__main__":
    nav_app_instance = NavigationApp()
    root_tk = tk.Tk()
    gui_app = NavigationGUI(root_tk, nav_app_instance)
    root_tk.mainloop()