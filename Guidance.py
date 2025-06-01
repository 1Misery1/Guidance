# guidance_pyqt_mod.py
import heapq
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton, QCheckBox, QSlider, QPushButton, QTextEdit, QFrame, QButtonGroup, QMessageBox, QGroupBox # Import QGroupBox
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Callable
import math
import random
import time

# 设置matplotlib中文字体
plt.rcParams["font.family"] = ["SimHei"]

# NavigationApp 类保持不变，因为它不依赖于GUI框架
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
        # 确保节点存在，如果不存在则使用默认坐标 (0,0) 添加
        self.add_node(node1, *self.node_coordinates.get(node1, (0, 0)))
        self.add_node(node2, *self.node_coordinates.get(node2, (0, 0)))

        self.graph[node1][node2] = weights.copy()
        self.graph[node2][node1] = weights.copy()
        self.road_segments.append((node1, node2, weights))

    def haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """计算两点间的欧几里得距离 (原函数名有误，实际是欧氏距离)"""
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
            return self.haversine_distance(coord_current, coord_goal) # 修正为coord_goal

    def multi_criteria_weight(self, edge_weights: Dict, criteria: Dict[str, float]) -> float:
        """基于多标准计算综合权重"""
        total = 0
        for weight_type, importance in criteria.items():
            if weight_type in edge_weights:
                total += edge_weights[weight_type] * importance
        return total

    def a_star(self, start: str, goal: str, weight_type: str = "distance",
               criteria: Dict[str, float] = None) -> Tuple[List[str], float, int, float]:
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
        f_score[start] = self.heuristic(start, goal, weight_type if not criteria else "distance")

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

    def dijkstra(self, start: str, goal: str, weight_type: str = "distance") -> Tuple[List[str], float, int, float]:
        """Dijkstra算法实现"""
        start_time_exec = time.time()
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

    def greedy_best_first_search(self, start: str, goal: str, weight_type: str = "distance") -> Tuple[
        List[str], float, int, float]:
        """贪婪最佳优先搜索算法实现"""
        start_time_exec = time.time()
        if start not in self.nodes or goal not in self.nodes:
            raise ValueError("起点或终点不在地图中")

        open_set = [(self.heuristic(start, goal, weight_type), start)]
        heapq.heapify(open_set)
        came_from = {}

        path_actual_cost = {node: float('inf') for node in self.nodes}
        path_actual_cost[start] = 0

        closed_set = set()
        expanded_nodes_count = 0

        while open_set:
            _, current_node = heapq.heappop(open_set)
            expanded_nodes_count += 1

            if current_node in closed_set:
                continue
            closed_set.add(current_node)

            if current_node == goal:
                path = [current_node]
                final_cost = 0
                temp_curr = current_node
                path_build_ok = True
                while temp_curr in came_from:
                    prev_node = came_from[temp_curr]
                    edge_weights_map = self.graph.get(prev_node, {}).get(temp_curr)
                    if not edge_weights_map:
                        path_build_ok = False
                        break
                    actual_edge_weight = edge_weights_map.get(weight_type, float('inf'))
                    if actual_edge_weight == float('inf'):
                        path_build_ok = False
                        break
                    final_cost += actual_edge_weight
                    path.append(prev_node)
                    temp_curr = prev_node

                if not path_build_ok:
                    execution_time = time.time() - start_time_exec
                    return [], float('inf'), expanded_nodes_count, execution_time

                execution_time = time.time() - start_time_exec
                return path[::-1], path_actual_cost[goal], expanded_nodes_count, execution_time

            for neighbor, weights in self.graph[current_node].items():
                if neighbor in closed_set:
                    continue

                edge_weight = weights.get(weight_type, float('inf'))
                if edge_weight == float('inf'): continue

                tentative_actual_cost = path_actual_cost[current_node] + edge_weight

                if tentative_actual_cost < path_actual_cost[neighbor]:
                    came_from[neighbor] = current_node
                    path_actual_cost[neighbor] = tentative_actual_cost

                if neighbor not in came_from:
                    came_from[neighbor] = current_node
                    path_actual_cost[neighbor] = tentative_actual_cost

                h_cost = self.heuristic(neighbor, goal, weight_type)
                heapq.heappush(open_set, (h_cost, neighbor))

        execution_time = time.time() - start_time_exec
        return [], float('inf'), expanded_nodes_count, execution_time

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
            "total_weight": total_weight,
            "weight_info": weight_info,
            "expanded_nodes": expanded_nodes,
            "exec_time": exec_time
        }


# NavigationGUI 类重构为 PyQt 版本
class NavigationGUI(QMainWindow):
    def __init__(self, navigation_app):
        super().__init__()
        self.app = navigation_app
        self.selected_start = None
        self.selected_goal = None
        self.current_path = None

        self.setWindowTitle("智能导航系统 (多算法版)")
        self.setGeometry(100, 100, 1920, 1080) # Increased window size for more space

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.create_widgets()
        self.initialize_map()
        self.draw_map()

        self.apply_styles()

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #e0e5ec; /* Softer background */
            }
            QFrame {
                background-color: #ffffff;
                border-radius: 12px; /* More rounded corners */
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); /* Stronger shadow */
                padding: 20px; /* Increased padding */
            }
            QLabel {
                font-size: 15px; /* Slightly larger font */
                font-weight: bold;
                color: #2c3e50; /* Darker, more professional text */
                margin-bottom: 8px; /* More spacing below labels */
            }
            QRadioButton, QCheckBox {
                font-size: 14px;
                color: #34495e;
                padding: 5px 0; /* Increased padding */
            }
            QSlider::groove:horizontal {
                border: 1px solid #aeb6bf; /* Softer border */
                height: 10px; /* Thicker groove */
                background: #d9e0e7; /* Softer background */
                margin: 2px 0;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #27ae60; /* More vibrant green for handle */
                border: 1px solid #1e8449;
                width: 20px; /* Larger handle */
                margin: -6px 0;
                border-radius: 10px;
            }
            QPushButton {
                background-color: #3498db; /* A friendly blue */
                color: white;
                border: none;
                padding: 12px 25px; /* Larger padding */
                font-size: 16px; /* Larger font */
                border-radius: 8px; /* More rounded buttons */
                margin-top: 20px; /* More space above button */
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9; /* Darker blue on hover */
            }
            QTextEdit {
                border: 1px solid #b0c4de; /* Softer border color */
                border-radius: 8px;
                padding: 10px; /* Increased padding */
                font-size: 14px;
                background-color: #f8faff; /* Slightly off-white background */
                color: #2c3e50;
            }
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                border: 1px solid #b0c4de;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* Title centered */
                padding: 0 3px;
                background-color: #e0e5ec;
                border-radius: 3px;
            }
        """)

    def create_widgets(self):
        # 左侧控制面板
        control_frame = QFrame(self) # QFrame 可以提供更好的视觉隔离和样式应用
        control_layout = QVBoxLayout(control_frame) # 使用垂直布局

        # 路径优化目标
        objective_group_box = QGroupBox("路径优化目标") # Use QGroupBox for grouping
        objective_layout = QVBoxLayout(objective_group_box)
        self.weight_group = QButtonGroup(self) # 创建一个按钮组，确保单选
        self.radio_time = QRadioButton("最短时间")
        self.radio_time.setChecked(True) # 默认选中
        self.radio_distance = QRadioButton("最短距离")
        self.radio_traffic = QRadioButton("最少红绿灯")
        self.radio_comprehensive = QRadioButton("综合最优 (仅A*)")

        self.weight_group.addButton(self.radio_time, id=1)
        self.weight_group.addButton(self.radio_distance, id=2)
        self.weight_group.addButton(self.radio_traffic, id=3)
        self.weight_group.addButton(self.radio_comprehensive, id=4)

        objective_layout.addWidget(self.radio_time)
        objective_layout.addWidget(self.radio_distance)
        objective_layout.addWidget(self.radio_traffic)
        objective_layout.addWidget(self.radio_comprehensive)
        control_layout.addWidget(objective_group_box) # Add the group box to the main layout
        control_layout.addSpacing(15) # Add spacing after the group box

        # 算法选择
        algo_group_box = QGroupBox("选择搜索算法") # Use QGroupBox for grouping
        algo_layout = QVBoxLayout(algo_group_box)
        self.algo_group = QButtonGroup(self)
        self.radio_astar = QRadioButton("A* 搜索")
        self.radio_astar.setChecked(True)
        self.radio_dijkstra = QRadioButton("Dijkstra 算法")
        self.radio_greedy_bfs = QRadioButton("贪婪最佳优先搜索")

        self.algo_group.addButton(self.radio_astar, id=1)
        self.algo_group.addButton(self.radio_dijkstra, id=2)
        self.algo_group.addButton(self.radio_greedy_bfs, id=3)

        algo_layout.addWidget(self.radio_astar)
        algo_layout.addWidget(self.radio_dijkstra)
        algo_layout.addWidget(self.radio_greedy_bfs)
        control_layout.addWidget(algo_group_box) # Add the group box to the main layout
        control_layout.addSpacing(15)

        # 约束条件
        constraints_group_box = QGroupBox("约束条件") # Use QGroupBox for grouping
        constraints_layout = QVBoxLayout(constraints_group_box)
        self.check_construction = QCheckBox("避开施工区域")
        self.check_construction.setChecked(True)
        constraints_layout.addWidget(self.check_construction)

        min_width_label = QLabel("最小道路宽度:")
        constraints_layout.addWidget(min_width_label)
        self.slider_min_width = QSlider(QtCore.Qt.Horizontal) # 水平滑动条
        self.slider_min_width.setRange(0, 15) # 设置范围
        self.slider_min_width.setValue(0) # 默认值
        self.slider_min_width.setTickInterval(1) # 设置刻度间隔
        self.slider_min_width.setTickPosition(QSlider.TicksBelow) # 刻度在下方显示
        constraints_layout.addWidget(self.slider_min_width)
        control_layout.addWidget(constraints_group_box) # Add the group box to the main layout
        control_layout.addSpacing(20)

        # 计算路径按钮
        calculate_button = QPushButton("计算路径")
        calculate_button.clicked.connect(self.calculate_path) # 连接按钮的点击信号到槽函数
        control_layout.addWidget(calculate_button)

        # 路径信息显示
        info_group_box = QGroupBox("路径信息") # Use QGroupBox for grouping
        info_layout = QVBoxLayout(info_group_box)
        self.path_info_text = QTextEdit()
        self.path_info_text.setReadOnly(True) # 设置为只读
        self.path_info_text.setMinimumHeight(250) # Increased height
        info_layout.addWidget(self.path_info_text)
        control_layout.addWidget(info_group_box) # Add the group box to the main layout

        control_layout.addStretch(1) # 在底部添加伸展器，使内容靠上对齐

        # 右侧地图可视化区域
        self.figure = Figure(figsize=(16, 9))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_press_event', self.on_map_click) # 连接 Matplotlib 的点击事件

        # 将控制面板和地图添加到主水平布局中
        self.main_layout.addWidget(control_frame, 1) # 控制面板占据1份伸展空间
        self.main_layout.addWidget(self.canvas, 3) # 地图占据3份伸展空间（更大）

    def initialize_map(self):
        # 保持与原始代码相同的地图数据初始化
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
        self.app.add_node("Q", 5, 5)
        self.app.add_node("R", 15, 0)
        self.app.add_node("S", 30, 0)
        self.app.add_node("T", 50, 0)
        self.app.add_node("U", 60, 10)
        self.app.add_node("V", 60, 20)
        self.app.add_node("W", 25, 5)


        # 调整原有边的权重和连接，使其更符合实际情况
        self.app.add_edge("A", "B", distance=10, time=15, traffic_light=1, road_width=8, congestion=1.1, construction=0)
        self.app.add_edge("B", "C", distance=15, time=20, traffic_light=2, road_width=9, congestion=1.3, construction=0)
        self.app.add_edge("C", "D", distance=10, time=12, traffic_light=1, road_width=10, congestion=1.0, construction=0)
        self.app.add_edge("D", "E", distance=10, time=18, traffic_light=1, road_width=7, congestion=1.5, construction=0.3)
        self.app.add_edge("E", "F", distance=10, time=15, traffic_light=0, road_width=11, congestion=1.2, construction=0)

        self.app.add_edge("A", "G", distance=10, time=12, traffic_light=1, road_width=7, congestion=1.0, construction=0)
        self.app.add_edge("G", "H", distance=15, time=25, traffic_light=2, road_width=6, congestion=1.6, construction=0.5)
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
        self.app.add_edge("R", "S", distance=15, time=20, traffic_light=2, road_width=6, congestion=1.4, construction=0.2)
        self.app.add_edge("S", "J", distance=10, time=12, traffic_light=1, road_width=7, congestion=1.2, construction=0)
        self.app.add_edge("S", "W", distance=5, time=6, traffic_light=0, road_width=8, congestion=1.0, construction=0)
        self.app.add_edge("K", "W", distance=5, time=6, traffic_light=0, road_width=8, congestion=1.0, construction=0)
        self.app.add_edge("W", "M", distance=15, time=18, traffic_light=1, road_width=7, congestion=1.3, construction=0)
        self.app.add_edge("S", "T", distance=20, time=25, traffic_light=2, road_width=9, congestion=1.5, construction=0)
        self.app.add_edge("T", "P", distance=10, time=12, traffic_light=1, road_width=8, congestion=1.1, construction=0)
        self.app.add_edge("F", "U", distance=5, time=7, traffic_light=0, road_width=10, congestion=1.0, construction=0)
        self.app.add_edge("U", "V", distance=10, time=12, traffic_light=1, road_width=9, congestion=1.1, construction=0)
        self.app.add_edge("E", "V", distance=5, time=7, traffic_light=0, road_width=8, congestion=1.0, construction=0)

    def draw_map(self):
        self.ax.clear()
        x_coords = [coord[0] for coord in self.app.node_coordinates.values()]
        y_coords = [coord[1] for coord in self.app.node_coordinates.values()]
        if not x_coords or not y_coords:
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
        self.ax.set_facecolor('#f5f5f5') # Softer map background color

        for node1, node2, weights in self.app.road_segments:
            if node1 not in self.app.node_coordinates or node2 not in self.app.node_coordinates: continue
            x1, y1 = self.app.node_coordinates[node1]
            x2, y2 = self.app.node_coordinates[node2]
            road_width = weights.get('road_width', 6)
            line_width = road_width / 2.5 # Slightly thinner lines for clarity
            construction = weights.get('construction', 0)
            congestion = weights.get('congestion', 1.0)
            color = 'grey'
            if construction > 0.3:
                color = '#555555' # Darker grey/black for construction
            else:
                if congestion < 1.2:
                    color = '#28a745' # Bootstrap green
                elif congestion < 1.5:
                    color = '#ffc107' # Bootstrap yellow/orange
                else:
                    color = '#dc3545' # Bootstrap red
            linestyle = '--' if construction > 0.3 else '-'
            self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=line_width, linestyle=linestyle, alpha=0.8)

        for node, (x, y) in self.app.node_coordinates.items():
            has_construction = False
            if node in self.app.graph:
                for neighbor, weights in self.app.graph[node].items():
                    if weights.get('construction', 0) > 0.3:
                        has_construction = True
                        break
            color = '#fd7e14' if has_construction else '#007bff' # Orange for construction, blue for normal
            self.ax.plot(x, y, 'o', color=color, markersize=9, alpha=0.9)
            self.ax.annotate(node, (x + 0.5, y + 0.5), fontsize=10, color='#333333', weight='bold') # Node label styling

        if self.selected_start and self.selected_start in self.app.node_coordinates:
            x, y = self.app.node_coordinates[self.selected_start]
            self.ax.plot(x, y, 'o', color='#28a745', markersize=14, alpha=0.9, label='起点') # Green for start
            self.ax.annotate("起点", (x + 1, y + 1.5), fontsize=12, color='#28a745', weight='bold')
        if self.selected_goal and self.selected_goal in self.app.node_coordinates:
            x, y = self.app.node_coordinates[self.selected_goal]
            self.ax.plot(x, y, 'o', color='#6f42c1', markersize=14, alpha=0.9, label='终点') # Purple for goal
            self.ax.annotate("终点", (x + 1, y + 1.5), fontsize=12, color='#6f42c1', weight='bold')

        if self.current_path and self.current_path['success']:
            path = self.current_path['path']
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                if node1 not in self.app.node_coordinates or node2 not in self.app.node_coordinates: continue
                x1, y1 = self.app.node_coordinates[node1]
                x2, y2 = self.app.node_coordinates[node2]
                self.ax.plot([x1, x2], [y1, y2], '#007bff', linewidth=4, alpha=0.9, zorder=5) # Distinct blue for path, higher zorder

        self.ax.set_title("智能导航地图", fontsize=16, color='#2c3e50', pad=15) # Title styling
        self.ax.grid(False)

        # 重建图例以避免重复并确保所有元素都显示
        handles, labels = self.ax.get_legend_handles_labels()
        static_handles = [
            plt.Line2D([0], [0], color='#28a745', linestyle='-', linewidth=2, label='畅通道路'),
            plt.Line2D([0], [0], color='#ffc107', linestyle='-', linewidth=2, label='缓行道路'),
            plt.Line2D([0], [0], color='#dc3545', linestyle='-', linewidth=2, label='拥堵道路'),
            plt.Line2D([0], [0], color='#555555', linestyle='--', linewidth=2, label='施工道路'),
            plt.Line2D([0], [0], marker='o', color='#007bff', label='正常路口', linestyle='None', markersize=8),
            plt.Line2D([0], [0], marker='o', color='#fd7e14', label='施工路口附近', linestyle='None', markersize=8),
            plt.Line2D([0], [0], color='#007bff', linestyle='-', linewidth=3, label='规划路径')
        ]
        existing_labels = set(labels)
        unique_static_handles = [h for h in static_handles if h.get_label() not in existing_labels]

        self.ax.legend(handles=handles + unique_static_handles,
                       loc='upper left', # Changed legend location
                       bbox_to_anchor=(1.02, 1), # Place legend outside the plot area on the right
                       borderaxespad=0.,
                       fontsize='medium',
                       frameon=True, # Add a frame
                       framealpha=0.9, # Semi-transparent frame
                       edgecolor='#cccccc', # Frame border color
                       facecolor='white' # Frame background color
                       )
        self.figure.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout to make space for the legend
        self.figure.canvas.draw()


    def on_map_click(self, event):
        if event.inaxes != self.ax: return
        x, y = event.xdata, event.ydata
        if x is None or y is None: return

        closest_node = None
        min_distance = float('inf')
        click_threshold = 2.0

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
                if closest_node == self.selected_start:
                    self.selected_start = None
                    print("起点取消选择")
                else:
                    self.selected_goal = closest_node
                    print(f"终点设置为: {self.selected_goal}")
            self.draw_map()

    def calculate_path(self):
        if not self.selected_start or not self.selected_goal:
            QMessageBox.warning(self, "错误", "请先选择起点和终点") # Using QMessageBox.warning
            return

        # 获取路径优化目标的选择
        weight_type_choice = ""
        if self.radio_time.isChecked():
            weight_type_choice = "time"
        elif self.radio_distance.isChecked():
            weight_type_choice = "distance"
        elif self.radio_traffic.isChecked():
            weight_type_choice = "traffic_light"
        elif self.radio_comprehensive.isChecked():
            weight_type_choice = "comprehensive"

        # 获取算法选择
        selected_algorithm = ""
        if self.radio_astar.isChecked():
            selected_algorithm = "a_star"
        elif self.radio_dijkstra.isChecked():
            selected_algorithm = "dijkstra"
        elif self.radio_greedy_bfs.isChecked():
            selected_algorithm = "greedy_bfs"

        criteria = None
        actual_weight_type_for_algo = weight_type_choice

        if weight_type_choice == "comprehensive":
            if selected_algorithm == "a_star":
                criteria = {"time": 0.4, "distance": 0.3, "traffic_light": 0.2, "congestion": 0.1}
                actual_weight_type_for_algo = "time"
            else:
                QMessageBox.warning(self, "提示",
                                    f"{selected_algorithm.upper()} 算法当前不支持“综合最优”。\n将使用“时间”作为优化目标。")
                self.radio_time.setChecked(True) # 强制在GUI上改回“时间”
                actual_weight_type_for_algo = "time"


        # --- 临时图结构，处理约束 ---
        temp_graph = {node: edges.copy() for node, edges in self.app.graph.items()}
        original_graph = self.app.graph # 保存原始图

        # 避开施工
        if self.check_construction.isChecked():
            roads_to_remove = []
            for node, neighbors_data in temp_graph.items():
                for neighbor, weights in list(neighbors_data.items()):
                    if weights.get('construction', 0) > 0.3:
                        roads_to_remove.append((node, neighbor))
            for node1, node2 in roads_to_remove:
                if node1 in temp_graph and node2 in temp_graph[node1]: del temp_graph[node1][node2]
                if node2 in temp_graph and node1 in temp_graph[node2]: del temp_graph[node2][node1]

        # 最小道路宽度
        min_width = self.slider_min_width.value() # 获取滑动条的值
        if min_width > 0:
            roads_to_remove = []
            for node, neighbors_data in temp_graph.items():
                for neighbor, weights in list(neighbors_data.items()):
                    if weights.get('road_width', float('inf')) < min_width:
                        roads_to_remove.append((node, neighbor))
            for node1, node2 in roads_to_remove:
                if node1 in temp_graph and node2 in temp_graph[node1]: del temp_graph[node1][node2]
                if node2 in temp_graph and node1 in temp_graph[node2]: del temp_graph[node2][node1]

        self.app.graph = temp_graph # 应用修改后的图

        try:
            self.current_path = self.app.get_path(
                self.selected_start,
                self.selected_goal,
                algorithm=selected_algorithm,
                weight_type=actual_weight_type_for_algo,
                criteria=criteria
            )
            self.draw_map()
            self.display_path_info()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"路径计算失败: {str(e)}") # Using QMessageBox.critical
            import traceback
            traceback.print_exc() # 打印详细错误信息到控制台
        finally:
            self.app.graph = original_graph # 恢复原始图

    def display_path_info(self):
        self.path_info_text.clear() # PyQt 的 QTextEdit 清空内容使用 .clear()
        if not self.current_path or not self.current_path['success']:
            msg = "未找到有效路径"
            if self.current_path and "message" in self.current_path:
                msg = self.current_path["message"]
            if self.current_path and "algorithm" in self.current_path:
                msg += f"\n(算法: {self.current_path['algorithm'].upper()})"
            self.path_info_text.setText(msg) # PyQt 的 QTextEdit 设置内容使用 .setText()
            return

        path_data = self.current_path
        path = path_data['path']
        weight_info = path_data['weight_info']

        # 提前计算并格式化各项值，避免f-string嵌套问题
        distance_val_formatted = 'N/A'
        if 'distance' in weight_info and not math.isnan(weight_info.get('distance', float('nan'))):
            distance_val_formatted = f"{weight_info['distance']:.2f}"

        time_val_formatted = 'N/A'
        if 'time' in weight_info and not math.isnan(weight_info.get('time', float('nan'))):
            time_val_formatted = f"{weight_info['time']:.2f}"

        traffic_light_val_formatted = 'N/A'
        if 'traffic_light' in weight_info and not math.isnan(weight_info.get('traffic_light', float('nan'))):
            traffic_light_val_formatted = f"{weight_info['traffic_light']:.0f}"


        avg_road_width_val = 'N/A'
        avg_congestion_val = 'N/A'
        avg_construction_val = 'N/A'
        num_segments = len(path) - 1
        if num_segments > 0:
            if 'road_width' in weight_info and not math.isnan(weight_info.get('road_width', float('nan'))):
                avg_road_width_val = f"{weight_info['road_width'] / num_segments:.2f} 米"
            if 'congestion' in weight_info and not math.isnan(weight_info.get('congestion', float('nan'))):
                avg_congestion_val = f"{weight_info['congestion'] / num_segments:.2f}"
            if 'construction' in weight_info and not math.isnan(weight_info.get('construction', float('nan'))):
                avg_construction_val = f"{weight_info['construction'] / num_segments:.2f}"

        info = f"算法: {path_data['algorithm'].upper()}\n"
        info += f"优化目标: {path_data['weight_type']}\n"
        if path_data['criteria']:
            info += f"综合标准: {path_data['criteria']}\n"
        info += f"路径: {' -> '.join(path)}\n\n"

        info += f"总优化成本 ({path_data['weight_type']}): {path_data['total_weight']:.2f}\n"
        info += "-----------------------------\n"
        # 使用预先格式化的值
        info += f"总距离: {distance_val_formatted} 公里\n"
        info += f"预计时间: {time_val_formatted} 分钟\n"
        info += f"红绿灯数量: {traffic_light_val_formatted}\n"
        info += f"平均道路宽度: {avg_road_width_val}\n"
        info += f"平均拥堵系数: {avg_congestion_val}\n"
        info += f"平均施工影响: {avg_construction_val}\n"
        info += "-----------------------------\n"
        info += f"扩展节点数: {path_data.get('expanded_nodes', 'N/A')}\n"
        info += f"执行时间: {path_data.get('exec_time', 'N/A'):.4f} 秒\n"

        self.path_info_text.setText(info)


if __name__ == "__main__":
    app = QApplication(sys.argv) # 创建 QApplication 实例
    nav_app_instance = NavigationApp()
    gui_app = NavigationGUI(nav_app_instance)
    gui_app.show() # 显示主窗口
    sys.exit(app.exec_()) # 运行应用程序的事件循环