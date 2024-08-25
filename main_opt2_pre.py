import os
import math
import random
import matplotlib.pyplot as plt


# 计算两个城市之间的欧几里得距离，防止分母为零
def euclidean_distance(city1, city2):
    epsilon = 1e-10  # 一个很小的常数，避免距离为零
    return math.sqrt((city1[1] - city2[1]) ** 2 + (city1[2] - city2[2]) ** 2) + epsilon


# 读取TSP文件并返回城市信息列表和距离矩阵
def read_tsp_file(file_path):
    cities = []
    distance_matrix = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # 读取标头信息，直到找到 'NODE_COORD_SECTION'
        start_reading = False
        num_cities = 0

        for line in lines:
            line = line.strip()
            if line.startswith('DIMENSION'):
                num_cities = int(line.split(':')[1].strip())
                distance_matrix = [[0] * num_cities for _ in range(num_cities)]
            elif line.startswith('NODE_COORD_SECTION'):
                start_reading = True
                continue

            if start_reading:
                if line.strip() == 'EOF':
                    break
                parts = line.split()
                # 确保每一行都是正确的格式
                if len(parts) == 3:
                    try:
                        city_index = int(parts[0]) - 1
                        x = float(parts[1])
                        y = float(parts[2])
                        cities.append((city_index, x, y))
                    except ValueError:
                        print(f"忽略无效行: {line.strip()}")

        # 计算距离矩阵
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    distance_matrix[i][j] = euclidean_distance(cities[i], cities[j])

    return cities, distance_matrix


# 蚁群算法来解决TSP问题
def ant_colony_optimization(distance_matrix, num_ants, num_iterations, alpha=1.0, beta=6.5, evaporation_rate=0.1,
                            Q=100):
    """
    蚁群算法实现函数
    :param distance_matrix: 城市距离矩阵
    :param num_ants: 蚂蚁数量
    :param num_iterations: 迭代次数
    :param alpha: α参数，信息素因子
    :param beta: β参数，启发函数因子
    :param evaporation_rate: 信息素挥发率
    :param Q: 信息素增强因子
    :return: 最佳路线和长度, 每一轮的最佳距离列表, 每一轮的平均距离列表
    """
    # 初始化信息素矩阵，并根据边的长度进行调整
    num_cities = len(distance_matrix)
    pheromone = [[0 for _ in range(num_cities)] for _ in range(num_cities)]
    max_distance = max(max(row) for row in distance_matrix)  # 获取最大距离
    min_distance = min(min(row) for row in distance_matrix if row)  # 获取最小距离

    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            # 根据距离对信息素初始值进行调整，距离越短，信息素初始值略多
            distance = distance_matrix[i][j]
            if distance == 0:
                pheromone[i][j] = pheromone[j][i] = 1.0 / (num_cities * num_cities)  # 防止零距离
            else:
                pheromone[i][j] = pheromone[j][i] = 1.0 / (num_cities * num_cities)
                pheromone[i][j] += (max_distance - distance) / (max_distance - min_distance + 1e-10) * 0.1

    best_tour = None
    best_distance = float('inf')
    best_distances = []  # 用于记录每一轮的最佳距离
    average_distances = []  # 用于记录每一轮的平均距离

    for iteration in range(num_iterations):
        all_tours = []
        all_distances = []

        for ant in range(num_ants):
            visited = set()
            current_city = 0
            tour = [current_city]
            visited.add(current_city)

            while len(visited) < num_cities:
                probabilities = []
                total_probability = 0.0

                for next_city in range(num_cities):
                    if next_city not in visited:
                        pheromone_strength = pheromone[current_city][next_city] ** alpha
                        distance_impact = (1 / distance_matrix[current_city][next_city]) ** beta
                        prob = pheromone_strength * distance_impact
                        probabilities.append((next_city, prob))
                        total_probability += prob

                if total_probability == 0:
                    next_city = random.choice([city for city in range(num_cities) if city not in visited])
                else:
                    r = random.random() * total_probability
                    cumulative_prob = 0.0
                    for next_city, prob in probabilities:
                        cumulative_prob += prob
                        if r < cumulative_prob:
                            break

                tour.append(next_city)
                visited.add(next_city)
                current_city = next_city

            total_distance = sum(distance_matrix[tour[i]][tour[i + 1]] for i in range(num_cities - 1))
            total_distance += distance_matrix[tour[-1]][tour[0]]

            all_tours.append(tour)
            all_distances.append(total_distance)

            if total_distance < best_distance:
                best_tour = tour
                best_distance = total_distance

        # 在每一轮迭代中使用2-opt优化最佳路径
        best_tour, best_distance = two_opt(best_tour, distance_matrix)
        best_distances.append(best_distance)  # 记录每一轮的最佳距离
        average_distances.append(sum(all_distances) / len(all_distances))  # 记录每一轮的平均距离

        # 信息素更新
        pheromone = [[(1 - evaporation_rate) * pheromone[i][j] for j in range(num_cities)] for i in range(num_cities)]

        for tour, distance in zip(all_tours, all_distances):
            for i in range(num_cities - 1):
                pheromone[tour[i]][tour[i + 1]] += Q / distance
            pheromone[tour[-1]][tour[0]] += Q / distance

    return best_tour, best_distance, best_distances, average_distances


# 2-opt算法来优化路径
def two_opt(tour, distance_matrix):
    """
    2-opt优化算法
    :param tour: 初始路径
    :param distance_matrix: 城市距离矩阵
    :return: 优化后的路径和总距离
    """
    def calculate_total_distance(tour):
        return sum(distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)) + distance_matrix[tour[-1]][tour[0]]

    def reverse_segment(tour, i, k):
        while i < k:
            tour[i], tour[k] = tour[k], tour[i]
            i += 1
            k -= 1

    best_tour = tour[:]
    best_distance = calculate_total_distance(best_tour)
    improvement = True

    while improvement:
        improvement = False
        for i in range(1, len(best_tour) - 1):
            for k in range(i + 1, len(best_tour)):
                if k - i == 1:
                    continue
                new_tour = best_tour[:]
                reverse_segment(new_tour, i, k - 1)
                new_distance = calculate_total_distance(new_tour)
                if new_distance < best_distance:
                    best_tour = new_tour
                    best_distance = new_distance
                    improvement = True

    return best_tour, best_distance


# 绘制收敛程度图
def plot_convergence(best_distances, average_distances, file_name, output_folder):
    """
    绘制收敛程度图
    :param best_distances: 每一轮的最佳距离列表
    :param average_distances: 每一轮的平均距离列表
    :param file_name: 文件名
    :param output_folder: 输出文件夹
    """
    plt.figure(figsize=(12, 6))
    plt.plot(best_distances, marker='o', linestyle='-', color='b', label='Best Distance')
    plt.plot(average_distances, marker='x', linestyle='--', color='r', label='Average Distance')
    plt.title(f'Convergence of Ant Colony Optimization ({file_name})')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f'{file_name}.png'))
    plt.close()


# 处理输入文件夹中的所有TSP文件
def process_tsp_files(input_folder, output_folder):
    tsp_folder = os.path.join(input_folder, "tsp")
    tour_folder = os.path.join(output_folder, "tour_opt2_pre_100_300_1.0_6.5_0.1_100")

    if not os.path.exists(tour_folder):
        os.makedirs(tour_folder)

    for file_name in os.listdir(tsp_folder):
        if file_name.endswith('.tsp'):
            file_path = os.path.join(tsp_folder, file_name)
            cities, distance_matrix = read_tsp_file(file_path)

            print(f"{file_name} 开始处理")

            # 蚁群算法找到初步的最佳路径
            best_tour, best_distance, best_distances, average_distances = ant_colony_optimization(distance_matrix, num_ants=100,
                                                                                               num_iterations=300)

            # 创建子文件夹
            base_name = os.path.splitext(file_name)[0]
            output_folder_path = os.path.join(tour_folder, base_name)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            # 保存最佳路径
            tour_file_path = os.path.join(output_folder_path, f'{base_name}.tour')
            with open(tour_file_path, 'w') as f:
                f.write(f"NAME : {file_name}\n")
                f.write("TYPE : TOUR\n")
                f.write(f"DIMENSION : {len(cities)}\n")
                f.write(f"TOUR_LENGTH : {best_distance}\n")
                f.write("TOUR_SECTION\n")
                for city in best_tour:
                    f.write(f"{city + 1}\n")  # 输出城市编号，1-indexed
                f.write("-1\n")

            # 绘制收敛程度图
            plot_convergence(best_distances, average_distances, base_name, output_folder_path)

            print(f"{file_name} 处理完成")


# 主函数
if __name__ == "__main__":
    input_folder = "./tsp_data"
    output_folder = "./tsp_data"
    process_tsp_files(input_folder, output_folder)
