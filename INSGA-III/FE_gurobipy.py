import math
import csv
import copy
import xlsxwriter
import matplotlib.pyplot as plt
from gurobipy import quicksum,Model,GRB
from haversine_python import haversine
# 读取文件
class FE_gurobi:
    def read_csv_file(self,depots,number_satellite , centers , satellite_cost):
        """
        :param demand_file: 需求文件
        :param depot_file: 车场文件
        :return:
        """
        I = []  # 衛星（satellite）
        J = []  # 車場（depot）
        Q = {}  # 衛星需求量（如果有）
        C = {}  # 成本（距離）
        XY = {}  # 節點座標

        # 處理衛星資料
        for i in range(number_satellite):
            node_id = f's{i}'
            # node_id = str(i)
            x, y = centers[i]
            cost = satellite_cost[i]  # 可選擇是否加入 Q[node_id] = cost
            I.append(node_id)
            XY[node_id] = (x, y)
            Q[node_id] = cost  # 假設 cost 是需求（視情況調整）

        # 處理車場資料
        for depot in depots:
            depot_id, x, y = depot
            J.append(depot_id)
            XY[depot_id] = (x, y)

        # 合併節點
        N = I + J
        H = haversine()  # 初始化 haversine 類別
        # 建立距離矩陣
        for i in N:
            lat1, lon1 = XY[i]
            for j in N:
                lat2, lon2 = XY[j]
                C[i, j] = H.getDistanceBetweenPointsNew(lat1, lon1, lat2, lon2)

        return N, I, J, C, Q, XY
    # 提取路径
    def extract_routes(self , I,J,X,K):
        I = copy.deepcopy(I)
        route_list = []
        for k in K:
            # 提取 派送阶段路径
            cur_node = None
            for j in J:
                for i in I:
                    if X[j, i,k].x > 0:
                        cur_node = i
                        route = [j,i]
                        I.remove(i)
                        break
            if cur_node is None:
                continue
            while cur_node not in J:
                for i in I+J:
                    if X[cur_node, i, k].x > 0:
                        cur_node = i
                        route.append(i)
                        if i not in J:
                            I.remove(i)
                        break
            route_list.append(route)
        return route_list
    # def draw_routes(self , route_list,XY,I,J):
    #     for route in route_list:
    #         path_x = []
    #         path_y = []
    #         for n in route:
    #             path_x.append(XY[n][0])
    #             path_y.append(XY[n][1])
    #         plt.plot(path_x, path_y, ms=5)
    #     demand_point_x = [XY[n][0] for n in I]
    #     demand_point_y = [XY[n][1] for n in I]
    #     depot_point_x = [XY[n][0] for n in J]
    #     depot_point_y = [XY[n][1] for n in J]
    #     plt.scatter( demand_point_x, demand_point_y, marker='s', c='b', s=30,zorder=0)
    #     plt.scatter( depot_point_x, depot_point_y, marker='*', c='r', s=100,zorder=1)
    #     plt.show()
    # 保存结果
    def print_route_info(self, route_list, total_cost, C , K):
        # print("第一階段總費用：", total_cost * 0.2)
        # print("{:<6} {:<30} {:<10}".format("車輛", "路徑", "距離"))

        # for idx, route in enumerate(route_list):
        #     route_str = [str(i) for i in route]
        #     dist = sum(C[route[i], route[i + 1]] for i in range(len(route) - 1))
        #     print("{:<6} {:<30} {:<10.2f}".format(idx + 1, ' -> '.join(route_str), dist))
        return total_cost * 0.2 + len(K) * 80
    # 建模和求解
    def solve_model(self , N,I,J,K,Q,V_CAP,C,XY):
        """
        :param N: 所有节点
        :param I: 客户节点
        :param J: 车场节点
        :param K: 车辆节点
        :param Q: 客户需求
        :param V_CAP: 车辆容量
        :param C: 成本矩阵
        :param XY: 节点坐标
        :return: nan
        """
        model = Model('MDVRP')
        # 添加变量
        X = model.addVars(N,N,K,vtype=GRB.BINARY,name='X[i,j,k]')
        U = model.addVars(K, N, vtype=GRB.CONTINUOUS, name='U[k,i]')
        # 目标函数
        obj = quicksum(X[i,j,k]*C[i,j] for i in N for j in N for k in K)
        model.setObjective(obj,GRB.MINIMIZE)
        # 需求覆盖约束
        model.addConstrs( (quicksum(X[i,j,k] for j in N for k in K if i != j) == 1 for i in I),name='eq1' )
        # 车辆容量约束
        model.addConstrs( (quicksum(X[i,j,k]*Q[i] for i in I for j in N if i != j) <= V_CAP for k in K),name= 'eq2')
        # 车辆起点约束
        model.addConstrs( (quicksum(X[j,i,k] for j in J for i in I if i != j) == 1 for k in K),name='eq3' )
        # 中间节点流平衡约束
        model.addConstrs( (quicksum(X[i, j, k] for j in N if i != j) == quicksum(X[j, i, k] for j in N if i != j) for i in I for k in K),name='eq4' )
        # 车辆终点约束
        model.addConstrs( (quicksum(X[i,j,k] for i in I for j in J if i != j) == 1 for k in K), name='eq5' ) # 开放式
        # model.addConstrs( (quicksum(X[j,i,k] for i in I) == quicksum(X[i,j,k] for i in I) for k in K for j in J), name='eq5')  # 不开放式
        # 破除子环
        # model.addConstrs(U[k, j] == 0 for k in K for j in J)
        model.addConstrs(U[k, j] == 0 for k in K for j in J)  # 起點設為 0
        model.addConstrs(U[k, i] - U[k, j] + V_CAP * X[i, j, k] <= V_CAP - Q[i] for i in I for j in I for k in K)
        model.addConstrs(Q[i] <= U[k, i] for k in K for i in I)
        model.addConstrs(U[k, i] <= V_CAP for k in K for i in I)
        # 避免车辆直接在车场间移动
        model.addConstrs( X[i,j,k] == 0 for i in J for j in J for k in K )
        # 求解
        # model.Params.TimeLimit = 300  # 设置求解时间上限
        # model.Params.OutputFlag = 1
        model.Params.OutputFlag = 0 
        model.optimize()
        if model.status == GRB.Status.OPTIMAL or model.status == GRB.Status.TIME_LIMIT:
            route_list = self.extract_routes(I,J,X,K)
            # self.draw_routes(route_list, XY, I,J)
            Cost = self.print_route_info(route_list, model.objVal, C , K)
            return Cost
        else:
            model.computeIIS()
            model.write('model.ilp')
            # for c in model.getConstrs():
            #     if c.IISConstr:
            #         print(f'{c.constrName}')
            print("no solution")
    def main(self , depots,number_satellite , centers ,K ,satellite_cost ,V_CAP):
        N, I, J, C, Q, XY = self.read_csv_file(depots,number_satellite , centers , satellite_cost)
        
        # print(N, I, J, K, Q, V_CAP, C,XY)
        Cost = self.solve_model(N, I, J, K, Q, V_CAP, C,XY)
        return Cost
