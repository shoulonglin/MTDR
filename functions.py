# !/usr/bin/env python
# encoding: utf-8
"""
@Project: new_manifold_projection
@Time: 2021/7/22 16:46
@Author: caozheng
@File: functions.py
@Idea: PyCharm
"""
import math
import copy
import pymars.soln2cti as ctiwritter

import cantera as ct
import pandas as pd
import numpy as np


def input_data(file_name='input.txt'):
    """
    解析输入文件
    :param file_name: 输入文件的路径
    :return: 返回解析后的一个包含缩减机理所需要的数据的元组
    """
    with open(file_name) as f_func:
        contents_func = f_func.readlines()

    mechanism_name_func = contents_func[0].split(':')[1].strip()

    T_func = []
    b_func = contents_func[1].split(':')[1].strip().replace(' ', '')[1:-1].split(',')
    for b_1 in b_func:
        T_func.append(float(b_1))

    sim_pressure_func = float(contents_func[2].split(':')[1].strip())

    sim_phi_func = float(contents_func[3].split(':')[1].strip())

    fuel_func = contents_func[4].strip().replace(' ', '')[5:]

    oxidizer_func = contents_func[5].strip().replace(' ', '')[9:]

    error_func = float(contents_func[6].split(':')[1].strip())

    important_sp_func = contents_func[7].split(':')[1].strip()[1:-1].replace(' ', '').split(',')

    mark_func = ',' in fuel_func

    return mechanism_name_func, T_func, sim_pressure_func, sim_phi_func, fuel_func, oxidizer_func, error_func, important_sp_func, mark_func


def calculate_ignition_delay(gas_func, gas_state_func, phi_func, fuel_func, oxidizer_func, estimated_ig_func=200):
    """
    calculate ignition delay
    :param estimated_ig_func: estimated ignition delay time,if no detailed mech data, set to 0.1. otherwise set to detailed data
    :param gas_func: the gas solution
    :param gas_state_func: state of simulation
    :param phi_func: equivalence ratio
    :param fuel_func: fuel
    :param oxidizer_func: oxidizer
    :return: ignition delay
    """
    gas_func.TP = gas_state_func
    gas_func.set_equivalence_ratio(phi=phi_func, fuel=fuel_func, oxidizer=oxidizer_func)
    r_func = ct.IdealGasConstPressureReactor(gas_func)
    r_net = ct.ReactorNet([r_func])
    time_history = ct.SolutionArray(gas_func, extra="t")

    estimated_ignition_delay_time_func = estimated_ig_func
    t_func = 0
    count_func = 1
    while t_func < estimated_ignition_delay_time_func:
        t_func = r_net.step()
        time_history.append(r_func.thermo.state, t=t_func)
        count_func += 1
    temperature = time_history.T
    temperature_list = list(temperature)
    time = time_history.t
    time_list = list(time)
    ig_delay_func = 0
    for temperature_point in temperature_list:
        if temperature_point >= gas_state_func[0] + 400:
            index_T = temperature_list.index(temperature_point)
            ig_delay_func = time_list[index_T]
            break

    return ig_delay_func


def calculate_ignition_delay_and_max_error(T_func, gas_func, pressure_func, phi_func, fuel_func,
                                           oxidizer_func, detailed_mech_ignition_delay_func=[]):
    """
    计算最大的着火延迟误差，也可计算着火延迟
    :param T_func: 温度列表
    :param gas_func: gas对象
    :param pressure_func: 压力值
    :param phi_func: 当量比
    :param fuel_func: 燃料组成
    :param oxidizer_func: 氧化剂组成
    :param detailed_mech_ignition_delay_func:详细机理在各温度点的着火延迟
    :return: 缩减机理在各温度点的着火延迟  和   缩减机理的最大相对误差
    """
    ignition_delay_at_t_func = []
    for t_func in T_func:
        ig_func = calculate_ignition_delay(gas_func, (t_func, pressure_func), phi_func, fuel_func, oxidizer_func)
        ignition_delay_at_t_func.append(ig_func)

    if not detailed_mech_ignition_delay_func:
        return ignition_delay_at_t_func

    else:
        ignition_delay_error_list_func = []
        for ii_func in range(len(T_func)):
            ignition_delay_error_list_func.append(
                abs((detailed_mech_ignition_delay_func[ii_func] - ignition_delay_at_t_func[ii_func])
                    / detailed_mech_ignition_delay_func[ii_func]))
        test_error_func = max(ignition_delay_error_list_func)

        return ignition_delay_at_t_func, test_error_func


def get_species_data(gas_func, gas_state_func, phi_func, tau_func, fuel_func, oxidizer_func):
    """
    通过cantera的模拟获取相在反应时的组分-时间数据
    :param oxidizer_func: 氧化剂组成
    :param fuel_func: 燃料的组成
    :param tau_func: 对应温度/压力下的点火延迟
    :param phi_func: 气体相的当量比
    :param gas_func: 代表气体相的solution对象
    :param gas_state_func: 气体相的状态，包括温度和压力，元组类型
    :return: 返回一个包含组分浓度随时间变化的dataframe对象
    """
    gas_func.TP = gas_state_func
    gas_func.set_equivalence_ratio(phi_func, fuel=fuel_func, oxidizer=oxidizer_func)
    r_gas_phase = ct.IdealGasConstPressureReactor(gas_func)
    sim_r_gas_phase = ct.ReactorNet([r_gas_phase])

    # 假设t_equilibrium_func 是稳态时间点
    t_equilibrium_func = 100
    sp_matrix_func = []  # sp_matrix_func为临时存储组分数据的列表
    for i_func in range(50):
        sim_r_gas_phase.advance(0.6 * tau_func / 50)
        tmp_vector_list = gas_func.X.tolist()
        tmp_vector_list.append(gas_func.T)
        sp_matrix_func.append(tmp_vector_list)
    for i_func in range(400):
        sim_r_gas_phase.advance(0.6 * tau_func + i_func * 0.6 * tau_func / 400)
        tmp_vector_list = gas_func.X.tolist()
        tmp_vector_list.append(gas_func.T)
        sp_matrix_func.append(tmp_vector_list)
    for i_func in range(50):
        sim_r_gas_phase.advance(1.2 * tau_func + i_func * (t_equilibrium_func - 1.2 * tau_func) / 50)
        tmp_vector_list = gas_func.X.tolist()
        tmp_vector_list.append(gas_func.T)
        sp_matrix_func.append(tmp_vector_list)
    columns_list = [r_gas_phase.component_name(i_func) for i_func in range(2, r_gas_phase.n_vars)]
    columns_list.append('Temperature')
    pd_species_func = pd.DataFrame(sp_matrix_func, columns=columns_list,
                                   index=[j_func * 10 / 500 for j_func in range(500)])

    return pd_species_func


def vector_angle(vector1_func, vector2_func):
    """
    calculate the angle between the vector
    :param vector1_func: vector1
    :param vector2_func: vector2
    :return: angle
    """
    length_func = len(vector1_func)
    vector_product_func = 0
    for i in range(length_func):
        vector_product_func = vector_product_func + vector1_func[i] * vector2_func[i]

    norm_v1_v2 = np.linalg.norm(vector1_func) * np.linalg.norm(vector2_func)
    if norm_v1_v2 == 0:
        angle_func = 0
    else:
        cos_angle_func = vector_product_func / norm_v1_v2
        if cos_angle_func > 1:
            cos_angle_func = math.floor(cos_angle_func)
        if cos_angle_func < -1:
            cos_angle_func = math.ceil(cos_angle_func)
        angle_func = math.acos(cos_angle_func)

    return angle_func


def point2path_angle(p_data):
    """
    calculate path and angle
    :param p_data:numpy 2d array
    :return:
    """
    v_list = []
    length_p = len(p_data)
    for i_func in range(1, length_p):
        v_list.append(p_data[i_func] - p_data[i_func - 1])

    vector_list_func = np.array(v_list)
    path_list_func = np.array([np.linalg.norm(vector_func) for vector_func in vector_list_func])
    angle_list_func = np.array([vector_angle(vector_list_func[i_func], vector_list_func[i_func + 1])
                                for i_func in range(len(vector_list_func) - 1)])
    return np.nan_to_num(path_list_func), np.nan_to_num(angle_list_func)


def remove_species(gas_func, species_func):
    """

    :param gas_func:移除组分
    :param species_func:one species or species list
    :return:
    """
    all_species_func = gas_func.species()
    new_species_func = []
    if isinstance(species_func, list):
        for sp_func in all_species_func:
            if sp_func.name in species_func:
                continue
            new_species_func.append(sp_func)
    else:
        for sp_func in all_species_func:
            if sp_func.name == species_func:
                continue
            new_species_func.append(sp_func)

    return new_species_func


def remove_reactions(gas_func, species_func):
    """

    :param gas_func:
    :param species_func:
    :return:
    """
    all_reactions_func = gas_func.reactions()
    new_reactions_func = []
    if isinstance(species_func, list):
        for R_func in all_reactions_func:
            sp_in_R = []
            for sp_func in R_func.reactants.keys():
                sp_in_R.append(sp_func)
            for sp_func in R_func.products.keys():
                sp_in_R.append(sp_func)
            mark_sp = 0
            for sp_func in sp_in_R:
                if sp_func in species_func:
                    mark_sp = 1
                    break
            if mark_sp == 0:
                new_reactions_func.append(R_func)
    else:
        for R in all_reactions_func:
            if species_func in R.reactants or species_func in R.products:
                continue
            new_reactions_func.append(R)

    return new_reactions_func


def get_sp_error(df_func, initial_path_func, initial_angle_func):
    """
    计算每个时间点的消除各个组分后的流形误差
    :param initial_angle_func:
    :param df_func: 该时间点的原始数据
    :param initial_path_func: 改时间点的初始流形距离矩阵之和
    :return: 误差列表
    """
    array_data = df_func.values
    width_func = len(array_data[0]) - 1
    eli_sp_array_list = []
    for i in range(width_func):
        tmp_array = copy.deepcopy(array_data)
        for j in tmp_array:
            j[i] = 0
        eli_sp_array_list.append(tmp_array)

    error_list_func = []
    for data in eli_sp_array_list:
        path_error_func = 0
        angle_error_func = 0
        tmp_data_func = point2path_angle(data)
        path_tmp = tmp_data_func[0]
        angle_tmp = tmp_data_func[1]

        for i_func in range(len(path_tmp)):
            if initial_path_func[i_func] == 0:
                if path_tmp[i_func] == 0:
                    error_func = 0
                else:
                    error_func = 1
            else:
                if path_tmp[i_func] == 0:
                    error_func = 1
                else:
                    error_func = abs(path_tmp[i_func] - initial_path_func[i_func]) / initial_path_func[i_func]
            path_error_func = path_error_func + error_func
        for i_func in range(len(angle_tmp)):
            if initial_angle_func[i_func] == 0:
                if angle_tmp[i_func] == 0:
                    error_func = 0
                else:
                    error_func = 1
            else:
                if angle_tmp[i_func] == 0:
                    error_func = 1
                else:
                    error_func = abs(angle_tmp[i_func] - initial_angle_func[i_func]) / initial_angle_func[i_func]
            angle_error_func = angle_error_func + error_func
        error_list_func.append(path_error_func + angle_error_func)

    return error_list_func


def classified_list(gas_func, sorted_list_func, important_species_func, mark_func):
    """
    return three classified list
    :param mark_func: single fuel(False) or multi fuel(True)
    :param important_species_func: important_species list
    :param gas_func: gas solution
    :param sorted_list_func: species name list
    :return: classified list
    """
    species_list_func = gas_func.species()
    bottom_sp_length_func = 0
    for sp_func in species_list_func:
        if 'C' not in sp_func.composition or sp_func.composition['C'] < 4:
            bottom_sp_length_func += 1

    sp_list_obj_func = []
    for sp_func in sorted_list_func:
        for sp_obj_func in species_list_func:
            if sp_obj_func.name == sp_func:
                sp_list_obj_func.append(sp_obj_func)
                break

    list_30_func = []
    list_60_func = []
    list_00_func = []
    list_sp_eliminate = []
    if mark_func:
        count_func = 0
        for sp_func in sp_list_obj_func:
            if 'C' not in sp_func.composition or sp_func.composition['C'] < 4:
                count_func += 1
            if count_func < math.floor(bottom_sp_length_func * 0.3):
                list_30_func.append(sp_func.name)
            else:
                list_00_func.append(sp_func.name)

        tmp_list_30_func = copy.deepcopy(list_30_func)
        for sp_func in tmp_list_30_func:
            if sp_func in important_species_func:
                list_30_func.remove(sp_func)

        tmp_list_00_func = copy.deepcopy(list_00_func)
        for sp_func in tmp_list_00_func:
            if sp_func in important_species_func:
                list_00_func.remove(sp_func)

        list_00_sp_obj_func = []
        for sp_func in list_00_func:
            for sp_obj_func in species_list_func:
                if sp_obj_func.name == sp_func:
                    list_00_sp_obj_func.append(sp_obj_func)
                    break

        tmp_list_00_C4_up_func = []
        for sp_func in list_00_sp_obj_func:
            if 'C' in sp_func.composition and sp_func.composition['C'] > 3:
                tmp_list_00_C4_up_func.append(sp_func.name)

        for sp_func in list_30_func:
            list_sp_eliminate.append(sp_func)
        for sp_func in tmp_list_00_C4_up_func[:-1 * math.floor(len(species_list_func) * 0.25)]:
            list_sp_eliminate.append(sp_func)

    else:
        count_func = 0
        for sp_func in sp_list_obj_func:
            if 'C' not in sp_func.composition or sp_func.composition['C'] < 4:
                count_func += 1

            if count_func < math.floor(bottom_sp_length_func * 0.4):
                list_30_func.append(sp_func.name)
            else:
                list_00_func.append(sp_func.name)

        tmp_list_30_func = copy.deepcopy(list_30_func)
        for sp_func in tmp_list_30_func:
            if sp_func in important_species_func:
                list_30_func.remove(sp_func)

        tmp_list_00_func = copy.deepcopy(list_00_func)
        for sp_func in tmp_list_00_func:
            if sp_func in important_species_func:
                list_00_func.remove(sp_func)

        list_00_sp_obj_func = []
        for sp_func in list_00_func:
            for sp_obj_func in species_list_func:
                if sp_obj_func.name == sp_func:
                    list_00_sp_obj_func.append(sp_obj_func)
                    break

        tmp_list_00_C4_up_func = []
        for sp_func in list_00_sp_obj_func:
            if 'C' in sp_func.composition and sp_func.composition['C'] > 3:
                tmp_list_00_C4_up_func.append(sp_func.name)

        for sp_func in list_30_func:
            list_sp_eliminate.append(sp_func)
        for sp_func in tmp_list_00_C4_up_func:
            list_sp_eliminate.append(sp_func)

    return list_sp_eliminate


def bottom_sp_list(gas_func, sorted_list, important_list):
    """

    :param gas_func:
    :param sorted_list:
    :param important_list:
    :return: 排好序的底层组分
    """
    for i in important_list:
        sorted_list.remove(i)
    bottom_list = []
    for i in sorted_list:
        for sp_func in gas_func.species():
            if sp_func.name == i:
                if 'C' not in sp_func.composition or sp_func.composition['C'] < 4:
                    bottom_list.append(i)
    for i in bottom_list:
        sorted_list.remove(i)

    return bottom_list


def list_after_gap(bottom_list_func, sequence_sp_error_list):
    tmp_list = []
    for sp_func in bottom_list_func:
        for ssp_func in sequence_sp_error_list:
            if sp_func == ssp_func[0]:
                tmp_list.append(ssp_func)
                break
    tmp_error_func = [sp_func[1] for sp_func in tmp_list]
    tmp_sp_func = [sp_func[0] for sp_func in tmp_list]
    multiplier_func = []
    non_zero_sp_func = copy.deepcopy(tmp_sp_func)
    non_zero_error_func = copy.deepcopy(tmp_error_func)
    count_func = 0
    for error in tmp_error_func:
        if error == 0.0:
            count_func += 1
            non_zero_error_func.remove(non_zero_error_func[0])
            non_zero_sp_func.remove(non_zero_sp_func[0])
    for sp_error_func in range(1, len(non_zero_error_func)):
        multiplier_func.append(non_zero_error_func[sp_error_func] / non_zero_error_func[sp_error_func - 1])

    index_max = multiplier_func.index(max(multiplier_func))
    c0_c4_after_gap = bottom_list_func[count_func + index_max:]

    return c0_c4_after_gap


def cal_speed_flame(gas_func, T0_func, P0_func, phi_func, fuel_func, oxidizer_func):
    """
    calculate speed flame at specified condition
    :param oxidizer_func: burning oxidizer
    :param fuel_func: burning fuel
    :param gas_func:gas object needed calculation
    :param T0_func:initial T
    :param P0_func:initial P
    :param phi_func:simulation phi
    :return:flame speed
    """
    gas_func.TP = T0_func, P0_func
    gas_func.set_equivalence_ratio(phi_func, fuel_func, oxidizer_func)
    flame_func = ct.FreeFlame(gas_func, width=0.014)
    flame_func.set_refine_criteria(ratio=3, slope=0.2, curve=0.1)
    flame_func.solve(loglevel=0, auto=False)
    flame_speed_func = flame_func.u[0]

    return flame_speed_func


def cal_psr(gas_func, phi_func, P_func, fuel_func, oxidizer_func, T_func=400, residence_time=0.01):
    """
    计算psr温度函数
    :param gas_func:gas对象
    :param T_func: psr反应器入口温度，默认为400K
    :param phi_func: 当量比
    :param P_func: 压力
    :param fuel_func: 燃料
    :param oxidizer_func:氧化剂
    :param residence_time: 滞留时间，默认为0.01
    :return: psr稳态温度
    """
    gas_func.TP = T_func, P_func
    gas_func.set_equivalence_ratio(phi_func, fuel_func, oxidizer_func)
    inlet = ct.Reservoir(gas_func)
    gas_func.equilibrate('HP')
    combustor = ct.IdealGasReactor(gas_func)
    combustor.volume = 1.0
    exhaust = ct.Reservoir(gas_func)
    inlet_mfc = ct.MassFlowController(inlet, combustor, mdot=combustor.mass / residence_time)
    outlet_mfc = ct.PressureController(combustor, exhaust, master=inlet_mfc, K=0.01)
    sim = ct.ReactorNet([combustor])
    sim.set_initial_time(0.0)
    sim.advance_to_steady_state()

    return combustor.T



