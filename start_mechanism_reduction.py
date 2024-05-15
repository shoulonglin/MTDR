# !/usr/bin/env python
# encoding: utf-8
import math
import os
import pathlib
import shutil
import time
import copy
from tqdm import tqdm
import numpy as np
import cantera as ct
import pymars.soln2cti as ctiwritter
from logger import get_logger
from functions import input_data, get_species_data, calculate_ignition_delay, remove_species, remove_reactions, \
    get_sp_error, calculate_ignition_delay_and_max_error, point2path_angle, classified_list, cal_psr, bottom_sp_list, list_after_gap

if __name__ == '__main__':
    # Parsing input files
    # os.makedirs('./mechanisms')
    logger = get_logger()
    logger.info('Reading input file')
    start_reduction = time.time()
    input_data_tuple = input_data()
    gas_name = input_data_tuple[0]
    gas = ct.Solution(input_data_tuple[0])
    T = input_data_tuple[1]
    sim_pressure = ct.one_atm * input_data_tuple[2]
    sim_phi = input_data_tuple[3]
    gas_fuel = input_data_tuple[4]
    gas_oxidizer = input_data_tuple[5]
    error_limit = input_data_tuple[6]
    important_species = input_data_tuple[7]
    mark = input_data_tuple[8]
    length = len(gas.species())
    logger.info('The input file has been read')
    logger.info(f'important species length is {len(important_species)}')
    logger.info('\n')

    # Exporting data through Cantera simulation
    # calculating ignition delay at given temperature point
    start_cal_ig = time.time()
    logger.info('Calculating detailed mechanism ignition delay')
    detailed_mech_ignition_delay_list = calculate_ignition_delay_and_max_error(T, gas, sim_pressure, sim_phi, gas_fuel,
                                                                               gas_oxidizer)
    end_cal_ig = time.time()
    logger.info(f'detailed mechanism ignition delay is {detailed_mech_ignition_delay_list}')
    logger.info(f'calculating detailed mechanism ignition delay cost {(end_cal_ig - start_cal_ig):.2f} s')
    logger.info('\n')
    # write detailed mechanism ignition delay data to output_data.csv
    with open('output_data.csv', 'w') as f:
        f.write('detailed mechanism ignition delay at given temperature point：\n')
    with open('output_data.csv', 'a') as f:
        for i in range(len(T)):
            f.write(f'{T[i]},{detailed_mech_ignition_delay_list[i]}\n')
    with open('output_data.csv', 'a') as f:
        f.write('\n')

    # export species concentration-time data file
    sp_dataframe_list = []
    start_export_ct_data = time.time()
    for t in T:
        tmp_sp_dataframe = get_species_data(gas, (t, sim_pressure), sim_phi,
                                            detailed_mech_ignition_delay_list[T.index(t)], gas_fuel, gas_oxidizer)
        sp_dataframe_list.append(tmp_sp_dataframe)
    end_export_ct_data = time.time()
    logger.info(
        f'concentration-time data has been exported, the process cost {(end_export_ct_data - start_export_ct_data):.2f} s')
    logger.info('\n')

    # calculating the sum of initial manifold geodesic distance matrix at given temperature point
    start_cal_initial_mf_sum = time.time()
    logger.info('calculating the sum of initial manifold geodesic distance matrix')
    init_manifold_geodesic_dist_list = []
    angle_list = []

    for t in T:
        array_sp_data = sp_dataframe_list[T.index(t)].values
        tmp_data = point2path_angle(array_sp_data)
        init_manifold_geodesic_dist_list.append(tmp_data[0])
        angle_list.append(tmp_data[1])
    end_cal_initial_mf_sum = time.time()
    logger.info(f'the calculating procedure cost {(end_cal_initial_mf_sum - start_cal_initial_mf_sum):.2f} s')
    logger.info('\n')
    logger.info('calculating error of each species')
    start_cal_sp_error = time.time()
    error_list_at_t = []
    count = 0
    for sp_dataframe in sp_dataframe_list:
        sp_error_list_t = get_sp_error(sp_dataframe, init_manifold_geodesic_dist_list[count], angle_list[count])
        count += 1
        error_list_at_t.append(sp_error_list_t)
    end_cal_sp_error = time.time()
    logger.info(f'The time of calculating error of each species is {(end_cal_sp_error - start_cal_sp_error):.2f} s')

    # Combine the results into an ascending sequence
    each_species_total_error = []
    tmp_np_species_error = np.array(error_list_at_t)
    np_species_error = copy.deepcopy(tmp_np_species_error.T)
    for i in range(length):
        each_species_total_error.append(np.sum(np_species_error[i]))
    # sort
    sp_name = []
    for sp in gas.species():
        sp_name.append(sp.name)
    sp_error_dict = {}
    for i in range(length):
        sp_error_dict[sp_name[i]] = each_species_total_error[i]
    sequence_sp_error = sorted(sp_error_dict.items(), key=lambda item: item[1])
    sequence_sp_name = []
    for i in sequence_sp_error:
        sequence_sp_name.append(i[0])
    tmp_sequence_name = copy.deepcopy(sequence_sp_name)
    with open('./sp-geodesic-distance-error.csv', 'w') as f:
        f.write('species,error\n')
    with open('./sp-geodesic-distance-error.csv', 'a') as f:
        for element in sequence_sp_error:
            f.write(f'{element[0]},{element[1]}\n')

    # mechanism reduction
    path = pathlib.Path('./reduced mechanism')
    if path.exists():
        shutil.rmtree('./reduced mechanism')
    os.mkdir('./reduced mechanism')
    count = 0
    print(len(gas.species()))

    bottom_list = bottom_sp_list(gas, sequence_sp_name, important_species)
    print(bottom_list)
    important_c0_c4 = list_after_gap(bottom_list, sequence_sp_error)
    print(important_c0_c4)
    print(len(important_c0_c4))

    for sp in important_c0_c4:
        tmp_sequence_name.remove(sp)
    for sp in important_species:
        tmp_sequence_name.remove(sp)

    sp_eliminated_list = copy.deepcopy(tmp_sequence_name)
    print(sequence_sp_error)
    psr_T_detailed = cal_psr(gas, sim_phi, sim_pressure, gas_fuel, gas_oxidizer)
    count_sp_list = copy.deepcopy(sp_eliminated_list)
    print(len(count_sp_list))
    num_for_eight = math.floor(len(sp_eliminated_list) / 8)
    for i in tqdm(range(num_for_eight)):
        t_start = time.time()
        logger.info(f'\n{i}-th reduction')
        tmp_sp_list = sp_eliminated_list[i * 8:i * 8 + 8]
        init_species = gas.species()
        init_reactions = gas.reactions()
        for sp in tmp_sp_list:
            new_species = remove_species(gas, sp)
            new_reactions = remove_reactions(gas, sp)
            gas = ct.Solution(thermo='ideal-gas', kinetics='gas', species=new_species, reactions=new_reactions)
            logger.info(len(gas.species()))
        rm_mech_ig_delay, tmp_sp_error_max = calculate_ignition_delay_and_max_error(T, gas, sim_pressure, sim_phi,
                                                                                    gas_fuel, gas_oxidizer,
                                                                                    detailed_mech_ignition_delay_list)
        psr_T_tmp = cal_psr(gas, sim_phi, sim_pressure, gas_fuel, gas_oxidizer)

        psr_error = abs(psr_T_tmp - psr_T_detailed) / psr_T_detailed
        global_error = max(tmp_sp_error_max, psr_error)
        if global_error <= error_limit:
            t_end = time.time()
            logger.info(
                f'8 species has been deleted from mechanism, and the max error of ignition delay is {(tmp_sp_error_max * 100):.2f}%, and cost {(t_end - t_start):.2f} s')
            logger.info(tmp_sp_list)
            ctiwritter.write(gas, f'./reduced mechanism/{i}-th___{len(gas.species())}SP_{len(gas.reactions())}R.cti')
            pass
        else:
            gas = ct.Solution(thermo='ideal-gas', kinetics='gas', species=init_species, reactions=init_reactions)
            for sp in tmp_sp_list:
                tt_start = time.time()
                logger.info(f'{i}.{tmp_sp_list.index(sp)}-th reduction')
                initial_species = gas.species()
                initial_reactions = gas.reactions()
                new_species = remove_species(gas, sp)
                new_reactions = remove_reactions(gas, sp)
                gas = ct.Solution(thermo='ideal-gas', kinetics='gas', species=new_species, reactions=new_reactions)

                rm_mech_ig_delay, tmp_sp_error_max = calculate_ignition_delay_and_max_error(T, gas, sim_pressure,
                                                                                            sim_phi,
                                                                                            gas_fuel, gas_oxidizer,
                                                                                            detailed_mech_ignition_delay_list)
                psr_T_tmp = cal_psr(gas, sim_phi, sim_pressure, gas_fuel, gas_oxidizer)

                psr_error = abs(psr_T_tmp - psr_T_detailed) / psr_T_detailed
                global_error = max(tmp_sp_error_max, psr_error)
                tt_end = time.time()
                if global_error <= error_limit:
                    logger.info(
                        f'max ignition delay error is {(tmp_sp_error_max * 100):.2f}%s, cost {(tt_end - tt_start):.2f} s')
                    ctiwritter.write(gas,
                                     f'./reduced mechanism/{i}.{tmp_sp_list.index(sp)}-th___{len(gas.species())}SP_{len(gas.reactions())}R.cti')
                    logger.info(sp)
                    pass
                else:
                    logger.info(
                        f'max ignition delay error is {(tmp_sp_error_max * 100):.2f}% s, cost {(tt_end - tt_start):.2f} s')

                    logger.info(sp)
                    gas = ct.Solution(thermo='ideal-gas', kinetics='gas', species=initial_species,
                                      reactions=initial_reactions)

    tmp_sp_list_last = count_sp_list[num_for_eight * 8:]
    for sp in tmp_sp_list_last:
        tt_start = time.time()
        logger.info(f'{num_for_eight}.{tmp_sp_list_last.index(sp)}-th reduction')
        initial_species = gas.species()
        initial_reactions = gas.reactions()
        new_species = remove_species(gas, sp)
        new_reactions = remove_reactions(gas, sp)
        gas = ct.Solution(thermo='ideal-gas', kinetics='gas', species=new_species, reactions=new_reactions)

        rm_mech_ig_delay, tmp_sp_error_max = calculate_ignition_delay_and_max_error(T, gas, sim_pressure, sim_phi,
                                                                                    gas_fuel, gas_oxidizer,
                                                                                    detailed_mech_ignition_delay_list)
        psr_T_tmp = cal_psr(gas, sim_phi, sim_pressure, gas_fuel, gas_oxidizer)

        psr_error = abs(psr_T_tmp - psr_T_detailed) / psr_T_detailed
        global_error = max(tmp_sp_error_max, psr_error)
        tt_end = time.time()
        if global_error <= error_limit:
            logger.info(
                f'max ignition delay error is {(tmp_sp_error_max * 100):.2f}% s, cost {(tt_end - tt_start):.2f} s')
            pass
        else:
            logger.info(
                f'max ignition delay error is {(tmp_sp_error_max * 100):.2f}% s, cost {(tt_end - tt_start):.2f} s')
            gas = ct.Solution(thermo='ideal-gas', kinetics='gas', species=initial_species,
                              reactions=initial_reactions)

    ctiwritter.write(gas, 'rd_mech.cti')
    final_ignition = []
    for t in T:
        final_ignition_at_t = calculate_ignition_delay(gas, (t, sim_pressure), sim_phi, gas_fuel, gas_oxidizer)
        final_ignition.append(final_ignition_at_t)
    with open('output_data.csv', 'a') as f:
        f.write('reduced mechanism ignition delay：\n')
    with open('output_data.csv', 'a') as f:
        for i in range(len(T)):
            f.write(f'{T[i]},{final_ignition[i]}\n')
    with open('output_data.csv', 'a') as f:
        f.write('\n')
    final_ignition_error_list = []
    for i in range(len(final_ignition)):
        final_ignition_error_at_t = abs((detailed_mech_ignition_delay_list[i] - final_ignition[i])
                                        / detailed_mech_ignition_delay_list[i])
        final_ignition_error_list.append(final_ignition_error_at_t)
    final_ignition_error = max(final_ignition_error_list)
    with open('output_data.csv', 'a') as f:
        f.write(f'Max ignition delay error of reduced mechanism: {(final_ignition_error * 100) :.2f}%\n')
    ctiwritter.write(gas, f'{len(gas.species())}SP_{len(gas.reactions())}R_{(final_ignition_error * 100) :.2f}.cti')
    with open('output_data.csv', 'a') as f:
        f.write('\n')
    final_sp = len(gas.species())
    final_reac = len(gas.reactions())
    with open('output_data.csv', 'a') as f:
        f.write(f'Number of species in reduced mechanism: {final_sp}\n')
        f.write(f'Number of reactions in reduced mechanism: {final_reac}\n\n')
    end_reduction = time.time()
    logger.info(f'The total process cost {(end_reduction - start_reduction):.2f} s \n')
    with open('output_data.csv', 'a') as f:
        f.write(f'The total process cost {(end_reduction - start_reduction):.2f}s \n')
