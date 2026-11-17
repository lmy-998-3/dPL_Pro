# -*- coding: utf-8 -*-
"""
自适应数据拼接
气象,参数,SM
2005训练
"""
# import os
# import pandas as pd
# ######## gz_model_2
# ##################################################################################### 参数拼接pars 

# path1_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND/South_China120x220/const'
# path2_base = r'/share/home/dq152/APL_MF/script_data/def_ERA5LAND/SM1+ST1/dpl_opt_results/csv_gz'   #拼接默认和优化参数
# output_base1 = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur2/South_China120x220/const'
# # os.makedirs(output_base1, exist_ok=True)

# varConst = ['k_s_l1',#'k_s_l2','k_s_l3','k_s_l4','k_s_l5','k_s_l6','k_s_l7','k_s_l8',
#             'theta_s_l1',#'theta_s_l2','theta_s_l3','theta_s_l4','theta_s_l5','theta_s_l6','theta_s_l7','theta_s_l8',
#             'VGM_alpha_l1',#'VGM_alpha_l2','VGM_alpha_l3','VGM_alpha_l4','VGM_alpha_l5','VGM_alpha_l6','VGM_alpha_l7','VGM_alpha_l8',           
#             'VGM_n_l1',#'VGM_n_l2','VGM_n_l3','VGM_n_l4','VGM_n_l5','VGM_n_l6','VGM_n_l7','VGM_n_l8',
#             'VGM_theta_r_l1',#'VGM_theta_r_l2','VGM_theta_r_l3','VGM_theta_r_l4','VGM_theta_r_l5','VGM_theta_r_l6','VGM_theta_r_l7','VGM_theta_r_l8',
#             'csol_l1',#'csol_l2','csol_l3','csol_l4','csol_l5','csol_l6','csol_l7','csol_l8',
#             'k_solids_l1',#'k_solids_l2','k_solids_l3','k_solids_l4','k_solids_l5','k_solids_l6','k_solids_l7','k_solids_l8',
#             'tkdry_l1',#'tkdry_l2','tkdry_l3','tkdry_l4','tkdry_l5','tkdry_l6','tkdry_l7','tkdry_l8',
#             'tksatf_l1',#'tksatf_l2','tksatf_l3','tksatf_l4','tksatf_l5','tksatf_l6','tksatf_l7','tksatf_l8',
#             'tksatu_l1',#'tksatu_l2','tksatu_l3','tksatu_l4','tksatu_l5','tksatu_l6','tksatu_l7','tksatu_l8'
#              ] 

# # 遍历文件名列表
# for file_name in varConst:
#     file_name_csv = file_name + '.csv'  # 文件名加上 .csv 扩展名

#     # 读取两个路径下的 CSV 文件
#     df1 = pd.read_csv(os.path.join(path1_base, file_name_csv), header=None)
#     df2 = pd.read_csv(os.path.join(path2_base, file_name_csv), header=None)

#     # 拼接两个 DataFrame
#     df_concat = pd.concat([df1, df2], axis=0)

#     # 输出拼接后的 CSV 文件到指定路径
#     output_path = os.path.join(output_base1, file_name_csv)
#     df_concat.to_csv(output_path, index=False, header=False)

# #####################################################################################  模拟值拼接
# ### pars 
# path1_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND/South_China120x220/2005'
# path2_base = r'/share/home/dq152/APL_MF/script_data/def_ERA5LAND/SM1+ST1/verify/2005/CoLM_gz'   #拼接默认和优化
# output_base1 = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur2/South_China120x220/2005'
# # os.makedirs(output_base1, exist_ok=True)

# varConst = [#'ET','runoff',
#             'SM_layer123',#,'SM_layer45','SM_layer67','SM_layer8910',
#             'ST_layer123'#,'ST_layer45','ST_layer67','ST_layer8910'
#             ] 

# # 遍历文件名列表
# for file_name in varConst:
#     file_name_csv = file_name + '.csv'  # 文件名加上 .csv 扩展名

#     # 读取两个路径下的 CSV 文件
#     df1 = pd.read_csv(os.path.join(path1_base, file_name_csv), header=None)
#     df2 = pd.read_csv(os.path.join(path2_base, file_name_csv), header=None)

#     # 拼接两个 DataFrame
#     df_concat = pd.concat([df1, df2], axis=0)

#     # 输出拼接后的 CSV 文件到指定路径
#     output_path = os.path.join(output_base1, file_name_csv)
#     df_concat.to_csv(output_path, index=False, header=False)
        
# ##################################################################################### forcing
# path1_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND/South_China120x220/2005'
# output_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur2/South_China120x220/2005'

# Forcing = ['ssrd', 'strd', 'Q', 't2m', 'tp', 'sp', 'u10', 'v10']

# for file_name in Forcing:
#     file_name_csv = file_name + '.csv'  # 文件名加上 .csv 扩展名

#     # 读取两个路径下的 CSV 文件
#     df1 = pd.read_csv(os.path.join(path1_base, file_name_csv), header=None)
#     df2 = df1

#     # 拼接两个 DataFrame
#     df_concat = pd.concat([df1, df2], axis=0)

#     # 输出拼接后的 CSV 文件到指定路径
#     output_path = os.path.join(output_base, file_name_csv)
#     df_concat.to_csv(output_path, index=False, header=False)

# #################################################################################### 统计值计算，模拟和forcing
# path1_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur2/South_China120x220/2005'
# output_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur2/Statistics'

# varSM = [#'ET','runoff',
#         'SM_layer123',#'SM_layer45','SM_layer67','SM_layer8910',
#         'ST_layer123',#'ST_layer45','ST_layer67','ST_layer8910',
#         'ssrd', 'strd', 'Q', 't2m', 'tp', 'sp', 'u10', 'v10']

# for file_name in varSM:
#     file_name_csv = file_name + '.csv'  # 文件名加上 .csv 扩展名

#     # 读取两个路径下的 CSV 文件
#     df = pd.read_csv(os.path.join(path1_base, file_name_csv), header=None)
    
#     min_val = df.values.min()  # 最小值
#     max_val = df.values.max()  # 最大值
#     mean_val = df.values.mean()  # 平均值
#     std_val = df.values.std()  # 标准差

#     # 将统计值存为 DataFrame，按 4 行 1 列格式保存
#     stats_df = pd.DataFrame([min_val, max_val, mean_val, std_val])

#     # 保存文件到指定路径，文件名保持一致
#     output_file = os.path.join(output_base, file_name+'_stat.csv')
#     stats_df.to_csv(output_file, index=False, header=False)

# ##################################################################################### 统计值计算，pars
# path1_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur2/South_China120x220/const'
# output_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur2/Statistics'

# varSM = [   'k_s_l1',#'k_s_l2','k_s_l3','k_s_l4','k_s_l5','k_s_l6','k_s_l7','k_s_l8',
#             'theta_s_l1',#'theta_s_l2','theta_s_l3','theta_s_l4','theta_s_l5','theta_s_l6','theta_s_l7','theta_s_l8',
#             'VGM_alpha_l1',#'VGM_alpha_l2','VGM_alpha_l3','VGM_alpha_l4','VGM_alpha_l5','VGM_alpha_l6','VGM_alpha_l7','VGM_alpha_l8',           
#             'VGM_n_l1',#'VGM_n_l2','VGM_n_l3','VGM_n_l4','VGM_n_l5','VGM_n_l6','VGM_n_l7','VGM_n_l8',
#             'VGM_theta_r_l1',#'VGM_theta_r_l2','VGM_theta_r_l3','VGM_theta_r_l4','VGM_theta_r_l5','VGM_theta_r_l6','VGM_theta_r_l7','VGM_theta_r_l8',
#             'csol_l1',#'csol_l2','csol_l3','csol_l4','csol_l5','csol_l6','csol_l7','csol_l8',
#             'k_solids_l1',#'k_solids_l2','k_solids_l3','k_solids_l4','k_solids_l5','k_solids_l6','k_solids_l7','k_solids_l8',
#             'tkdry_l1',#'tkdry_l2','tkdry_l3','tkdry_l4','tkdry_l5','tkdry_l6','tkdry_l7','tkdry_l8',
#             'tksatf_l1',#'tksatf_l2','tksatf_l3','tksatf_l4','tksatf_l5','tksatf_l6','tksatf_l7','tksatf_l8',
#             'tksatu_l1',#'tksatu_l2','tksatu_l3','tksatu_l4','tksatu_l5','tksatu_l6','tksatu_l7','tksatu_l8'
#             ]

# for file_name in varSM:
#     file_name_csv = file_name + '.csv'  # 文件名加上 .csv 扩展名

#     # 读取两个路径下的 CSV 文件
#     df = pd.read_csv(os.path.join(path1_base, file_name_csv), header=None)
    
#     min_val = df.values.min()  # 最小值
#     max_val = df.values.max()  # 最大值
#     mean_val = df.values.mean()  # 平均值
#     std_val = df.values.std()  # 标准差

#     # 将统计值存为 DataFrame，按 4 行 1 列格式保存
#     stats_df = pd.DataFrame([min_val, max_val, mean_val, std_val])

#     # 保存文件到指定路径，文件名保持一致
#     output_file = os.path.join(output_base, 'const_'+file_name+'_stat.csv')
#     stats_df.to_csv(output_file, index=False, header=False)

# #####################################################   sur3
# import os
# import pandas as pd
# ######## gz_model_3
# ##################################################################################### 参数拼接pars 

# path1_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur2/South_China120x220/const'
# path2_base = r'/share/home/dq152/APL_MF/script_data/def_ERA5LAND/SM1+ST1/dpl_opt_results_2/csv_gz'   #拼接默认和优化参数
# output_base1 = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur3/South_China120x220/const'
# # os.makedirs(output_base1, exist_ok=True)

# varConst = ['k_s_l1',#'k_s_l2','k_s_l3','k_s_l4','k_s_l5','k_s_l6','k_s_l7','k_s_l8',
#             'theta_s_l1',#'theta_s_l2','theta_s_l3','theta_s_l4','theta_s_l5','theta_s_l6','theta_s_l7','theta_s_l8',
#             'VGM_alpha_l1',#'VGM_alpha_l2','VGM_alpha_l3','VGM_alpha_l4','VGM_alpha_l5','VGM_alpha_l6','VGM_alpha_l7','VGM_alpha_l8',           
#             'VGM_n_l1',#'VGM_n_l2','VGM_n_l3','VGM_n_l4','VGM_n_l5','VGM_n_l6','VGM_n_l7','VGM_n_l8',
#             'VGM_theta_r_l1',#'VGM_theta_r_l2','VGM_theta_r_l3','VGM_theta_r_l4','VGM_theta_r_l5','VGM_theta_r_l6','VGM_theta_r_l7','VGM_theta_r_l8',
#             'csol_l1',#'csol_l2','csol_l3','csol_l4','csol_l5','csol_l6','csol_l7','csol_l8',
#             'k_solids_l1',#'k_solids_l2','k_solids_l3','k_solids_l4','k_solids_l5','k_solids_l6','k_solids_l7','k_solids_l8',
#             'tkdry_l1',#'tkdry_l2','tkdry_l3','tkdry_l4','tkdry_l5','tkdry_l6','tkdry_l7','tkdry_l8',
#             'tksatf_l1',#'tksatf_l2','tksatf_l3','tksatf_l4','tksatf_l5','tksatf_l6','tksatf_l7','tksatf_l8',
#             'tksatu_l1',#'tksatu_l2','tksatu_l3','tksatu_l4','tksatu_l5','tksatu_l6','tksatu_l7','tksatu_l8' 
#             ] 

# # 遍历文件名列表
# for file_name in varConst:
#     file_name_csv = file_name + '.csv'  # 文件名加上 .csv 扩展名

#     # 读取两个路径下的 CSV 文件
#     df1 = pd.read_csv(os.path.join(path1_base, file_name_csv), header=None)
#     df2 = pd.read_csv(os.path.join(path2_base, file_name_csv), header=None)

#     # 拼接两个 DataFrame
#     df_concat = pd.concat([df1, df2], axis=0)

#     # 输出拼接后的 CSV 文件到指定路径
#     output_path = os.path.join(output_base1, file_name_csv)
#     df_concat.to_csv(output_path, index=False, header=False)

# #####################################################################################  模拟值拼接
# ### pars 
# path1_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur2/South_China120x220/2005'
# path2_base = r'/share/home/dq152/APL_MF/script_data/def_ERA5LAND/SM1+ST1/verify_2/2005/CoLM_gz'   #拼接默认和优化
# output_base1 = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur3/South_China120x220/2005'
# # os.makedirs(output_base1, exist_ok=True)

# varConst = [#'ET','runoff',
#             'SM_layer123',#'SM_layer45','SM_layer67','SM_layer8910',
#             'ST_layer123',#'ST_layer45','ST_layer67','ST_layer8910'
#             ] 

# # 遍历文件名列表
# for file_name in varConst:
#     file_name_csv = file_name + '.csv'  # 文件名加上 .csv 扩展名

#     # 读取两个路径下的 CSV 文件
#     df1 = pd.read_csv(os.path.join(path1_base, file_name_csv), header=None)
#     df2 = pd.read_csv(os.path.join(path2_base, file_name_csv), header=None)

#     # 拼接两个 DataFrame
#     df_concat = pd.concat([df1, df2], axis=0)

#     # 输出拼接后的 CSV 文件到指定路径
#     output_path = os.path.join(output_base1, file_name_csv)
#     df_concat.to_csv(output_path, index=False, header=False)
        
# ##################################################################################### forcing
# path1_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND/South_China120x220/2005'
# output_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur3/South_China120x220/2005'

# Forcing = ['ssrd', 'strd', 'Q', 't2m', 'tp', 'sp', 'u10', 'v10']

# for file_name in Forcing:
#     file_name_csv = file_name + '.csv'  # 文件名加上 .csv 扩展名

#     # 读取两个路径下的 CSV 文件
#     df1 = pd.read_csv(os.path.join(path1_base, file_name_csv), header=None)
#     df2 = df1
#     df3 = df1

#     # 拼接两个 DataFrame
#     df_concat = pd.concat([df1, df2, df3], axis=0)

#     # 输出拼接后的 CSV 文件到指定路径
#     output_path = os.path.join(output_base, file_name_csv)
#     df_concat.to_csv(output_path, index=False, header=False)

# #################################################################################### 统计值计算，模拟和forcing
# path1_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur3/South_China120x220/2005'
# output_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur3/Statistics'

# varSM = [#'ET','runoff',
#         'SM_layer123',#'SM_layer45','SM_layer67','SM_layer8910',
#         'ST_layer123',#'ST_layer45','ST_layer67','ST_layer8910',
#         'ssrd', 'strd', 'Q', 't2m', 'tp', 'sp', 'u10', 'v10']

# for file_name in varSM:
#     file_name_csv = file_name + '.csv'  # 文件名加上 .csv 扩展名

#     # 读取两个路径下的 CSV 文件
#     df = pd.read_csv(os.path.join(path1_base, file_name_csv), header=None)
    
#     min_val = df.values.min()  # 最小值
#     max_val = df.values.max()  # 最大值
#     mean_val = df.values.mean()  # 平均值
#     std_val = df.values.std()  # 标准差

#     # 将统计值存为 DataFrame，按 4 行 1 列格式保存
#     stats_df = pd.DataFrame([min_val, max_val, mean_val, std_val])

#     # 保存文件到指定路径，文件名保持一致
#     output_file = os.path.join(output_base, file_name+'_stat.csv')
#     stats_df.to_csv(output_file, index=False, header=False)

# ##################################################################################### 统计值计算，pars
# path1_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur3/South_China120x220/const'
# output_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur3/Statistics'

# varSM = [   'k_s_l1',#'k_s_l2','k_s_l3','k_s_l4','k_s_l5','k_s_l6','k_s_l7','k_s_l8',
#             'theta_s_l1',#'theta_s_l2','theta_s_l3','theta_s_l4','theta_s_l5','theta_s_l6','theta_s_l7','theta_s_l8',
#             'VGM_alpha_l1',#'VGM_alpha_l2','VGM_alpha_l3','VGM_alpha_l4','VGM_alpha_l5','VGM_alpha_l6','VGM_alpha_l7','VGM_alpha_l8',           
#             'VGM_n_l1',#'VGM_n_l2','VGM_n_l3','VGM_n_l4','VGM_n_l5','VGM_n_l6','VGM_n_l7','VGM_n_l8',
#             'VGM_theta_r_l1',#'VGM_theta_r_l2','VGM_theta_r_l3','VGM_theta_r_l4','VGM_theta_r_l5','VGM_theta_r_l6','VGM_theta_r_l7','VGM_theta_r_l8',
#             'csol_l1',#'csol_l2','csol_l3','csol_l4','csol_l5','csol_l6','csol_l7','csol_l8',
#             'k_solids_l1',#'k_solids_l2','k_solids_l3','k_solids_l4','k_solids_l5','k_solids_l6','k_solids_l7','k_solids_l8',
#             'tkdry_l1',#'tkdry_l2','tkdry_l3','tkdry_l4','tkdry_l5','tkdry_l6','tkdry_l7','tkdry_l8',
#             'tksatf_l1',#'tksatf_l2','tksatf_l3','tksatf_l4','tksatf_l5','tksatf_l6','tksatf_l7','tksatf_l8',
#             'tksatu_l1',#'tksatu_l2','tksatu_l3','tksatu_l4','tksatu_l5','tksatu_l6','tksatu_l7','tksatu_l8'
#             ]

# for file_name in varSM:
#     file_name_csv = file_name + '.csv'  # 文件名加上 .csv 扩展名

#     # 读取两个路径下的 CSV 文件
#     df = pd.read_csv(os.path.join(path1_base, file_name_csv), header=None)
    
#     min_val = df.values.min()  # 最小值
#     max_val = df.values.max()  # 最大值
#     mean_val = df.values.mean()  # 平均值
#     std_val = df.values.std()  # 标准差

#     # 将统计值存为 DataFrame，按 4 行 1 列格式保存
#     stats_df = pd.DataFrame([min_val, max_val, mean_val, std_val])

#     # 保存文件到指定路径，文件名保持一致
#     output_file = os.path.join(output_base, 'const_'+file_name+'_stat.csv')
#     stats_df.to_csv(output_file, index=False, header=False)



#####################################################   sur4
import os
import pandas as pd
######## gz_model_4
##################################################################################### 参数拼接pars 

path1_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur3/South_China120x220/const'
path2_base = r'/share/home/dq152/APL_MF/script_data/def_ERA5LAND/SM1+ST1/dpl_opt_results_3/csv_gz'   #拼接默认和优化参数
output_base1 = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur4/South_China120x220/const'
# os.makedirs(output_base1, exist_ok=True)

varConst = ['k_s_l1',#'k_s_l2','k_s_l3','k_s_l4','k_s_l5','k_s_l6','k_s_l7','k_s_l8',
            'theta_s_l1',#'theta_s_l2','theta_s_l3','theta_s_l4','theta_s_l5','theta_s_l6','theta_s_l7','theta_s_l8',
            'VGM_alpha_l1',#'VGM_alpha_l2','VGM_alpha_l3','VGM_alpha_l4','VGM_alpha_l5','VGM_alpha_l6','VGM_alpha_l7','VGM_alpha_l8',           
            'VGM_n_l1',#'VGM_n_l2','VGM_n_l3','VGM_n_l4','VGM_n_l5','VGM_n_l6','VGM_n_l7','VGM_n_l8',
            'VGM_theta_r_l1',#'VGM_theta_r_l2','VGM_theta_r_l3','VGM_theta_r_l4','VGM_theta_r_l5','VGM_theta_r_l6','VGM_theta_r_l7','VGM_theta_r_l8',
            'csol_l1',#'csol_l2','csol_l3','csol_l4','csol_l5','csol_l6','csol_l7','csol_l8',
            'k_solids_l1',#'k_solids_l2','k_solids_l3','k_solids_l4','k_solids_l5','k_solids_l6','k_solids_l7','k_solids_l8',
            'tkdry_l1',#'tkdry_l2','tkdry_l3','tkdry_l4','tkdry_l5','tkdry_l6','tkdry_l7','tkdry_l8',
            'tksatf_l1',#'tksatf_l2','tksatf_l3','tksatf_l4','tksatf_l5','tksatf_l6','tksatf_l7','tksatf_l8',
            'tksatu_l1',#'tksatu_l2','tksatu_l3','tksatu_l4','tksatu_l5','tksatu_l6','tksatu_l7','tksatu_l8' 
            ] 

# 遍历文件名列表
for file_name in varConst:
    file_name_csv = file_name + '.csv'  # 文件名加上 .csv 扩展名

    # 读取两个路径下的 CSV 文件
    df1 = pd.read_csv(os.path.join(path1_base, file_name_csv), header=None)
    df2 = pd.read_csv(os.path.join(path2_base, file_name_csv), header=None)

    # 拼接两个 DataFrame
    df_concat = pd.concat([df1, df2], axis=0)

    # 输出拼接后的 CSV 文件到指定路径
    output_path = os.path.join(output_base1, file_name_csv)
    df_concat.to_csv(output_path, index=False, header=False)

#####################################################################################  模拟值拼接
### pars 
path1_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur3/South_China120x220/2005'
path2_base = r'/share/home/dq152/APL_MF/script_data/def_ERA5LAND/SM1+ST1/verify_3/2005/CoLM_gz'   #拼接默认和优化
output_base1 = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur4/South_China120x220/2005'
# os.makedirs(output_base1, exist_ok=True)

varConst = [#'ET','runoff',
            'SM_layer123',#'SM_layer45','SM_layer67','SM_layer8910',
            'ST_layer123',#'ST_layer45','ST_layer67','ST_layer8910'
            ] 

# 遍历文件名列表
for file_name in varConst:
    file_name_csv = file_name + '.csv'  # 文件名加上 .csv 扩展名

    # 读取两个路径下的 CSV 文件
    df1 = pd.read_csv(os.path.join(path1_base, file_name_csv), header=None)
    df2 = pd.read_csv(os.path.join(path2_base, file_name_csv), header=None)

    # 拼接两个 DataFrame
    df_concat = pd.concat([df1, df2], axis=0)

    # 输出拼接后的 CSV 文件到指定路径
    output_path = os.path.join(output_base1, file_name_csv)
    df_concat.to_csv(output_path, index=False, header=False)
        
##################################################################################### forcing
path1_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND/South_China120x220/2005'
output_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur4/South_China120x220/2005'

Forcing = ['ssrd', 'strd', 'Q', 't2m', 'tp', 'sp', 'u10', 'v10']

for file_name in Forcing:
    file_name_csv = file_name + '.csv'  # 文件名加上 .csv 扩展名

    # 读取两个路径下的 CSV 文件
    df1 = pd.read_csv(os.path.join(path1_base, file_name_csv), header=None)
    df2 = df1
    df3 = df1
    df4 = df1

    # 拼接两个 DataFrame
    df_concat = pd.concat([df1, df2, df3, df4], axis=0)

    # 输出拼接后的 CSV 文件到指定路径
    output_path = os.path.join(output_base, file_name_csv)
    df_concat.to_csv(output_path, index=False, header=False)

#################################################################################### 统计值计算，模拟和forcing
path1_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur4/South_China120x220/2005'
output_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur4/Statistics'

varSM = [#'ET','runoff',
        'SM_layer123',#'SM_layer45','SM_layer67','SM_layer8910',
        'ST_layer123',#'ST_layer45','ST_layer67','ST_layer8910',
        'ssrd', 'strd', 'Q', 't2m', 'tp', 'sp', 'u10', 'v10']

for file_name in varSM:
    file_name_csv = file_name + '.csv'  # 文件名加上 .csv 扩展名

    # 读取两个路径下的 CSV 文件
    df = pd.read_csv(os.path.join(path1_base, file_name_csv), header=None)
    
    min_val = df.values.min()  # 最小值
    max_val = df.values.max()  # 最大值
    mean_val = df.values.mean()  # 平均值
    std_val = df.values.std()  # 标准差

    # 将统计值存为 DataFrame，按 4 行 1 列格式保存
    stats_df = pd.DataFrame([min_val, max_val, mean_val, std_val])

    # 保存文件到指定路径，文件名保持一致
    output_file = os.path.join(output_base, file_name+'_stat.csv')
    stats_df.to_csv(output_file, index=False, header=False)

##################################################################################### 统计值计算，pars
path1_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur4/South_China120x220/const'
output_base = r'/share/home/dq152/APL_MF/script_data/data_input_ERA5LAND_sur4/Statistics'

varSM = [   'k_s_l1',#'k_s_l2','k_s_l3','k_s_l4','k_s_l5','k_s_l6','k_s_l7','k_s_l8',
            'theta_s_l1',#'theta_s_l2','theta_s_l3','theta_s_l4','theta_s_l5','theta_s_l6','theta_s_l7','theta_s_l8',
            'VGM_alpha_l1',#'VGM_alpha_l2','VGM_alpha_l3','VGM_alpha_l4','VGM_alpha_l5','VGM_alpha_l6','VGM_alpha_l7','VGM_alpha_l8',           
            'VGM_n_l1',#'VGM_n_l2','VGM_n_l3','VGM_n_l4','VGM_n_l5','VGM_n_l6','VGM_n_l7','VGM_n_l8',
            'VGM_theta_r_l1',#'VGM_theta_r_l2','VGM_theta_r_l3','VGM_theta_r_l4','VGM_theta_r_l5','VGM_theta_r_l6','VGM_theta_r_l7','VGM_theta_r_l8',
            'csol_l1',#'csol_l2','csol_l3','csol_l4','csol_l5','csol_l6','csol_l7','csol_l8',
            'k_solids_l1',#'k_solids_l2','k_solids_l3','k_solids_l4','k_solids_l5','k_solids_l6','k_solids_l7','k_solids_l8',
            'tkdry_l1',#'tkdry_l2','tkdry_l3','tkdry_l4','tkdry_l5','tkdry_l6','tkdry_l7','tkdry_l8',
            'tksatf_l1',#'tksatf_l2','tksatf_l3','tksatf_l4','tksatf_l5','tksatf_l6','tksatf_l7','tksatf_l8',
            'tksatu_l1',#'tksatu_l2','tksatu_l3','tksatu_l4','tksatu_l5','tksatu_l6','tksatu_l7','tksatu_l8'
            ]

for file_name in varSM:
    file_name_csv = file_name + '.csv'  # 文件名加上 .csv 扩展名

    # 读取两个路径下的 CSV 文件
    df = pd.read_csv(os.path.join(path1_base, file_name_csv), header=None)
    
    min_val = df.values.min()  # 最小值
    max_val = df.values.max()  # 最大值
    mean_val = df.values.mean()  # 平均值
    std_val = df.values.std()  # 标准差

    # 将统计值存为 DataFrame，按 4 行 1 列格式保存
    stats_df = pd.DataFrame([min_val, max_val, mean_val, std_val])

    # 保存文件到指定路径，文件名保持一致
    output_file = os.path.join(output_base, 'const_'+file_name+'_stat.csv')
    stats_df.to_csv(output_file, index=False, header=False)


