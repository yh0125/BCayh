import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
# 修正导入：使用 IIDBootstrap 而不是不存在的 BCaBootstrap
from arch.bootstrap import IIDBootstrap
import io # 用于处理 BytesIO

# --- 应用配置和标题 ---
st.set_page_config(layout="wide")
st.title("配对数据分析工具 (BCa Bootstrap CI & Wilcoxon Test)")
st.write("""
分别上传包含两次测量数据的 Excel 文件。程序将使用指定的 'Patient ID' 列
来匹配两个文件中的患者，并对两个文件中都存在的测量指标进行配对分析，
生成包含均值、偏差、标准误、指定置信区间和P值的汇总表格。
""")

# --- 用户输入：分析参数 ---
st.sidebar.header("分析参数设置") # 将参数设置放到侧边栏

# 1. Patient ID 列名
if 'patient_id_col' not in st.session_state:
    st.session_state.patient_id_col = 'PatientID'
# 使用 st.session_state 来持久化输入值
st.session_state.patient_id_col = st.sidebar.text_input(
    "输入用于匹配患者的列名:",
    value=st.session_state.patient_id_col
)
patient_id_col = st.session_state.patient_id_col

# 2. 自助法重采次数
if 'n_reps_bootstrap' not in st.session_state:
    st.session_state.n_reps_bootstrap = 2999 # 默认值
st.session_state.n_reps_bootstrap = st.sidebar.number_input(
    "自助法重采次数 (Reps):",
    min_value=100,       # 设置最小值
    max_value=20000,     # 设置最大值 (防止过大导致卡顿)
    value=st.session_state.n_reps_bootstrap, # 当前值
    step=100,            # 步长
    format="%d",         # 格式化为整数
    help="推荐至少 1000-2000 次以获得稳定的置信区间。"
)
n_reps_bootstrap = st.session_state.n_reps_bootstrap

# 3. 置信水平 (Confidence Level)
if 'confidence_level_percent' not in st.session_state:
    st.session_state.confidence_level_percent = 95.0 # 默认值 95%
st.session_state.confidence_level_percent = st.sidebar.slider(
    "置信区间水平 (%):",
    min_value=80.0,
    max_value=99.9,
    value=st.session_state.confidence_level_percent, # 当前值
    step=0.5,            # 步长
    format="%.1f%%",     # 显示格式
    help="选择计算置信区间的百分位水平。"
)
confidence_level_percent = st.session_state.confidence_level_percent
confidence_level = confidence_level_percent / 100.0 # 转换为小数 (例如 0.95)

st.sidebar.info(f"当前设置: Patient ID 列 = '{patient_id_col}', 重采次数 = {n_reps_bootstrap}, 置信水平 = {confidence_level_percent:.1f}%")

# --- 文件格式说明 (移到主区域) ---
st.markdown(f"""
**Excel 文件格式要求:**
- **两个文件** 都必须包含名为 `{patient_id_col}` 的列。
- 每个患者在每个文件中最多占一行。
- 两个文件应包含**相同名称**的测量指标列 (例如, 两个文件都有 `PDFF_ROI1` 列)。
""")

# --- 文件上传 ---
col1, col2 = st.columns(2)
with col1:
    uploaded_file_t1 = st.file_uploader("上传第一次测量数据文件 (.xlsx)", type="xlsx", key="file1")
with col2:
    uploaded_file_t2 = st.file_uploader("上传第二次测量数据文件 (.xlsx)", type="xlsx", key="file2")

# --- 定义统计量函数 (平均差值) ---
def stat_mean(x):
    """计算输入数组的平均值"""
    if len(x) == 0:
        return np.nan
    return np.nanmean(x)

# --- 主分析逻辑 ---
if uploaded_file_t1 is not None and uploaded_file_t2 is not None:
    try:
        # 读取 Excel 文件
        df_t1 = pd.read_excel(uploaded_file_t1, engine='openpyxl')
        df_t2 = pd.read_excel(uploaded_file_t2, engine='openpyxl')
        st.success("两个 Excel 文件读取成功！")

        # --- 验证 Patient ID 列是否存在 ---
        if patient_id_col not in df_t1.columns:
            st.error(f"错误：文件1中未找到列 '{patient_id_col}'。请检查列名或修改侧边栏中的输入。")
            st.stop()
        if patient_id_col not in df_t2.columns:
            st.error(f"错误：文件2中未找到列 '{patient_id_col}'。请检查列名或修改侧边栏中的输入。")
            st.stop()

        # --- 数据预览 ---
        st.subheader("数据预览")
        expander1 = st.expander("第一次测量数据 (前5行)")
        expander1.dataframe(df_t1.head())
        expander2 = st.expander("第二次测量数据 (前5行)")
        expander2.dataframe(df_t2.head())

        # --- 查找共同的测量指标列 ---
        measurement_cols_t1 = set(df_t1.columns) - {patient_id_col}
        measurement_cols_t2 = set(df_t2.columns) - {patient_id_col}
        common_measurement_cols = sorted(list(measurement_cols_t1 & measurement_cols_t2))

        if not common_measurement_cols:
            st.error("错误：两个文件中未能找到任何共同的测量指标列（已排除 Patient ID 列）。")
            st.stop()
        else:
            st.write(f"找到 {len(common_measurement_cols)} 个共同的测量指标进行分析: {', '.join(common_measurement_cols)}")

        # --- 合并数据 ---
        df_merged = pd.merge(
            df_t1,
            df_t2,
            on=patient_id_col,
            how='inner',
            suffixes=('_T1', '_T2')
        )

        if df_merged.empty:
            st.error("错误：根据 Patient ID 未找到任何匹配的患者。请检查两个文件中的 Patient ID 是否一致。")
            st.stop()

        st.write(f"成功匹配 {len(df_merged)} 位患者进行分析。")
        expander_merged = st.expander("匹配后的数据预览 (前5行)")
        expander_merged.dataframe(df_merged.head())

        results = []
        st.header("开始分析...")
        progress_bar = st.progress(0)
        total_cols = len(common_measurement_cols)
        # n_reps_bootstrap 和 confidence_level 现在从用户输入获取

        # 动态生成 CI 列名
        ci_lower_col_name = f"Lower {confidence_level_percent:.1f}% CI"
        ci_upper_col_name = f"Upper {confidence_level_percent:.1f}% CI"

        for i, base_name in enumerate(common_measurement_cols):
            col1_name = base_name + '_T1'
            col2_name = base_name + '_T2'

            # 初始化结果变量
            mean_diff_val = np.nan
            bias_val = np.nan
            std_err_val = np.nan
            lower_ci_val = "N/A"
            upper_ci_val = "N/A"
            wilcoxon_p_val = np.nan
            ci_method_used = "" # 记录使用的CI方法

            try:
                # 提取并清理当前指标的配对数据
                paired_data = df_merged[[col1_name, col2_name]].dropna()
                n_valid_pairs = len(paired_data)

                if n_valid_pairs < 2:
                     st.warning(f"指标 '{base_name}' 的有效配对数据不足 ({n_valid_pairs} 行)，跳过分析。")
                     lower_ci_val = "数据不足"
                     upper_ci_val = "数据不足"
                else:
                    measurement_1 = paired_data[col1_name].values
                    measurement_2 = paired_data[col2_name].values
                    differences = measurement_1 - measurement_2

                    # 1. 计算原始差值均值
                    mean_diff_val = stat_mean(differences)

                    # 2. 执行 Bootstrap 获取分布、偏差、标准误
                    bs = IIDBootstrap(differences, seed=42)
                    try:
                        # 使用用户输入的重采次数
                        bootstrap_reps = bs.apply(stat_mean, reps=n_reps_bootstrap)
                        valid_bootstrap_reps = bootstrap_reps[~np.isnan(bootstrap_reps)]
                        if len(valid_bootstrap_reps) > 1:
                           std_err_val = np.std(valid_bootstrap_reps, ddof=1)
                           bias_val = np.mean(valid_bootstrap_reps) - mean_diff_val
                        else:
                            st.warning(f"指标 '{base_name}': 有效自助抽样统计量不足 ({len(valid_bootstrap_reps)} 个)，无法计算偏差和标准误。")
                    except Exception as e_apply:
                        st.warning(f"指标 '{base_name}': 计算自助抽样分布失败: {e_apply}")


                    # 3. 尝试计算 BCa CI (使用用户输入的参数)
                    ci_calculated = False
                    try:
                        st.write(f"指标 '{base_name}': 尝试计算 BCa CI (Reps={n_reps_bootstrap}, Level={confidence_level_percent}%)")
                        # 使用用户输入的重采次数和置信水平
                        ci = bs.conf_int(stat_mean, reps=n_reps_bootstrap, size=confidence_level, method='bca')
                        st.write(f"指标 '{base_name}': BCa CI 原始结果 (ci): {ci}, 类型: {type(ci)}")

                        lower_bound_bca, upper_bound_bca = None, None
                        valid_ci_bca = False
                        if isinstance(ci, np.ndarray) and ci.shape == (2, 1):
                            lower_bound_bca, upper_bound_bca = float(ci[0, 0]), float(ci[1, 0])
                        elif isinstance(ci, (list, tuple)) and len(ci) == 2:
                            lower_bound_bca = float(np.asarray(ci[0]).item())
                            upper_bound_bca = float(np.asarray(ci[1]).item())

                        if lower_bound_bca is not None and upper_bound_bca is not None and np.isfinite(lower_bound_bca) and np.isfinite(upper_bound_bca):
                            lower_ci_val = lower_bound_bca
                            upper_ci_val = upper_bound_bca
                            ci_method_used = "BCa"
                            ci_calculated = True
                            st.write(f"指标 '{base_name}': BCa CI 计算并提取成功。")
                        else:
                            st.warning(f"指标 '{base_name}': BCa CI 返回值无效或包含非有限数值: {ci}")
                            raise ValueError("BCa CI 返回值无效")

                    except Exception as e_bca:
                        st.warning(f"指标 '{base_name}': BCa Bootstrap 计算失败或结果无效. 尝试 Percentile 方法...")
                        try:
                            st.write(f"指标 '{base_name}': 尝试计算 Percentile CI (Reps={n_reps_bootstrap}, Level={confidence_level_percent}%)")
                            bs_perc = IIDBootstrap(differences, seed=42)
                            # 使用用户输入的重采次数和置信水平
                            ci = bs_perc.conf_int(stat_mean, reps=n_reps_bootstrap, size=confidence_level, method='percentile')
                            st.write(f"指标 '{base_name}': Percentile CI 原始结果 (ci): {ci}, 类型: {type(ci)}")

                            lower_bound_perc, upper_bound_perc = None, None
                            valid_ci_perc = False
                            if isinstance(ci, np.ndarray) and ci.shape == (2, 1):
                                lower_bound_perc, upper_bound_perc = float(ci[0, 0]), float(ci[1, 0])
                            elif isinstance(ci, (list, tuple)) and len(ci) == 2:
                                lower_bound_perc = float(np.asarray(ci[0]).item())
                                upper_bound_perc = float(np.asarray(ci[1]).item())

                            if lower_bound_perc is not None and upper_bound_perc is not None and np.isfinite(lower_bound_perc) and np.isfinite(upper_bound_perc):
                                lower_ci_val = lower_bound_perc
                                upper_ci_val = upper_bound_perc
                                ci_method_used = "Percentile"
                                ci_calculated = True
                                st.write(f"指标 '{base_name}': Percentile CI 计算并提取成功。")
                            else:
                                st.error(f"指标 '{base_name}': Percentile CI 返回值也无效或包含非有限数值: {ci}")
                                lower_ci_val = "计算错误"
                                upper_ci_val = "计算错误"

                        except Exception as e_perc:
                             st.error(f"指标 '{base_name}': Percentile Bootstrap 计算本身失败.")
                             st.exception(e_perc)
                             lower_ci_val = "计算错误"
                             upper_ci_val = "计算错误"

                    if not ci_calculated and lower_ci_val not in ["计算错误", "数据不足"]: # 确保在数据足够但CI计算失败时标记
                         lower_ci_val = "计算错误"
                         upper_ci_val = "计算错误"


                    # 4. 执行 Wilcoxon 检验
                    try:
                        if np.all(differences == 0):
                            wilcoxon_p_val = 1.0
                        elif len(np.unique(differences[differences != 0])) < 1 and n_valid_pairs < 10:
                             st.warning(f"指标 '{base_name}': Wilcoxon 检验的非零差异太少。")
                             wilcoxon_p_val = np.nan
                        else:
                            _, wilcoxon_p_val = stats.wilcoxon(measurement_1, measurement_2)
                    except ValueError as e_wilcoxon:
                         st.warning(f"指标 '{base_name}': Wilcoxon 检验计算失败: {e_wilcoxon}")
                         wilcoxon_p_val = np.nan

            except Exception as outer_e:
                st.error(f"处理指标 '{base_name}' 时发生意外错误: {outer_e}")
                # 其他值保持 NaN 或 N/A

            # 存储结果 (使用动态 CI 列名)
            results.append({
                "Parameter": base_name,
                "Mean (Diff)": mean_diff_val,
                "Bias": bias_val,
                "Std. Error": std_err_val,
                ci_lower_col_name: lower_ci_val, # 使用动态列名
                ci_upper_col_name: upper_ci_val, # 使用动态列名
                "P Value (Wilcoxon)": wilcoxon_p_val,
                "CI Method": ci_method_used
            })

            # 更新进度条
            progress_bar.progress((i + 1) / total_cols)

        # --- 显示汇总结果 ---
        st.header("分析结果汇总")
        if results: # 确保 results 列表不为空
            results_df = pd.DataFrame(results)

            # 定义格式化字典 (CI列需要特殊处理)
            format_dict = {
                "Mean (Diff)": "{:.4f}",
                "Bias": "{:.5f}",
                "Std. Error": "{:.4f}",
                "P Value (Wilcoxon)": "{:.4f}"
            }
            # 条件格式化 CI 列
            format_dict[ci_lower_col_name] = lambda x: f"{x:.4f}" if isinstance(x, (int, float, np.number)) else x
            format_dict[ci_upper_col_name] = lambda x: f"{x:.4f}" if isinstance(x, (int, float, np.number)) else x

            st.dataframe(results_df.style.format(format_dict)\
                        .set_properties(**{'text-align': 'center'})\
                        .set_table_styles([dict(selector='th', props=[('text-align', 'center')])]))

            # --- 提供结果下载 ---
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8-sig')

            csv_data = convert_df_to_csv(results_df)

            st.download_button(
                label="下载结果为 CSV 文件",
                data=csv_data,
                file_name='paired_data_analysis_results_formatted.csv',
                mime='text/csv',
            )
        else:
            st.warning("没有计算出任何结果。")


    except Exception as e:
        st.error(f"处理文件时发生严重错误: {e}")
        st.exception(e)
        st.error("请确保上传的是有效的 Excel (.xlsx) 文件，并且格式符合要求。")

else:
    st.info("请在侧边栏设置参数并上传两个 Excel 文件开始分析。")

