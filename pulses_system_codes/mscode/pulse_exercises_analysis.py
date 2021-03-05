from utils import *
from openpyxl import load_workbook


def pulse_analysis(xls_path, sheet_name="pulse_zyy", scale=[0.2, 0.8], is_show=True, is_out=False):
    """
    主要分析程序
    :param xls_path:    数据路径 需要有表头 第一个是时间 第二个是响应
    :param sheet_name:  要处理的是哪个sheet 输入name
    :param scale:       要截取的数据
    :param is_show:     是否展示图片
    :param is_out:      是否保存图片 输出结果 会输出在outputs 以及results.xlsx里
    :return:             返回特征字典 以及 处理的各个中间数据
    """

    # 保存处理的所有数据
    signals = {}
    # 读取数据
    data = pd.read_excel(xls_path, header=0, sheet_name=sheet_name)
    # 读取表头
    times_title, signals_title = data.columns.tolist()
    # 读取数据
    raw_times = data[times_title]
    raw_pulses = data[signals_title]
    # 绘制原始数据
    # common_plot(raw_times,raw_pulses,title="Raw Data",is_show=isShow)

    # 取中间那一段数据 相对较稳定 按照scale里面的数据分析
    if scale[1] <= 1:
        selected_times = raw_times[int(scale[0] * len(raw_times)):int(scale[1] * len(raw_times))]
        selected_pulses = raw_pulses[int(scale[0] * len(raw_pulses)):int(scale[1] * len(raw_pulses))]
    else:
        selected_times = raw_times[scale[0]:scale[1]]
        selected_pulses = raw_pulses[scale[0]:scale[1]]
    # 绘制筛选的中间部分数据
    # common_plot(selected_times, selected_pulses, title="Selected Data", is_show=isShow)

    # 数据标准化
    mean_pulses = np.nanmean(selected_pulses, axis=0)
    std_pulses = np.nanstd(selected_pulses, axis=0, ddof=1)
    standard_pulses = (selected_pulses - mean_pulses) / std_pulses
    # 计算采样率 第一列 上下行相减diff 取均值然后倒数
    sampling_rate = np.round(1 / np.mean(selected_times.diff()),2)
    # 提取特征点
    feature_points = signal_process(standard_pulses, sampling_rate)

    signals["Raw Times"] = raw_times
    signals["Raw Pulses"] = raw_pulses
    signals["Selected Times"] = selected_times
    signals["Selected Pulses"] = selected_pulses
    signals["Sampling Rate"] = sampling_rate
    signals["Standard Pulses"] = standard_pulses
    signals["Mean Pulses"] = mean_pulses
    signals["Std Pulses"] = std_pulses
    signals["Feature Points"] = feature_points
    # 分析数据
    eval_matrixes_dict = signals_analysis(signals, is_show=is_show, is_out=is_out, out_path=sheet_name)
    # 是否输出显示图片
    if is_show:
        signals_plot(signals, is_out=is_out, out_path=sheet_name)
    # 是否输出到excel中
    if is_out:
        # 输出结果
        out_df = pd.DataFrame(
            {
                "Mean Pulses": signals["Mean Pulses"],
                "Std Pulses": signals["Std Pulses"],
                "HR_Time": signals["Feature Points"]["HR_Time"],
                "HR_FFT": eval_matrixes_dict['HRV New']['HR_FFT'],
                "F1": eval_matrixes_dict['HRV New']['F1'],
                "F2": eval_matrixes_dict['HRV New']['F2'],
                "F2_F1": eval_matrixes_dict['HRV New']['F2_F1']
            },
            index=[0]
        )
        dataframe = pd.concat([out_df, eval_matrixes_dict['HRV Time'], eval_matrixes_dict['HRV Frequency'],
                               eval_matrixes_dict['HRV Nonlinear']], axis=1)
        # 存入数据
        if not os.path.exists("outputs"):
            os.mkdir("outputs")
        output_path_excel = os.path.join("outputs", "results2.xlsx")
        if not os.path.exists(output_path_excel):
            dataframe.to_excel(output_path_excel, sheet_name=sheet_name)
        else:
            writer = pd.ExcelWriter(output_path_excel)
            book = load_workbook(output_path_excel)
            writer.book = book
            dataframe.to_excel(writer, sheet_name=sheet_name)
            writer.save()
    return signals, eval_matrixes_dict


def analysis_All(xls_path, scale=[0.1, 0.9], is_show=True, is_out=False):
    xl = pd.ExcelFile(xls_path)
    all_sheets = xl.sheet_names
    all_features_dict = {}
    signals_dict = {}
    for sheet_name in all_sheets:
        signals_dict[sheet_name], all_features_dict[sheet_name] = pulse_analysis(xls_path, sheet_name, scale=scale, is_show=is_show, is_out=is_out)
    pass

    # plt.figure(figsize=(40, 8))
    # sheet_name1 = 'Male_after_exercises'
    # sheet_name2 = 'Male_neck_2'
    # y1 = signals_dict[sheet_name1]['Feature Points']['Heart Rates']
    # y2 = signals_dict[sheet_name2]['Feature Points']['Heart Rates']
    # ppg_rate_mean1 = signals_dict[sheet_name1]['Feature Points']['HR_Time']
    # ppg_rate_mean2 = signals_dict[sheet_name2]['Feature Points']['HR_Time']
    # x = np.linspace(0, y.shape[0] / 50, y.shape[0])
    # plt.plot(y1, color="#FB1CF0", label=sheet_name1, linewidth=1.5)
    # plt.axhline(y=ppg_rate_mean1, label="Mean Heart Rate=" + str(np.round(ppg_rate_mean1, 2)), linestyle="--",
    #             color="#FB1CF0")
    # plt.annotate("Mean Heart Rate=" + str(np.round(ppg_rate_mean1, 2)), xy=(0, int(ppg_rate_mean1)),
    #              xytext=(-1.5, int(ppg_rate_mean1) + 1), color="#FB1CF0")
    # plt.plot(y2, color="#1E90FF", label=sheet_name2, linewidth=1.5)
    # plt.axhline(y=ppg_rate_mean2, label="Mean Heart Rate=" + str(np.round(ppg_rate_mean2, 2)), linestyle="--",
    #             color="#1E90FF")
    # plt.annotate("Mean Heart Rate=" + str(np.round(ppg_rate_mean2, 2)), xy=(0, int(ppg_rate_mean2)),
    #              xytext=(-1.5, int(ppg_rate_mean2) + 1), color="#1E90FF")
    # plt.legend(loc="upper right")
    # title = "Heart Rate Varability (HRV)"
    # plt.title("Heart Rate Varability (HRV)")
    # plt.xlabel("Time (seconds)")
    # plt.ylabel("Heart Rate")
    # plt.show()
    #
    # # 频域分析
    # plt.figure(figsize=(40, 8))
    # sheet_name1 = 'Male_after_exercises'
    # sheet_name2 = 'Male_neck_2'
    # y1 = all_features_dict[sheet_name1]['HRV New']['Power']
    # y2 = all_features_dict[sheet_name2]['HRV New']['Power']
    # x1 = all_features_dict[sheet_name1]['HRV New']['Freq']
    # x2 = all_features_dict[sheet_name2]['HRV New']['Freq']
    # plt.plot(x1, y1, label=sheet_name1)
    # # plt.plot(x2, y2, label=sheet_name2)
    # plt.legend(loc="upper right")
    # plt.ylabel("Power")
    # plt.xlabel("Freq(Hz)")
    # title = "FFT analysis"
    # plt.title(title)
    # plt.show()
    #
    #
    # plt.figure(figsize=(40, 8))
    # sheet_name1 = 'Male_after_exercises'
    # sheet_name2 = 'Female_after_exercises'
    # y1 = signals_dict[sheet_name1]['Feature Points']['Cleaned Pulses']
    # y2 = signals_dict[sheet_name2]['Feature Points']['Cleaned Pulses']
    # x = np.linspace(0, y.shape[0] / 50, y.shape[0])
    # plt.plot(x, y1, label=sheet_name1)
    # plt.plot(x, y2, label=sheet_name2)
    # x_label = "Time (seconds)"
    # y_label = "Pulse amplitude (mA)"
    # plt.legend(loc="upper right")
    # plt.title("Standard Data")
    # plt.xlabel(x_label)
    # plt.ylabel(y_label)
    # plt.show()
    #
    # plt.figure(figsize=(20, 8))
    # for sheet_name in all_sheets:
    #     x = signals_dict[sheet_name]['Selected Times']
    #     y = signals_dict[sheet_name]['Selected Pulses']
    #     plt.plot(x, y, label=sheet_name)
    # x_label = "Time (seconds)"
    # y_label = "Pulse amplitude (mA)"
    # plt.legend(loc="upper right")
    # plt.title("All Raw Selected Data")
    # plt.xlabel(x_label)
    # plt.ylabel(y_label)
    # plt.show()



if __name__ == '__main__':
    xls_path = './data/exercises-analysis2.xlsx'
    # eval_matrixes_dict1 = pulse_analysis(xls_path, sheet_name, is_show=True, is_out=False)

    # sheet_name = "pulse_female_before_exercises"
    # eval_matrixes_dict2 = pulse_analysis(xls_path, sheet_name, is_show=True, is_out=False)
    #
    # sheet_name = "pulse_male_before_exercises"
    # eval_matrixes_dict3 = pulse_analysis(xls_path, sheet_name, is_show=True, is_out=False)

    analysis_All(xls_path, scale=[100, 3100], is_show=True, is_out=True)

    pass
