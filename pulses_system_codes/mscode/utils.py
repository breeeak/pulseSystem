# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal as spsg
import matplotlib.pyplot as plt
import neurokit2 as nk
import os
import datetime


def as_vector(x):
    """
    数据检查 转化为标准格式.
    :param x 一维数据 电流响应值
    """
    if isinstance(x, (pd.Series, pd.DataFrame)):
        out = x.values
    elif isinstance(x, (str, float, int, np.int, np.intc, np.int8, np.int16, np.int32, np.int64)):
        out = np.array([x])
    else:
        out = np.array(x)

    if isinstance(out, np.ndarray):
        shape = out.shape
        if len(shape) == 1:
            pass
        elif len(shape) != 1 and len(shape) == 2 and shape[1] == 1:
            out = out[:, 0]
        else:
            raise ValueError(
                "需要一个一维数据"
            )

    return out


def segment_pulses(pulses, peaks, samplingrate=100, is_out=False, out_path="figs"):
    """
    分割每一个波形 并进行绘制
    :param pulses:      波形数据
    :param peaks:       波峰位置
    :param is_out:      是否输出
    :param out_path:    输出的名字  会进行拼接的
    :return:
    """

    cmap = iter(plt.cm.YlOrRd(np.linspace(0, 1, num=len(peaks) - 1)))
    lines = []
    plt.figure(figsize=(5, 8))
    for i, color in zip(range(len(peaks) - 1), cmap):
        x_length = peaks[i + 1] - peaks[i]
        (line,) = plt.plot((np.array(range(x_length)) - int(x_length / 2))/samplingrate, pulses[peaks[i]:peaks[i + 1]], color=color)
        lines.append(line)
    plt.xlabel("Standard Time (seconds)")
    plt.ylabel("Standard Pulse amplitude (mA)")
    title = "All Segmented Pulses"
    plt.title(title)
    if is_out:
        no = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        out_name = os.path.join("outputs", str(out_path) + "___" + no + title + ".png")
        plt.savefig(out_name, dpi=300)
    plt.show()
    plt.figure(figsize=(5, 8))
    plt.plot(range(peaks[2] - peaks[1])/samplingrate, pulses[peaks[1]:peaks[2]], color="orange")
    plt.xlabel("Standard Time (seconds)")
    plt.ylabel("Standard Pulse amplitude (mA)")
    title = "Individual Pulses"
    plt.title(title)
    if is_out:
        no = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        out_name = os.path.join("outputs", str(out_path) + "___" + no + title + ".png")
        plt.savefig(out_name, dpi=300)
    plt.show()
    return lines


def joint_plot(signals, is_out=False, out_path="figs"):
    """
    联合绘制多个图
    :param signals:     初步处理的波形数据
    :param is_out:      是否输出
    :param out_path:    输出路径
    :return:
    """
    feature_points = signals["Feature Points"]
    cleaned_pulses = feature_points["Cleaned Pulses"]
    # X轴
    if signals["Sampling Rate"] is not None:
        x_axis = np.linspace(0, cleaned_pulses.shape[0] / signals["Sampling Rate"], cleaned_pulses.shape[0])
    else:
        x_axis = np.arange(0, cleaned_pulses.shape[0])

    plt.figure(figsize=(20, 8))
    plt.plot(x_axis, signals["Standard Pulses"], color="#B0BEC5", label="Raw Data", zorder=1)
    # plt.plot(x_axis, signals["Feature Points"]["Cleaned Pulses"], color="#FB661C", label="Cleaned Data", zorder=1, linewidth=1.5)
    plt.scatter(x_axis[signals["Feature Points"]["Peaks"]],
                np.array(signals["Standard Pulses"])[signals["Feature Points"]["Peaks"]], color="#DC143C",
                label="Peaks", zorder=2)
    plt.scatter(x_axis[signals["Feature Points"]["Notches"]],
                np.array(signals["Standard Pulses"])[signals["Feature Points"]["Notches"]], color="#7FFFAA",
                label="Notches", zorder=2)
    plt.scatter(x_axis[signals["Feature Points"]["Other Peaks"]],
                np.array(signals["Standard Pulses"])[signals["Feature Points"]["Other Peaks"]], color="#FBB41C",
                label="Other Peaks", zorder=2)
    plt.scatter(x_axis[signals["Feature Points"]["Other Notches"]],
                np.array(signals["Standard Pulses"])[signals["Feature Points"]["Other Notches"]], color="#4169E1",
                label="Other Notches", zorder=2)
    plt.legend(loc="upper right")
    title = "Raw Cleaned and Peaks Signal"
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Standard Pulse amplitude (mA)")
    if is_out:
        no = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        out_name = os.path.join("outputs", str(out_path) + "___" + no + title + ".png")
        plt.savefig(out_name, dpi=300)
    plt.show()

    plt.figure(figsize=(15, 8))
    ppg_rate_mean = signals["Feature Points"]["HR_Time"]
    plt.plot(signals["Feature Points"]["Heart Rates"], color="#FB1CF0", label="Heart Rate", linewidth=1.5)
    plt.axhline(y=ppg_rate_mean, label="Mean Heart Rate=" + str(np.round(ppg_rate_mean, 2)), linestyle="--",
                color="#D60574")
    plt.annotate("Mean Heart Rate=" + str(np.round(ppg_rate_mean, 2)), xy=(0, int(ppg_rate_mean)),
                 xytext=(-1.5, int(ppg_rate_mean) + 2), color="#D60574")
    plt.legend(loc="upper right")
    title = "Heart Rate Varability (HRV)"
    plt.title("Heart Rate Varability (HRV)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Heart Rate")
    if is_out:
        no = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        out_name = os.path.join("outputs", str(out_path) + "___" + no + title + ".png")
        plt.savefig(out_name, dpi=300)
    plt.show()
    pass


def common_plot(x, y, x_label="Time (seconds)", y_label="Pulse amplitude (mA)", title="Raw Data", is_out=True,
                args=None, sampling_rate=None, out_path="figs"):
    """
    基础的绘图方法  同plt 只不过针对于心率的绘制了
    :param x:
    :param y:
    :param x_label:
    :param y_label:
    :param title:
    :param is_out:
    :param args:
    :param sampling_rate:
    :param out_path:
    :return:
    """
    if x is None:
        # X轴
        if sampling_rate is not None:
            x = np.linspace(0, y.shape[0] / sampling_rate, y.shape[0])
        else:
            x = np.arange(0, y.shape[0])
    plt.figure(figsize=(20, 8))
    plt.plot(x, y, label=title)
    plt.legend(loc="upper right")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if is_out:
        no = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        out_name = os.path.join("outputs", str(out_path) + "___" + no + title + ".png")
        plt.savefig(out_name, dpi=300)
    plt.show()
    pass


def signals_analysis(signals, is_out=False, is_show=False, out_path="figs"):
    """
    数据的分析 主要是特征参数的获取
    :param signals:     初步处理过的数据
    :param is_out:
    :param is_show:
    :param out_path:
    :return:
    """
    sampling_rate = signals["Sampling Rate"]
    feature_points = signals["Feature Points"]
    cleaned_pulses = feature_points["Cleaned Pulses"]
    peaks = feature_points["Peaks"]

    title = "Time Analysis"
    hrv_time = nk.hrv_time(peaks, sampling_rate, show=is_show)
    if is_out:
        no = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        out_name = os.path.join("outputs", str(out_path) + "___" + no + title + ".png")
        plt.savefig(out_name, dpi=300)
    if is_show:
        plt.show()

    title = "Frequency Analysis"
    hrv_freq = nk.hrv_frequency(peaks, sampling_rate, show=is_show)
    if is_out:
        no = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        out_name = os.path.join("outputs", str(out_path) + "___" + no + title + ".png")
        plt.savefig(out_name, dpi=300)
    if is_show:
        plt.show()

    title = "Nonlinear Analysis"
    hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate, show=is_show)
    if is_out:
        no = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        out_name = os.path.join("outputs", str(out_path) + "___" + no + title + ".png")
        plt.savefig(out_name, dpi=300)
    if is_show:
        plt.show()

    # 傅里叶分析  对应给的文章
    xFFT = np.abs(np.fft.rfft(cleaned_pulses) / len(cleaned_pulses))
    xFFT = xFFT[:600]
    xFreqs = np.linspace(0, sampling_rate // 2, len(cleaned_pulses) // 2 + 1)
    xFreqs = xFreqs[:600]
    # 滤波处理 平滑去噪 只处理前200个信号即可
    # TODO 去噪方法可以调节
    cleaned_xFFT = nk.signal_smooth(xFFT, method="loess")
    # 计算特征值
    # F1值
    cleaned_FFT = cleaned_xFFT.copy()
    locmax, props = spsg.find_peaks(cleaned_xFFT)
    hr_hz_index = np.argmax(cleaned_xFFT)
    f1 = np.argmax(xFFT)
    fmax = np.argmax(cleaned_xFFT[locmax])
    cleaned_xFFT[locmax[fmax]] = np.min(cleaned_xFFT)
    f2s = np.argmax(cleaned_xFFT[locmax])
    if f2s - fmax != 1:
        hr_hz_index = locmax[0] + int(np.sqrt(locmax[1] - locmax[0]))
    # F2值
    f2 = locmax[np.argmax(cleaned_xFFT[locmax])]
    F1 = np.round(xFreqs[f1], 2)
    F2 = np.round(xFreqs[f2], 2)
    # 相位差
    F2_F1 = F2 - F1
    # 心率
    HR_FFT = xFreqs[hr_hz_index] * 60
    print(HR_FFT)

    if is_show:
        plt.plot(xFreqs, cleaned_FFT)
        plt.scatter(xFreqs[f1], cleaned_FFT[f1], color="red", label="F1 = " + str(F1) + "HZ")
        plt.scatter(xFreqs[f2], cleaned_FFT[f2], color="orange", label="F2 = " + str(F2) + "HZ")
        plt.legend(loc="upper right")
        plt.ylabel("Power")
        plt.xlabel("Freq(Hz)")
        title = "FFT analysis"
        plt.title(title)
        if is_out:
            no = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            out_name = os.path.join("outputs", str(out_path) + "___" + no + title + ".png")
            plt.savefig(out_name, dpi=300)
        plt.show()
    hrv_new = {"Power": xFFT, "Freq": cleaned_FFT, "X":xFreqs, "F1": F1, "F2": F2, "F2_F1": F2_F1, "HR_FFT": HR_FFT}
    return {"HRV New": hrv_new, "HRV Time": hrv_time, "HRV Frequency": hrv_freq, "HRV Nonlinear": hrv_nonlinear}


def signals_plot(signals, is_out=False, out_path="figs"):
    """
    数据的绘制
    :param signals:
    :param is_out:
    :param out_path:
    :return:
    """
    feature_points = signals["Feature Points"]
    cleaned_pulses = feature_points["Cleaned Pulses"]
    # X轴
    if signals["Sampling Rate"] is not None:
        x_axis = np.linspace(0, cleaned_pulses.shape[0] / signals["Sampling Rate"], cleaned_pulses.shape[0])
    else:
        x_axis = np.arange(0, cleaned_pulses.shape[0])
    # 绘制原始数据
    common_plot(signals["Raw Times"], signals["Raw Pulses"], title="Raw Data", is_out=is_out, out_path=out_path)
    common_plot(signals["Selected Times"], signals["Selected Pulses"], title="Selected Data", is_out=is_out,
                out_path=out_path)
    common_plot(x_axis, signals["Feature Points"]["Cleaned Pulses"], title="Cleaned Data", is_out=is_out,
                out_path=out_path)
    joint_plot(signals, is_out=is_out, out_path=out_path)
    segment_pulses(signals["Standard Pulses"], signals["Feature Points"]["Peaks"], samplingrate=signals["Sampling Rate"], is_out=is_out, out_path=out_path)
    pass


def signal_process(pulses, sampling_rate):
    """
    数据的初步处理分析 寻找峰值 初步时域分析
    :param pulses:          标准化的数据
    :param sampling_rate:   采样率
    :return:
    """
    # 数据检查 是否是1维格式
    pulses = nk.as_vector(pulses)
    # 数据去噪 低通滤波去噪 bessel
    cleaned_pulses = nk.signal_filter(
        pulses, sampling_rate=sampling_rate, lowcut=0.5, highcut=8, order=3, method="bessel"
    )
    # common_plot(None, cleaned_pulses, title="Cleand Data", sampling_rate=sampling_rate)
    feature_points = find_feature_points(cleaned_pulses, sampling_rate)
    # 寻找其他特征点
    peaks = feature_points["Peaks"]
    # 计算心率
    rates = nk.signal_rate(peaks, sampling_rate=sampling_rate, desired_length=None)
    feature_points["Heart Rates"] = rates
    feature_points["HR_Time"] = np.mean(rates)
    print(np.round(np.mean(rates), 2))
    return feature_points
    pass


def find_feature_points(signal, sampling_rate=50, threshold=0.5):
    """
    寻找峰值以及槽点
    :param signal:          初始数据 已经去噪
    :param sampling_rate:   采样率
    :param threshold:       过近的点可以认为是异常值 给予一个threshold去除异常识别的峰值点
    :return:
    """
    origin_signal = signal.copy()
    peaks = nk.ppg_findpeaks(signal, sampling_rate=sampling_rate)
    # 先找到峰值点
    locmax, props = spsg.find_peaks(signal)
    notches = []
    other_notches = []
    # 两种方法找到的共同峰值点
    all_peaks = np.intersect1d(locmax, peaks["PPG_Peaks"])
    mean_T = np.mean(np.diff(all_peaks))
    # 真正的峰值点
    true_peaks = [all_peaks[0]]
    # 周期
    T = []
    for i in range(len(all_peaks) - 1):
        if all_peaks[i + 1] - all_peaks[i] < threshold * mean_T:
            continue
        true_peaks.append(all_peaks[i + 1])
    for i in range(len(true_peaks) - 1):
        notches.append(true_peaks[i] + np.argmin(origin_signal[true_peaks[i]:true_peaks[i + 1]]))
        T.append(true_peaks[i + 1] - true_peaks[i])

    for i in range(len(locmax) - 1):
        other_notches.append(locmax[i] + np.argmin(origin_signal[locmax[i]:locmax[i + 1]]))
    other_peaks = list(set(locmax) - set(true_peaks))
    other_notches = list(set(other_notches) - set(notches))
    other_peaks = [i for i in other_peaks if origin_signal[i] > 0]
    other_notches = [i for i in other_notches if origin_signal[i] > 0]

    return {"Peaks": true_peaks, "Other Peaks": other_peaks, "Other Notches": other_notches, "Notches": notches, "T": T,
            "Cleaned Pulses": origin_signal}
