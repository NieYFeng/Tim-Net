import numpy as np
import librosa
import tensorflow as tf
import argparse
from pathlib import Path
from TIMNET import TIMNET  # 导入TIMNET模型结构
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# CASIA 配置
CASIA_SETTINGS = {
    "sr": 16000,  # 采样率
    "duration": 6,  # 音频长度（秒）
    "n_mfcc": 39,  # MFCC系数
    "timesteps": 96,  # 时间步长（从CASIA_MFCC_96推断）
    "labels": ("angry", "fear", "happy", "neutral", "sad", "surprise"),
}


class WeightLayer(tf.keras.layers.Layer):
    """自定义权重层"""

    def __init__(self, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[1], 1),
            initializer='uniform',
            trainable=True
        )
        super(WeightLayer, self).build(input_shape)

    def call(self, x):
        tempx = tf.transpose(x, [0, 2, 1])
        x = tf.matmul(tempx, self.kernel)
        x = tf.squeeze(x, axis=-1)
        return x


def build_timnet_model(input_shape, num_classes, args):
    """构建TIMNET模型结构"""
    inputs = Input(shape=(input_shape[0], input_shape[1]))

    # TIMNET 层
    multi_decision = TIMNET(
        nb_filters=args.filter_size,  # 滤波器数量
        kernel_size=args.kernel_size,  # 卷积核大小
        nb_stacks=args.stack_size,  # 堆叠层数
        dilations=args.dilation_size,  # 膨胀率
        dropout_rate=args.dropout,  # Dropout 率
        activation=args.activation,  # 激活函数
        return_sequences=True  # 返回序列
    )(inputs)

    # 自定义权重层
    decision = WeightLayer()(multi_decision)

    # 输出层
    predictions = Dense(num_classes, activation='softmax')(decision)

    # 构建模型
    model = Model(inputs=inputs, outputs=predictions)
    return model


def extract_features(audio_path: str):
    """提取音频特征"""
    # 加载音频并调整长度
    signal, _ = librosa.load(audio_path, sr=CASIA_SETTINGS["sr"])
    signal = librosa.util.fix_length(signal, size=CASIA_SETTINGS["sr"] * CASIA_SETTINGS["duration"])

    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=CASIA_SETTINGS["sr"],
        n_mfcc=CASIA_SETTINGS["n_mfcc"],
        n_fft=512,
        hop_length=256
    ).T

    # 在线标准化（使用当前音频的统计量）
    mfcc_mean = np.mean(mfcc, axis=0)
    mfcc_std = np.std(mfcc, axis=0)
    mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-8)

    # 调整时间步
    if mfcc.shape[0] < CASIA_SETTINGS["timesteps"]:
        mfcc = np.pad(mfcc, ((0, CASIA_SETTINGS["timesteps"] - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:CASIA_SETTINGS["timesteps"]]

    # 添加批次和通道维度
    input_tensor = mfcc[np.newaxis, ..., np.newaxis]  # (1, 96, 39, 1)
    return input_tensor


def predict_single_audio(audio_path: str, model_weights_path: str, args):
    """预测单一音频"""
    # 构建模型结构
    input_shape = (CASIA_SETTINGS["timesteps"], CASIA_SETTINGS["n_mfcc"])
    num_classes = len(CASIA_SETTINGS["labels"])
    model = build_timnet_model(input_shape, num_classes, args)

    # 加载权重
    model.load_weights(model_weights_path)
    print(f"成功加载模型权重: {model_weights_path}")

    # 提取特征
    input_tensor = extract_features(audio_path)

    # 执行预测
    prediction = model.predict(input_tensor, verbose=0)[0]
    predicted_class = np.argmax(prediction)
    predicted_label = CASIA_SETTINGS["labels"][predicted_class]

    # 显示结果
    print("\n预测结果：")
    for label, prob in zip(CASIA_SETTINGS["labels"], prediction):
        print(f"{label.upper():<10} {prob * 100:.2f}%")
    print(f"\n预测类别: {predicted_label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="测试音频路径")
    parser.add_argument("--weights", required=True, help="模型权重文件路径")
    parser.add_argument("--filter_size", type=int, default=64, help="滤波器数量")
    parser.add_argument("--kernel_size", type=int, default=2, help="卷积核大小")
    parser.add_argument("--stack_size", type=int, default=1, help="堆叠层数")
    parser.add_argument("--dilation_size", type=list, default=[1, 2, 4, 8], help="膨胀率")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 率")
    parser.add_argument("--activation", type=str, default="relu", help="激活函数")
    args = parser.parse_args()

    predict_single_audio(args.audio, args.weights, args)