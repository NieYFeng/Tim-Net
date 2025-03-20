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
    "model_glob": "10-fold_weights_best_*.hdf5"  # 模型文件命名模式
}


def build_timnet_model(input_shape, num_classes):
    """构建TIMNET模型结构"""
    inputs = Input(shape=(input_shape[0], input_shape[1]))
    multi_decision = TIMNET(
        nb_filters=64,  # 根据训练参数调整
        kernel_size=2,  # 根据训练参数调整
        nb_stacks=1,    # 根据训练参数调整
        dilations=[1, 2, 4, 8],  # 根据训练参数调整
        dropout_rate=0.1,  # 根据训练参数调整
        activation='relu',  # 根据训练参数调整
        return_sequences=True
    )(inputs)
    decision = Dense(64, activation='relu')(multi_decision)  # 假设中间层
    predictions = Dense(num_classes, activation='softmax')(decision)
    model = Model(inputs=inputs, outputs=predictions)
    return model


def load_models(models_dir: str, input_shape, num_classes) -> list:
    """加载全部10个交叉验证模型"""
    model_paths = list(Path(models_dir).glob(CASIA_SETTINGS["model_glob"]))
    assert len(model_paths) == 10, f"应找到10个模型，实际找到{len(model_paths)}"

    models = []
    for path in sorted(model_paths):  # 按fold顺序加载
        try:
            model = build_timnet_model(input_shape, num_classes)
            model.load_weights(str(path))  # 加载权重
            models.append(model)
            print(f"成功加载模型权重: {path.name}")
        except Exception as e:
            raise RuntimeError(f"加载{path}失败: {str(e)}")
    return models


def ensemble_predict(models: list, input_data: np.ndarray) -> dict:
    """集成预测（软投票）"""
    all_probs = []
    for model in models:
        prob = model.predict(input_data, verbose=0)[0]
        all_probs.append(prob)

    # 计算平均概率
    avg_probs = np.mean(all_probs, axis=0)
    return {label: float(p) for label, p in
            zip(CASIA_SETTINGS["labels"], avg_probs)}


def main(audio_path: str, models_dir: str):
    # 特征提取
    signal, _ = librosa.load(audio_path, sr=CASIA_SETTINGS["sr"])
    signal = librosa.util.fix_length(signal, size=CASIA_SETTINGS["sr"] * CASIA_SETTINGS["duration"])

    # MFCC特征
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

    input_tensor = mfcc[np.newaxis, ..., np.newaxis]  # (1, 96, 39, 1)

    # 加载模型
    input_shape = (CASIA_SETTINGS["timesteps"], CASIA_SETTINGS["n_mfcc"])
    num_classes = len(CASIA_SETTINGS["labels"])
    models = load_models(models_dir, input_shape, num_classes)

    # 集成预测
    results = ensemble_predict(models, input_tensor)

    # 显示结果
    print("\n集成预测结果（10折平均）：")
    for emo, prob in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{emo.upper():<10} {prob * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="测试音频路径")
    parser.add_argument("--models", required=True, help="包含10个模型的目录")
    args = parser.parse_args()

    main(args.audio, args.models)