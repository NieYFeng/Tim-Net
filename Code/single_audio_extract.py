# ensemble_casia_test.py
import numpy as np
import librosa
import tensorflow as tf
import argparse
from pathlib import Path

# CASIA 配置
CASIA_SETTINGS = {
    "sr": 16000,  # 采样率
    "duration": 6,  # 音频长度（秒）
    "n_mfcc": 39,  # MFCC系数
    "timesteps": 96,  # 时间步长（从CASIA_MFCC_96推断）
    "labels": ("angry", "fear", "happy", "neutral", "sad", "surprise"),
    "model_glob": "10-fold_weights_best_*.hdf5"  # 模型文件命名模式
}


def load_models(models_dir: str) -> list:
    """加载全部10个交叉验证模型"""
    model_paths = list(Path(models_dir).glob(CASIA_SETTINGS["model_glob"]))
    assert len(model_paths) == 10, f"应找到10个模型，实际找到{len(model_paths)}"

    models = []
    for path in sorted(model_paths):  # 按fold顺序加载
        try:
            models.append(tf.keras.models.load_model(str(path)))
            print(f"成功加载模型: {path.name}")
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

    # 标准化（需使用训练时的全局统计量）
    mfcc = (mfcc - np.load("casia_mfcc_mean.npy")) / (np.load("casia_mfcc_std.npy") + 1e-8)

    # 调整时间步
    if mfcc.shape[0] < CASIA_SETTINGS["timesteps"]:
        mfcc = np.pad(mfcc, ((0, CASIA_SETTINGS["timesteps"] - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:CASIA_SETTINGS["timesteps"]]

    input_tensor = mfcc[np.newaxis, ..., np.newaxis]  # (1, 96, 39, 1)

    # 加载模型
    models = load_models(models_dir)

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