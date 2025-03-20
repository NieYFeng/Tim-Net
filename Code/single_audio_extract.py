import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse


# 特征提取函数（MFCC）
def extract_features(audio_path):
    # 加载音频文件
    audio, sr = librosa.load(audio_path, sr=None)

    # 提取梅尔频率倒谱系数（MFCC）
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # 提取13个MFCC系数

    # 返回特征的平均值作为最终特征
    return np.mean(mfccs.T, axis=0)


# 测试单一音频的分类
def test_single_audio(model_folder, audio_path):
    # 提取音频特征
    features = extract_features(audio_path)
    features = np.expand_dims(features, axis=0)  # 增加batch维度

    # 列出所有的权重文件
    weight_files = [f for f in os.listdir(model_folder) if f.endswith('.hdf5')]

    # 存储所有折叠模型的预测结果
    all_predictions = []

    for weight_file in weight_files:
        model_path = os.path.join(model_folder, weight_file)

        # 加载模型权重
        model = load_model(model_path)

        # 进行预测
        prediction = model.predict(features)
        all_predictions.append(prediction)

        print(f"Prediction for model {weight_file}: {prediction}")

    # 对所有预测进行平均
    average_prediction = np.mean(all_predictions, axis=0)
    print(f"Average prediction: {average_prediction}")

    # 根据最大概率确定分类标签
    predicted_class = np.argmax(average_prediction)
    print(f"Predicted class: {predicted_class}")


if __name__ == '__main__':
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Test single audio file using multiple model weights")
    parser.add_argument('--audio', type=str, required=True, help="Path to the audio file")
    parser.add_argument('--models', type=str, required=True, help="Folder containing the model weights")

    # 解析命令行参数
    args = parser.parse_args()

    # 测试单一音频
    test_single_audio(args.models, args.audio)
