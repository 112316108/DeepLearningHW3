# 使用卷積神經網路進行眼底影像黃斑部病變分類之研究

深度學習 第三次作業

使用 PyTorch 比較 SimpleCNN、ResNet18、MobileNetV2 對眼底影像進行四類分類。

類別：`amd`、`cataract`、`diabetes`、`normal`

## 檔案

- `CNN.ipynb`：主程式（資料前處理、模型定義、訓練與評估）
- `第三次深度學習作業 (1) - 複製.pdf`：書面報告（實驗設計與結果分析）

## 資料集

資料集約 600MB，未放入此 repo，請從 Kaggle 下載：

**Macular Degeneration Disease Dataset**
<https://www.kaggle.com/datasets/orvile/macular-degeneration-disease-dataset>

下載後解壓縮，放在與 `CNN.ipynb` 同一層：

```
DeepLearningHW3/
├── CNN.ipynb
├── train/
│   ├── amd/        (394 張)
│   ├── cataract/   (400 張)
│   ├── diabetes/   (400 張)
│   └── normal/     (400 張)
└── valid/
    ├── amd/        (100 張)
    ├── cataract/   (100 張)
    ├── diabetes/   (100 張)
    └── normal/     (100 張)
```

## 主要結果（驗證集）

| 實驗 | 模型 | img_size | Accuracy | F1 (macro) |
|---|---|---|---|---|
| 模型比較 | SimpleCNN | 224 | 0.725 | 0.721 |
| 模型比較 | MobileNetV2 | 224 | 0.940 | 0.939 |
| 模型比較 | ResNet18 | 224 | 0.950 | 0.950 |
| 輸入尺寸 | ResNet18 | 128 / 256 | 0.898 / 0.940 | 0.894 / 0.940 |
| Freeze backbone | ResNet18 | 224 | 0.803 | 0.799 |
| 資料增強 (Flip + Rotation) | ResNet18 | 224 | **0.963** | **0.962** |

## 環境

- Python 3、PyTorch、torchvision
- numpy、pandas、scikit-learn、matplotlib

開啟 `CNN.ipynb` 依序執行即可，有 CUDA GPU 時會自動使用。
