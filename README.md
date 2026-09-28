# 使用卷積神經網路進行眼底影像黃斑部病變分類之研究

深度學習 第三次作業

## 1. 問題

黃斑部病變（AMD）等眼疾若未及早診斷，可能造成嚴重視力受損。本作業以眼底彩色影像建立 CNN 自動分類模型，將影像分為四類，並探討：

1. 不同 CNN 架構（SimpleCNN、ResNet18、MobileNetV2）的效能差異
2. 輸入影像尺寸對效能的影響
3. 遷移學習時凍結骨幹（Freeze）與全模型微調（Fine-tune）的差異
4. 資料增強對泛化能力的影響

## 2. 資料

**Macular Degeneration Disease Dataset**（Kaggle）
<https://www.kaggle.com/datasets/orvile/macular-degeneration-disease-dataset>

| 類別 | 說明 | train | valid |
|---|---|---|---|
| amd | 黃斑部病變 | 394 | 100 |
| cataract | 白內障 | 400 | 100 |
| diabetes | 糖尿病視網膜病變 | 400 | 100 |
| normal | 正常眼底 | 400 | 100 |

資料集約 600MB，未放入此 repo。資料切分方式：

- `train/` 再以 80 / 20 隨機切成訓練集與驗證集（驗證集用於早停）
- `valid/` 作為獨立測試集，所有結果皆以此計算

## 3. 方法

**前處理**
- Resize 至指定尺寸（預設 224×224），轉為 Tensor，以 ImageNet mean/std 正規化
- 使用 `torchvision.datasets.ImageFolder` 依資料夾自動標註類別
- 資料增強（僅訓練集）：RandomHorizontalFlip(p=0.5)、RandomRotation(15°)

**模型**

| 模型 | 說明 |
|---|---|
| SimpleCNN | 3 × [Conv3×3 → BatchNorm → ReLU → MaxPool2×2]（32→64→128 通道）→ GAP → FC(128) → Dropout(0.3) → FC(4) |
| ResNet18 | ImageNet 預訓練，替換最後 fc 為 4 類 |
| MobileNetV2 | ImageNet 預訓練，替換分類層為 4 類 |

## 4. 實驗設定

**共同訓練參數**

| 項目 | 設定 |
|---|---|
| Loss | CrossEntropyLoss |
| Optimizer | Adam，learning rate = 1e-4 |
| Epochs | 最多 50 |
| Early stopping | 驗證 loss 連續 10 epoch 未改善即停止 |
| Batch size | 128 |
| Random seed | 42（固定 Python / NumPy / PyTorch，cudnn deterministic） |
| 裝置 | 自動偵測 CUDA GPU |

**實驗**

| 實驗 | 固定條件 | 比較變因 |
|---|---|---|
| 一 | img_size 224、有增強 | SimpleCNN / ResNet18 / MobileNetV2 |
| 二 | ResNet18 | img_size 128 / 224 / 256 |
| 三 | ResNet18、224 | Freeze backbone（只訓練 fc） vs Fine-tune |
| 四 | ResNet18、224 | 無增強 vs 有增強（Flip + Rotation） |

評估指標：Accuracy、Macro Precision / Recall / F1、Confusion Matrix、訓練時間。

## 5. 結果（測試集 `valid/`）

| 實驗 | 設定 | Accuracy | Macro F1 | 停止 epoch |
|---|---|---|---|---|
| 一 | SimpleCNN | 0.725 | 0.721 | 50 |
| 一 | MobileNetV2 | 0.940 | 0.939 | 19 |
| 一 | ResNet18 | 0.950 | 0.950 | 17 |
| 二 | ResNet18, 128 | 0.898 | 0.894 | 13 |
| 二 | ResNet18, 256 | 0.940 | 0.940 | 14 |
| 三 | ResNet18, Freeze backbone | 0.803 | 0.799 | 50 |
| 三 | ResNet18, Fine-tune | 0.950 | 0.950 | 17 |
| 四 | ResNet18, 有資料增強 | **0.963** | **0.962** | 21 |

**結論**
- SimpleCNN 欠擬合；預訓練模型（ResNet18、MobileNetV2）大幅領先
- 224 為最佳輸入尺寸，128 會損失病灶細節，256 無額外提升
- Fine-tune 明顯優於 Freeze backbone（0.950 vs 0.803）
- 資料增強縮小 train / valid 差距，達到最佳 0.963

詳細曲線、混淆矩陣與分析請見 `第三次深度學習作業 (1) - 複製.pdf`。

## 6. 如何重現

1. 安裝套件（建議使用有 CUDA 的 GPU 環境）：

   ```bash
   pip install torch torchvision numpy pandas scikit-learn matplotlib jupyter
   ```

2. 從上方 Kaggle 連結下載資料集，解壓後讓 `train/`、`valid/` 與 `CNN.ipynb` 位於同一層：

   ```
   DeepLearningHW3/
   ├── CNN.ipynb
   ├── train/{amd,cataract,diabetes,normal}/
   └── valid/{amd,cataract,diabetes,normal}/
   ```

3. 開啟 `CNN.ipynb`，由上而下依序執行所有 cell（`DATA_ROOT = "."`）。
4. 已固定 random seed = 42；因 GPU 運算的非決定性，數值可能有些微差異。

## 檔案

- `CNN.ipynb`：主程式（前處理、模型定義、訓練與評估）
- `第三次深度學習作業 (1) - 複製.pdf`：書面報告
