# Deep Learning with PyTorch: Custom Dataset and Neural Network Training

Dự án này triển khai một hệ thống deep learning hoàn chỉnh sử dụng PyTorch, bao gồm việc xây dựng custom dataset và huấn luyện mạng neural network cho bài toán phân loại hình ảnh.

## 📋 Tổng quan

Repository này chứa code để:
- Xây dựng custom PyTorch Dataset cho CIFAR-10 và Animal datasets
- Triển khai và huấn luyện neural network
- Đánh giá hiệu suất model với classification report

## 🗂️ Cấu trúc dự án

```
├── dataset.py              # Custom dataset classes
├── train_neuralnetwork.py   # Training script chính
├── models.py               # Neural network architecture (cần được thêm)
└── data/                   # Thư mục chứa dữ liệu
    ├── cifar-10-batches-py/ # CIFAR-10 dataset
    └── animals/             # Animal dataset
        ├── train/
        └── test/
```

## 🚀 Tính năng chính

### 1. Custom Dataset Classes

#### CIFARDataset
- **Mục đích**: Xử lý CIFAR-10 dataset từ batch files
- **Đặc điểm**:
  - Đọc dữ liệu từ pickle files (data_batch_1 đến data_batch_5 cho training, test_batch cho testing)
  - Reshape images từ flat array thành (3, 32, 32) RGB format
  - Normalize pixel values về range [0, 1]
  - Hỗ trợ cả training và test splits

#### AnimalDataset  
- **Mục đích**: Xử lý custom animal dataset
- **10 loại động vật**: butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel
- **Đặc điểm**:
  - Đọc images từ folder structure
  - Hỗ trợ custom transforms (resize, normalization, etc.)
  - Tự động label encoding cho các categories
  - Tương thích với PIL Image processing

### 2. Training Pipeline

#### Cấu hình Training
```python
- Epochs: 100
- Batch size: 64
- Optimizer: SGD với learning rate 1e-3, momentum 0.9
- Loss function: CrossEntropyLoss
- GPU support: Tự động detect và sử dụng CUDA nếu có
```

#### Training Process
1. **Forward Pass**: Tính predictions từ model
2. **Loss Calculation**: Sử dụng CrossEntropyLoss
3. **Backward Pass**: Backpropagation với gradient descent
4. **Evaluation**: Đánh giá trên test set sau mỗi epoch
5. **Metrics**: Classification report với precision, recall, f1-score

## 📦 Dependencies

```python
torch
torchvision
numpy
opencv-python (cv2)
Pillow (PIL)
scikit-learn
pickle
```

## 🔧 Cài đặt và sử dụng

### 1. Clone repository
```bash
git clone https://github.com/QuyDatSadBoy/Deeplearning_Build_Pytorch_Dataset_and_train_neuralnetwork.git
cd Deeplearning_Build_Pytorch_Dataset_and_train_neuralnetwork
```

### 2. Cài đặt dependencies
```bash
pip install torch torchvision numpy opencv-python Pillow scikit-learn
```

### 3. Chuẩn bị dữ liệu
```
data/
├── cifar-10-batches-py/
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   └── test_batch
└── animals/
    ├── train/
    │   ├── butterfly/
    │   ├── cat/
    │   └── ...
    └── test/
        ├── butterfly/
        ├── cat/
        └── ...
```

### 4. Test dataset
```bash
python dataset.py
```

### 5. Huấn luyện model
```bash
python train_neuralnetwork.py
```

## 🏗️ Kiến trúc Model

Model sử dụng `SimpleNeuralNetwork` class với:
- Input: Images với shape phù hợp với dataset
- Output: 10 classes (cho CIFAR-10 hoặc animal dataset)
- Architecture details: *Cần kiểm tra file models.py*

## 📊 Đánh giá và Metrics

Sau mỗi epoch, system sẽ output:
- Classification report với precision, recall, f1-score cho từng class
- Overall accuracy
- Support cho từng class

## 🔄 Workflow

1. **Data Loading**: CIFARDataset/AnimalDataset load và preprocess data
2. **Model Initialization**: Khởi tạo SimpleNeuralNetwork với 10 classes
3. **Training Loop**: 
   - Forward pass qua model
   - Tính loss với CrossEntropyLoss
   - Backward pass và optimizer step
4. **Evaluation**: Test trên validation set sau mỗi epoch
5. **Metrics**: In classification report chi tiết

## ⚡ Performance Features

- **GPU Support**: Tự động sử dụng CUDA nếu available
- **Multi-threading**: DataLoader với num_workers=4 để tăng tốc data loading
- **Memory Efficient**: Drop_last=True cho training để tránh batch size inconsistency
- **Evaluation Mode**: Sử dụng torch.no_grad() trong evaluation để tiết kiệm memory

## 🎯 Use Cases

1. **Học tập Deep Learning**: Hiểu cách xây dựng dataset và training pipeline
2. **Computer Vision**: Template cho image classification tasks
3. **Custom Dataset**: Mở rộng cho datasets khác
4. **Prototype**: Base code cho các dự án vision lớn hơn

## 🔮 Tương lai phát triển

- [ ] Thêm data augmentation
- [ ] Implement more complex model architectures
- [ ] Add validation split
- [ ] Tensorboard integration
- [ ] Model checkpointing
- [ ] Hyperparameter tuning
- [ ] Transfer learning support

## 👤 Tác giả

**QuyDatSadBoy**
- GitHub: [@QuyDatSadBoy](https://github.com/QuyDatSadBoy)


