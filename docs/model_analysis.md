# Phân tích mô hình

## 1. Bài toán và lựa chọn mô hình

Bài toán là phân loại ảnh Oxford-IIIT Pet thành 37 giống chó và mèo. Nhóm sử dụng VGG16 và DenseNet121 đã được pretrained trên ImageNet, sau đó thay classifier cuối để tạo 37 đầu ra.

Hai kiến trúc được chọn để đối chiếu hai cách trích xuất đặc trưng khác nhau: VGG16 xử lý đặc trưng qua các tầng tích chập tuần tự, còn DenseNet121 truyền và tái sử dụng đặc trưng giữa các lớp trong dense block. Cả hai cùng dùng một bộ dữ liệu, cách tiền xử lý và quy trình huấn luyện để việc so sánh có ý nghĩa.

## 2. Cách mô hình hoạt động trong bài

Ảnh sau tiền xử lý được đưa qua backbone pretrained để trích xuất đặc trưng. Classifier cuối nhận các đặc trưng này và xuất ra 37 giá trị (logits), mỗi giá trị tương ứng với một giống. Trong lúc huấn luyện, CrossEntropyLoss đo sai khác giữa logits và nhãn thật; Adam cập nhật các tham số được phép train.

Mặc định, các tầng trích xuất đặc trưng được đóng băng và chỉ classifier mới được cập nhật. Cách này tận dụng kiến thức từ ImageNet, đồng thời hạn chế số tham số cần học trên bộ dữ liệu mới. Có thể bật `--fine-tune-last-block` để mở thêm khối đặc trưng cuối, nhưng tùy chọn này không được sử dụng trong các kết quả được phân tích ở đây.

### Đoạn code chính

Phần khởi tạo từng kiến trúc được tách riêng: `src/vgg16_model.py` và `src/densenet121_model.py`. `src/models.py` giữ hàm `build_model()` để `main.py` tiếp tục chọn model theo tên như trước.

```python
# src/vgg16_model.py
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
in_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features, num_classes)
for parameter in model.features.parameters():
    parameter.requires_grad = False
for parameter in model.classifier[:6].parameters():
    parameter.requires_grad = False
for parameter in model.classifier[6].parameters():
    parameter.requires_grad = True

# src/densenet121_model.py
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, num_classes)
for parameter in model.features.parameters():
    parameter.requires_grad = False
for parameter in model.classifier.parameters():
    parameter.requires_grad = True
```

Với VGG16, lớp Linear cuối được thay để xuất 37 lớp; các tầng convolution và các lớp ẩn của classifier được giữ nguyên. Với DenseNet121, classifier Linear cuối được thay mới. Trong cả hai trường hợp, backbone pretrained bị đóng băng theo mặc định.

## 3. Kết quả thực nghiệm

Các số liệu dưới đây là từ hai cấu hình được chọn bằng validation trong `outputs/reduced-experiments/results.csv`. Batch size đều là 16; learning rate được chọn là 0.001.

| Mô hình | Validation loss | Validation accuracy | Test loss | Test accuracy |
| --- | ---: | ---: | ---: | ---: |
| VGG16 | 0.2831 | 91.34% | 0.3450 | 88.80% |
| DenseNet121 | 0.2767 | 91.88% | 0.3236 | 91.31% |

Trong lần chạy này, DenseNet121 cao hơn VGG16 **2.51 điểm phần trăm** về test accuracy và có test loss thấp hơn. Validation accuracy của hai mô hình khá gần nhau, nhưng DenseNet121 nhỉnh hơn ở cả validation loss và accuracy.

## 4. Nhận xét

- **DenseNet121:** cho kết quả tốt hơn trong cấu hình và split hiện tại. Các dense block cho phép các tầng sử dụng lại đặc trưng đã được trích xuất trước đó.
- **VGG16:** có cấu trúc tuần tự dễ theo dõi và là mô hình đối chiếu phù hợp. Trong cấu hình này, tổng số tham số của VGG16 lớn hơn đáng kể DenseNet121.
- **Transfer learning:** phù hợp với bài toán này vì có thể bắt đầu từ đặc trưng đã học trên ImageNet thay vì huấn luyện toàn bộ mạng từ đầu.
- Số tham số không tự cho biết thời gian chạy hoặc mức sử dụng bộ nhớ. Dự án chưa có phép đo đối chứng tài nguyên giữa hai mô hình nên không kết luận mô hình nào chạy nhanh hơn.

## 5. Giới hạn khi diễn giải

Kết quả chỉ đại diện cho split và lần chạy hiện tại. Thí nghiệm dùng một seed, batch size cố định 16 và tối đa 5 epoch; chưa có nhiều seed hoặc k-fold để đánh giá độ ổn định. Các cấu hình không được chọn theo validation không có kết quả test trong CSV, nên bảng trên chỉ so sánh hai cấu hình đã chọn, không đại diện cho mọi learning rate hay mọi thiết lập có thể có.
