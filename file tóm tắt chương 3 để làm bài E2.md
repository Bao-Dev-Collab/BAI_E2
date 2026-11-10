# Chương 3 - HUẤN LUYỆN MẠNG THẦN KINH SÂU (Training Deep Neural Networks)

**Thông tin chung**
* **Cơ quan**: Trường Đại học Giao thông Vận tải TP.HCM, Viện Công nghệ Thông tin, Điện, Điện tử.
* **Giảng viên**: Tiến sĩ Nguyễn Thị Khánh Tiên.

## 1. Khái niệm cơ bản về Mạng thần kinh sâu (DNN)
* **Định nghĩa**: Mạng thần kinh sâu là các mô hình máy tính gồm nhiều lớp nơ-ron nhân tạo nhằm mô phỏng cách não bộ xử lý thông tin. Từ "sâu" chỉ số lượng các lớp ẩn; mạng càng nhiều lớp ẩn thì càng học được nhiều mẫu dữ liệu phức tạp.
* **Quá trình huấn luyện**: Là quá trình dạy mô hình thực hiện một tác vụ cụ thể bằng cách sử dụng một lượng lớn dữ liệu được gán nhãn.
* **Thuật toán**: 
  * **Lan truyền ngược (Backpropagation)**: Điều chỉnh trọng số để giảm thiểu sai số giữa đầu ra dự đoán và thực tế.
  * **Tối ưu hóa**: Các thuật toán như Adam, SGD giúp tìm trọng số tối ưu qua nhiều vòng lặp (epoch).
* **Thách thức**: Quá khớp (overfitting), gradient biến mất/bùng nổ, và yêu cầu tài nguyên phần cứng lớn.

## 2. Quy trình 6 bước huấn luyện DNN với PyTorch
1. **Chuẩn bị dữ liệu**: Sử dụng lớp `Dataset` để lưu trữ dữ liệu/nhãn và `DataLoader` để chia lô (batch), xáo trộn và nạp dữ liệu song song.
2. **Xác định mô hình**: Tạo kiến trúc mạng bằng cách kế thừa `torch.nn.Module`, định nghĩa các lớp (Linear, Conv2d, ReLU...) trong `__init__` và luồng truyền dữ liệu qua hàm `forward`.
3. **Chọn Hàm mất mát và Trình tối ưu hóa**: 
   * *Hàm mất mát* (Loss Function): `CrossEntropyLoss` cho phân loại, `MSELoss` cho hồi quy.
   * *Trình tối ưu hóa* (Optimizer): SGD, Adam, RMSprop.
4. **Vòng lặp huấn luyện**: Đưa dữ liệu vào GPU, xóa gradient (`optimizer.zero_grad()`), truyền xuôi, tính độ mất mát, truyền ngược (`loss.backward()`) và cập nhật trọng số (`optimizer.step()`).
5. **Xác thực (Validation)**: Dùng `model.eval()` và `torch.no_grad()` để đánh giá mô hình trên tập dữ liệu riêng biệt, giúp phát hiện quá khớp và tinh chỉnh siêu tham số.
6. **Lưu và Tải mô hình**: Dùng `torch.save(model.state_dict(), 'model.pth')` để lưu các tham số và `load_state_dict` để nạp lại khi tái sử dụng.

## 3. Các Kỹ thuật Nâng cao và Tối ưu hóa
* **TensorBoard**: Công cụ giúp trực quan hóa thời gian thực (hiển thị loss, accuracy, biểu đồ kiến trúc, hình ảnh...) thông qua `SummaryWriter`.
* **Xử lý Gradient biến mất/bùng nổ**: Sử dụng hàm kích hoạt ReLU, cắt xén gradient (gradient clipping), hoặc Chuẩn hóa lô (Batch Normalization).
* **Khởi tạo trọng số**: Áp dụng phân phối ngẫu nhiên, Xavier/Glorot hoặc He để mô hình huấn luyện ổn định hơn.
* **Hàm kích hoạt & Chuẩn hóa lô**: 
  * *Hàm kích hoạt*: ReLU, Sigmoid, Tanh, LeakyReLU tạo tính phi tuyến.
  * *Chuẩn hóa lô (BatchNorm)*: Chuẩn hóa đầu ra của các lớp để tăng tốc huấn luyện.
* **Học chuyển giao (Transfer Learning) & Tinh chỉnh (Fine-tuning)**:
  * Tận dụng mô hình đã huấn luyện sẵn (như ResNet, VGG).
  * Đóng băng (freeze) các lớp cũ và thay đổi lớp phân loại cuối cùng.
  * Mở khóa (unfreeze) một số lớp để tinh chỉnh trọng số cho phù hợp với tập dữ liệu nhỏ.
* **Chống quá khớp (Regularization)**: 
  * *Dropout*: Vô hiệu hóa ngẫu nhiên nơ-ron trong quá trình huấn luyện.
  * *L1/L2*: Phạt trọng số lớn bằng cách cộng thêm giá trị vào hàm mất mát (L2 cấu hình qua `weight_decay`).

## 4. Thực hành
* Sử dụng thư viện `torchvision.models` để nạp các mô hình kiến trúc nổi tiếng (như ResNet-18).
* Điều chỉnh lớp phân loại cuối cùng để khớp số lượng nhãn của tập dữ liệu mới (ví dụ: tập hình ảnh CIFAR-10 có 10 lớp).
* Viết vòng lặp huấn luyện, đánh giá mô hình và theo dõi qua TensorBoard.