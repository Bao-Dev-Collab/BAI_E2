# Hướng dẫn kiểm tra yêu cầu Bài thực hành 2

Tài liệu này chuyển các yêu cầu trong đề **“Practice 2 - Hands-on practice with pre-trained neural network architectures”** thành checklist kiểm tra project. Dùng để rà soát trước khi nộp và chuẩn bị trình bày.

## 1. Thử các mô hình pretrained khác nhau

Đề yêu cầu thử các kiến trúc pretrained, ví dụ VGG và DenseNet.

- [ ] Có ít nhất hai mô hình pretrained trong mã nguồn.
- [ ] Mỗi mô hình tải được trọng số pretrained và chạy forward thành công.
- [ ] Có kết quả huấn luyện/đánh giá cho từng mô hình.
- [ ] Báo cáo so sánh kết quả và nêu ưu, nhược điểm; không chỉ liệt kê tên mô hình.

**Cách kiểm tra trong project này**

- Mở `models.py`, kiểm tra hàm `build_model` hỗ trợ `vgg16` và `densenet121`.
- Mở `outputs/reduced-experiments/results.csv`, kiểm tra có kết quả cho cả hai mô hình.
- Mở mục “So sánh VGG16 và DenseNet121” trong `bao_cao_thuyet_trinh_thuc_hanh_2.md`.

## 2. Điều chỉnh siêu tham số

Đề nêu các ví dụ như learning rate và batch size. Cần thử ít nhất một siêu tham số ở nhiều giá trị và ghi lại cấu hình cùng kết quả; đề không bắt buộc phải thử mọi giá trị có thể.

- [ ] Ghi rõ các giá trị siêu tham số đã thử.
- [ ] So sánh kết quả trên cùng bộ dữ liệu và cùng cách đánh giá.
- [ ] Dùng validation để chọn cấu hình; không chọn cấu hình dựa trên test.
- [ ] Nêu rõ những giá trị chưa thử để tránh tạo ấn tượng rằng đã khảo sát đầy đủ.

**Cấu hình đã chạy trong project này**

- Learning rate: `0.01` và `0.001`.
- Batch size: `16`.
- Mô hình: VGG16 và DenseNet121.

Như vậy project đã có so sánh learning rate. Batch size 32 chưa được chạy; đây là một thử nghiệm mở rộng nếu muốn khảo sát thêm, không phải một giá trị bắt buộc được nêu cụ thể trong đề.

**Cách kiểm tra**

- Đối chiếu các cột `model`, `learning_rate`, `batch_size`, `best_validation_loss` và `validation_accuracy_at_best_loss` trong `outputs/reduced-experiments/results.csv`.
- Xác nhận chỉ cấu hình được chọn bằng validation mới có `test_loss` và `test_accuracy`.

## 3. Theo dõi quá trình huấn luyện bằng TensorBoard

- [ ] Có ghi loss/accuracy theo epoch.
- [ ] Có thể mở TensorBoard và xem log của các lần chạy.
- [ ] Khi trình bày biểu đồ, ghi rõ biểu đồ thuộc lần chạy nào.

**Mở TensorBoard trong PowerShell tại thư mục project**

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir .\outputs\reduced-experiments
```

Sau đó mở địa chỉ TensorBoard được in trong terminal, thường là `http://localhost:6006`. Dừng máy chủ bằng `Ctrl+C`.

Trong TensorBoard, kiểm tra các tag như `Loss/train`, `Loss/train_eval`, `Loss/validation`, `Accuracy/train`, `Accuracy/train_eval` và `Accuracy/validation`. Các tag test chỉ xuất hiện ở cấu hình được chọn.

## 4. Khảo sát kiến trúc mô hình

- [ ] Có in hoặc xem cấu trúc mô hình.
- [ ] Xác định classifier/lớp cuối được thay để khớp số lớp của dữ liệu.
- [ ] Ghi nhận cách đóng băng hoặc fine-tune các lớp pretrained.

**Cách kiểm tra trong project này**

- Trong `models.py`, kiểm tra classifier của VGG16 và DenseNet121 được thay bằng đầu ra 37 lớp.
- Trong `main.py`, kiểm tra mô hình và số tham số được in khi bắt đầu mỗi cấu hình.
- Xem `bao_cao_kiem_tra_du_lieu_va_mo_hinh.md` để đối chiếu số tham số tổng và tham số được huấn luyện.

## 5. Chuẩn bị dữ liệu

- [ ] Dữ liệu được nạp và tiền xử lý trước khi huấn luyện.
- [ ] Có batch dữ liệu qua `DataLoader`.
- [ ] Số lớp và thứ tự lớp nhất quán giữa train, validation và test.
- [ ] Test được giữ riêng, không dùng để cập nhật trọng số hoặc chọn siêu tham số.

**Cách kiểm tra trong project này**

- Trong `data_loader.py`, kiểm tra các transform và DataLoader cho train, train-eval, validation và test.
- Dataset hiện dùng Oxford-IIIT Pet với 37 lớp.
- Cấu trúc dữ liệu mong đợi:

```text
dataset/
  train/<class-name>/
  val/<class-name>/
  test/<class-name>/
```

## 6. Huấn luyện mô hình

- [ ] Có hàm loss và optimizer.
- [ ] Có vòng lặp huấn luyện trên dữ liệu train.
- [ ] Có đánh giá validation và lưu checkpoint phù hợp.
- [ ] Có lưu cấu hình và kết quả để có thể kiểm tra lại.

**Cách kiểm tra trong project này**

- `main.py` dùng `CrossEntropyLoss` và Adam.
- Validation loss được dùng để chọn checkpoint; early stopping giới hạn thời gian huấn luyện.
- Checkpoint và log được lưu dưới thư mục output của lần chạy.
- `results.csv` ghi cấu hình cùng các metric chính.

## 7. Đánh giá trên tập test

- [ ] Có đánh giá mô hình trên test sau quá trình huấn luyện/chọn cấu hình.
- [ ] Báo cáo test loss và/hoặc test accuracy.
- [ ] Không dùng test để chọn learning rate, batch size, epoch hay mô hình tốt nhất.

**Kết quả hiện có**

- VGG16, cấu hình được chọn bằng validation: test accuracy `88.80%`.
- DenseNet121, cấu hình được chọn bằng validation: test accuracy `91.31%`.
- Các số liệu này là kết quả của một lần chạy với split/seed hiện tại; không nên diễn giải như trung bình nhiều lần chạy.

## Checklist trước khi nộp

- [ ] Báo cáo mô tả đúng dữ liệu, hai mô hình và cách thay classifier.
- [ ] Có bảng kết quả khớp với `outputs/reduced-experiments/results.csv`.
- [ ] Có so sánh hai mô hình và giải thích ưu/nhược điểm.
- [ ] Có nêu learning rate đã thử; ghi rõ batch size cố định ở 16.
- [ ] Có biểu đồ TensorBoard từ các log mới, hoặc ghi chú rõ nếu dùng hình demo cũ.
- [ ] Có test metrics cho cấu hình được chọn và giải thích cách chọn bằng validation.
- [ ] Không trình bày batch size 32, k-fold hoặc nhiều seed như đã thực hiện nếu chưa chạy.

## Kết luận theo đề

Theo các mục thể hiện trong ảnh đề, project đã triển khai hai mô hình pretrained, điều chỉnh learning rate, theo dõi bằng TensorBoard, khảo sát cấu trúc và classifier, chuẩn bị dữ liệu, huấn luyện và đánh giá trên test. Đề đưa batch size như ví dụ về siêu tham số; project đã thay đổi learning rate nhưng chưa thay đổi batch size. Nếu giảng viên yêu cầu cụ thể phải so sánh nhiều batch size, cần bổ sung thí nghiệm tương ứng.
