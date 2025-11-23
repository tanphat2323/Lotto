# Hướng Dẫn Cài Đặt & Sử Dụng (Lotto 5/35 AI)

Tài liệu này hướng dẫn chi tiết cách cài đặt môi trường, huấn luyện mô hình và chạy hệ thống dự đoán Lotto 5/35 trên hệ điều hành **Windows**.

## 1. Yêu cầu hệ thống

Trước khi bắt đầu, hãy đảm bảo máy tính của bạn đã cài đặt:

*   **Python 3.10 trở lên**: [Tải tại python.org](https://www.python.org/downloads/).
    *   *Lưu ý khi cài đặt:* Hãy tích vào ô **"Add Python to PATH"** ở màn hình cài đặt đầu tiên.
*   **Git** (Tùy chọn): Để tải mã nguồn về. Nếu không có Git, bạn có thể tải file `.zip` và giải nén.

## 2. Cài đặt môi trường

Làm theo các bước sau trong **PowerShell** hoặc **Command Prompt (CMD)**.

### Bước 1: Tải mã nguồn
Nếu bạn đã có mã nguồn trong thư mục (ví dụ `C:\Projects\LottoAI`), hãy mở terminal tại thư mục đó.

```powershell
cd C:\Projects\LottoAI
```

### Bước 2: Tạo môi trường ảo (Virtual Environment)
Môi trường ảo giúp cô lập các thư viện của dự án, tránh xung đột với hệ thống.

```powershell
python -m venv .venv
```

### Bước 3: Kích hoạt môi trường
Sau khi chạy lệnh trên, thư mục `.venv` sẽ được tạo. Hãy kích hoạt nó:

*   **Với PowerShell:**
    ```powershell
    .venv\Scripts\activate
    ```
*   **Với Command Prompt (cmd.exe):**
    ```cmd
    .venv\Scripts\activate.bat
    ```

*Dấu hiệu thành công:* Bạn sẽ thấy chữ `(.venv)` xuất hiện ở đầu dòng lệnh.

### Bước 4: Cài đặt thư viện phụ thuộc
Chạy lệnh sau để tải các thư viện cần thiết (TensorFlow, FastAPI, Pandas...):

```powershell
pip install -r requirements.txt
```

*Quá trình này có thể mất vài phút tùy tốc độ mạng.*

---

## 3. Huấn luyện mô hình AI

Trước khi dự đoán, bạn cần dạy cho AI học từ dữ liệu lịch sử (`data/dataset.csv`).

Chạy lệnh:
```powershell
python src/train.py
```

**Quá trình diễn ra:**
1.  Hệ thống đọc dữ liệu từ file CSV.
2.  Tạo các chuỗi số lịch sử (Lookback window).
3.  Huấn luyện mô hình LSTM qua 100 chu kỳ (epochs).
4.  Lưu mô hình đã học vào thư mục `models/lotto_model.keras`.
5.  Kết quả đánh giá (độ chính xác) được lưu tại `models/latest_metrics.json`.

---

## 4. Khởi chạy hệ thống dự đoán (API)

Sau khi huấn luyện xong, hãy bật Web Server để bắt đầu dự đoán.

Chạy lệnh:
```powershell
uvicorn src.main:app --reload
```

Nếu thành công, bạn sẽ thấy thông báo:
`INFO:     Uvicorn running on http://127.0.0.1:8000`

---

## 5. Sử dụng

### Cách 1: Dự đoán kỳ quay tiếp theo
Mở trình duyệt web (Chrome, Edge...) và truy cập đường dẫn:

[http://127.0.0.1:8000/predict/next](http://127.0.0.1:8000/predict/next)

Kết quả trả về sẽ có dạng JSON, bao gồm:
*   `predicted_main_top10`: 10 con số có xác suất về cao nhất (cho bộ 5 số chính).
*   `predicted_special_top3`: 3 con số có xác suất cao nhất (cho số đặc biệt).
*   `main_probabilities`: Bảng xác suất chi tiết của từng số (1-35).

### Cách 2: Xem thống kê mô hình
Truy cập: [http://127.0.0.1:8000/stats](http://127.0.0.1:8000/stats)
Để xem độ chính xác của mô hình trên tập kiểm tra.

---

## 6. Khắc phục lỗi thường gặp

**Lỗi: "python is not recognized..."**
*   Nguyên nhân: Bạn chưa cài Python hoặc chưa thêm vào PATH.
*   Khắc phục: Cài lại Python và nhớ tích chọn "Add Python to PATH".

**Lỗi: "SecurityError: PSSecurityException" khi chạy activate trên PowerShell**
*   Nguyên nhân: Chính sách bảo mật của Windows chặn script.
*   Khắc phục: Chạy lệnh sau trên PowerShell (Run as Administrator):
    ```powershell
    Set-ExecutionPolicy RemoteSigned
    ```
    Hoặc chuyển sang dùng Command Prompt (CMD).

**Lỗi: "ModuleNotFoundError"**
*   Nguyên nhân: Chưa cài đủ thư viện hoặc chưa kích hoạt `.venv`.
*   Khắc phục: Đảm bảo thấy `(.venv)` và chạy lại `pip install -r requirements.txt`.
