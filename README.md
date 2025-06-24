# Biểu đồ Phân tích Chứng Khoán

Dự án này tạo ra ba biểu đồ phân tích chứng khoán:
1. Biểu đồ đường về lợi nhuận tích lũy
2. Biểu đồ cột về tỷ lệ thắng và tỷ lệ Sharpe
3. Heatmap về tương quan giữa các chỉ báo kỹ thuật

## Cài đặt

1. Cài đặt Python 3.8 trở lên
2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Sử dụng

Chạy script Python để tạo các biểu đồ:
```bash
python stock_visualizations.py
```

Script sẽ tạo ra ba file PNG:
- `cumulative_returns.png`: Biểu đồ đường về lợi nhuận tích lũy
- `win_rate_sharpe.png`: Biểu đồ cột về tỷ lệ thắng và Sharpe
- `correlation_heatmap.png`: Heatmap về tương quan

## Lưu ý

- Dữ liệu hiện tại là dữ liệu mẫu. Để sử dụng dữ liệu thực tế, hãy thay thế hàm `generate_sample_data()` bằng dữ liệu thực của bạn.
- Các biểu đồ được xuất ra dưới dạng file PNG với độ phân giải cao (scale=2). 