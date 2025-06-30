# Dự Án Dự Báo & Phân Tích Chứng Khoán Việt Nam

**Tác giả: Vũ Đăng Khoa - 22010357**

## 1. Ý Tưởng

Dự án hướng tới xây dựng một hệ thống hoàn chỉnh cho việc thu thập, xử lý, phân tích và dự báo dữ liệu chứng khoán Việt Nam. Hệ thống này giúp nhà đầu tư và nhà nghiên cứu có thể:
- Tự động thu thập dữ liệu thị trường, giao dịch, cơ bản, tin tức.
- Phân tích dữ liệu đa chiều: kỹ thuật, cơ bản, sentiment, thống kê.
- Xây dựng các mô hình dự báo ngắn hạn (1-3 ngày) tối ưu, có thể triển khai thực tế.
- Đánh giá, trực quan hóa và xuất báo cáo kết quả mô hình.

## 2. Thu Thập Dữ Liệu

- **Công cụ chính:** `data/stock_data_collector.py` sử dụng API từ `vnstock` để lấy dữ liệu tự động.
- **Các loại dữ liệu:**
  - Dữ liệu giá, khối lượng (daily, intraday)
  - Bảng giá tổng hợp (market data)
  - Thống kê giao dịch (trading stats)
  - Dữ liệu cơ bản doanh nghiệp (fundamental)
  - Tin tức, sự kiện liên quan đến cổ phiếu
- **Thư mục lưu trữ:**
  - `data/collected_data/` chứa các file CSV cho từng loại dữ liệu, phân chia theo mã cổ phiếu và ngày.

## 3. Xử Lý Dữ Liệu

- **Tiền xử lý:**
  - Làm sạch dữ liệu, chuẩn hóa định dạng, loại bỏ giá trị bất thường/null.
  - Tính toán các chỉ báo kỹ thuật (SMA, Bollinger Bands, RSI, MACD, ATR, OBV, v.v.)
  - Tổng hợp các đặc trưng (features) cho mô hình dự báo: momentum, volatility, micro-patterns, lag features, time features.
- **Công cụ:**
  - `data/analysis/data_type_analyzer.py`: Phân tích kiểu dữ liệu, kiểm tra chất lượng dữ liệu.
  - `data/analysis/result_visualizer.py`: Trực quan hóa dữ liệu và kết quả mô hình.
  - `data/analysis/enhanced_predictor.py`: Tích hợp các đặc trưng kỹ thuật, cơ bản, sentiment cho mô hình nâng cao.

## 4. Xây Dựng Mô Hình

- **Mô hình chính:**
  - `optimized_short_term_model.py`: Mô hình ensemble tuyến tính tối ưu cho dự báo ngắn hạn (1-3 ngày), sử dụng các đặc trưng kỹ thuật, micro-patterns, momentum, v.v.
  - Hỗ trợ huấn luyện, đánh giá, lưu mô hình (dạng JSON), xuất kết quả chi tiết và dữ liệu cho LaTeX.
- **Các file mô hình:**
  - Lưu tại `saved_models/` dưới dạng JSON (weights, feature_names, feature_stats cho từng mã và từng horizon).
  - Có thể nạp lại để dự báo trên máy khác mà không cần huấn luyện lại.
- **Các mô hình khác:**
  - `lightweight_prediction_model.py`, `ultra_light_prediction_model.py`, `pure_python_prediction_model.py`: Các phiên bản mô hình đơn giản hơn, dùng cho so sánh.

## 5. Kết Quả Mô Hình & Trực Quan Hóa

- **Kết quả chi tiết:**
  - Lưu tại `latex_figures/comprehensive_results.json` (metrics, dự báo tiếp theo, v.v.)
  - Các file CSV cho scatter, histogram, performance, multi-horizon để vẽ biểu đồ.
- **Trực quan hóa:**
  - `plot_results.py`: Vẽ các biểu đồ tổng hợp (histogram, scatter, rolling accuracy, multi-horizon) cho từng mã.
  - `data/analysis/result_visualizer.py`: Vẽ feature importance, performance, tín hiệu giao dịch, phân tích kỹ thuật/cơ bản.
  - Các file hình ảnh lưu tại `latex_figures/`.
- **Báo cáo LaTeX:**
  - Tự động sinh mã TikZ/PGFPlots cho báo cáo khoa học.

## 6. Dashboard & Ứng Dụng

- **Dashboard:**
  - `data/stock_dashboard.py`: Ứng dụng Streamlit trực quan hóa dữ liệu, quy trình phân tích, kết quả mô hình, tín hiệu giao dịch.
  - Hỗ trợ chọn mã cổ phiếu, xem dữ liệu, biểu đồ, tiến độ pipeline.

## 7. Hướng Dẫn Sử Dụng

1. **Cài đặt thư viện:**
   - Xem file `requirements_model.txt` hoặc `requirements.txt` trong các thư mục.
   - Cài đặt: `pip install -r requirements_model.txt`
2. **Thu thập dữ liệu:**
   - Chạy: `python data/stock_data_collector.py`
3. **Huấn luyện mô hình:**
   - Chạy: `python optimized_short_term_model.py`
4. **Trực quan hóa kết quả:**
   - Chạy: `python plot_results.py`
5. **Xem dashboard:**
   - Chạy: `streamlit run data/stock_dashboard.py`

## 8. Cấu Trúc Thư Mục Chính

- `data/collected_data/`: Dữ liệu gốc thu thập
- `data/analysis/`: Script phân tích, trực quan hóa
- `saved_models/`: File mô hình đã huấn luyện
- `latex_figures/`: Kết quả, hình ảnh, file cho báo cáo
- `optimized_short_term_model.py`: Mô hình dự báo chính
- `plot_results.py`: Vẽ biểu đồ tổng hợp
- `data/stock_dashboard.py`: Dashboard Streamlit

## 9. Liên Hệ & Đóng Góp

- Tác giả: **Vũ Đăng Khoa**
- Đóng góp, phản hồi: Vui lòng tạo issue hoặc liên hệ trực tiếp.
