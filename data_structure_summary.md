# Cấu Trúc Dữ Liệu KhoaStock

## 1. Tổng Quan
Dữ liệu được tổ chức thành các thư mục chính:
- `daily/`: Dữ liệu giá theo ngày
- `intraday/`: Dữ liệu giao dịch trong ngày
- `market_data/`: Dữ liệu thị trường
- `news/`: Dữ liệu tin tức
- `company/`: Thông tin công ty
- `fundamental/`: Dữ liệu cơ bản
- `technical/`: Chỉ báo kỹ thuật
- `trading_stats/`: Thống kê giao dịch

## 2. Dữ Liệu Hàng Ngày (daily/)

### 2.1. Cấu Trúc File
- Định dạng: CSV
- Số lượng dòng: 249 dòng/file
- Khoảng thời gian: 2024-06-03 đến 2025-03-18
- Các mã: VNINDEX, VN30F1M, FPT, VNM, VCB

### 2.2. Cấu Trúc Cột
| Cột | Kiểu dữ liệu | Mô tả | Phạm vi giá trị |
|-----|--------------|--------|-----------------|
| time | datetime (YYYY-MM-DD) | Ngày giao dịch | 2024-06-03 → 2025-03-18 |
| open | float64 | Giá mở cửa | Tùy mã |
| high | float64 | Giá cao nhất | Tùy mã |
| low | float64 | Giá thấp nhất | Tùy mã |
| close | float64 | Giá đóng cửa | Tùy mã |
| volume | int64 | Khối lượng giao dịch | Tùy mã |

### 2.3. Phạm Vi Dữ Liệu Theo Mã

#### VNINDEX
- Giá: 1,094.30 → 1,341.87
- Volume: 336,332,868 → 1,977,592,840
- Trung bình giá đóng cửa: 1,267.86
- Độ lệch chuẩn giá: 35.18

#### VN30F1M
- Giá: 1,115.50 → 1,427.90
- Volume: 48,891 → 580,281
- Trung bình giá đóng cửa: 1,326.47
- Độ lệch chuẩn giá: 39.31

#### FPT
- Giá: 97.80 → 154.10
- Volume: 1,005,400 → 21,574,500
- Trung bình giá đóng cửa: 132.29
- Độ lệch chuẩn giá: 11.19

#### VNM
- Giá: 49.64 → 70.38
- Volume: 890,720 → 21,167,413
- Trung bình giá đóng cửa: 61.48
- Độ lệch chuẩn giá: 4.08

#### VCB
- Giá: 52.00 → 67.80
- Volume: 174,400 → 11,887,700
- Trung bình giá đóng cửa: 60.57
- Độ lệch chuẩn giá: 2.38

## 3. Dữ Liệu Trong Ngày (intraday/)

### 3.1. Cấu Trúc File
- Định dạng: CSV
- Tên file: {MÃ}_intraday_{NGÀY}.csv
- Số lượng giao dịch/ngày:
  - FPT: 788 giao dịch
  - VNM: 708 giao dịch
  - VCB: 561 giao dịch

### 3.2. Cấu Trúc Cột
| Cột | Kiểu dữ liệu | Mô tả | Phạm vi giá trị |
|-----|--------------|--------|-----------------|
| time | datetime with timezone | Thời điểm giao dịch | YYYY-MM-DD HH:MM:SS+07:00 |
| price | float64 | Giá khớp | Tùy mã |
| volume | int64 | Khối lượng khớp | 100 → 27,100 |
| match_type | string | Loại khớp | "Buy" hoặc "Sell" |
| id | int64 | ID giao dịch | Số nguyên duy nhất |

### 3.3. Phạm Vi Dữ Liệu Theo Mã (Mẫu ngày 2025-06-03)

#### FPT
- Giá: 116.30 → 117.00
- Volume trung bình: 414.85
- Số lệnh lớn nhất: 10,000 cp
- Tổng số giao dịch: 788

#### VNM
- Giá: 54.70 → 55.40
- Volume trung bình: 1,070.90
- Số lệnh lớn nhất: 27,100 cp
- Tổng số giao dịch: 708

#### VCB
- Giá: 56.40 → 56.70
- Volume trung bình: 676.29
- Số lệnh lớn nhất: 10,300 cp
- Tổng số giao dịch: 561

## 4. Đặc Điểm Dữ Liệu
1. Tính đầy đủ:
   - Không có giá trị null trong dữ liệu
   - Dữ liệu liên tục theo ngày và theo giao dịch
   
2. Định dạng thời gian:
   - Daily: Ngày không có timezone
   - Intraday: Datetime có timezone +07:00
   
3. Độ chính xác số:
   - Giá daily: 2 số thập phân
   - Giá intraday: 1 số thập phân
   - Volume: Số nguyên
   
4. Tính nhất quán:
   - Cấu trúc đồng nhất giữa các file cùng loại
   - ID giao dịch là duy nhất và liên tục
   - Thời gian được sắp xếp tăng dần 