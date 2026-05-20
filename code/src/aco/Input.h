// ═══════════════════════════════════════════════════════════════════════════
// FILE: input.h
//
// Xử lý tất cả các tác vụ liên quan đến đầu vào của chương trình:
//   - Phân tích tham số dòng lệnh     (LoadInput)
//   - Đọc file dữ liệu bài toán       (LoadInstance)
//   - In cấu hình chạy ra màn hình    (LogParameters)
//
// Cả ba hàm đều hoạt động trên các biến toàn cục khai báo trong main.cpp:
//   instance, parameters, maxIter, termination_time, instance_path
//
// Chỉ include file này từ main.cpp (sau khi đã định nghĩa các biến trên).
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include "aco/ACO.h"

// ── Các biến toàn cục định nghĩa trong main.cpp, dùng ở đây ──────────────
extern Instance         instance;
extern Parameters       parameters;
extern int              maxIter;
extern double           termination_time;
extern string           instance_path;


// ═══════════════════════════════════════════════════════════════════════════
// LoadInput()
//
// Phân tích tham số dòng lệnh theo dạng --key value.
// Gán giá trị vào các biến toàn cục `parameters`, `maxIter` và
// `termination_time`.
//
// Tham số:
//   argc, argv : tham số chuẩn của hàm main
//   seed       : [out] seed cho bộ sinh số ngẫu nhiên (0 = không xác định)
//
// Trả về:
//   Đường dẫn tới file dữ liệu bài toán (từ --instance).
//   Thoát chương trình với thông báo lỗi nếu thiếu --instance hoặc gặp
//   tham số không hợp lệ.
// ═══════════════════════════════════════════════════════════════════════════
string LoadInput(int argc, const char *argv[], unsigned &seed);


// ═══════════════════════════════════════════════════════════════════════════
// LoadInstance()
//
// Mở và phân tích file dữ liệu, ghi kết quả vào biến toàn cục `instance`.
//
// Các định dạng được hỗ trợ:
//
//   "p" (planar)     Ma trận khoảng cách cho trực tiếp trong file.
//                    Dòng 1 : "p"
//                    Dòng 2 : nV  nK  nT
//                    Mỗi nút: <nhãn>  W[i][0..nT-1]  D[i][0..nV-1]
//                    Cuối   : WL rồi WU cho từng cụm k
//
//   "t" (tsp-like)   Các nút được cho bởi tọa độ (x, y); khoảng cách tính
//                    theo công thức D[i][j] = round(Euclid(i,j) / 2 * 1e6) / 1e6.
//                    Cấu trúc giống "p" nhưng thay hàng D bằng x y.
//
//   "h" (handover)   Biến thể mạng không dây. Khoảng cách là trọng số cạnh
//                    đảo dấu, giúp tối thiểu hoá chi phí tương đương tối đa
//                    hoá lưu lượng handover giữa các nút cùng cụm.
//                    Dòng 1  : "h"
//                    Dòng 2  : nV  nK  handover_threshold
//                    Các dòng: "Cluster <k> <WU>"
//                    Các dòng: "Node <i> <W>"
//                    Các dòng: "edge <src> <dst> <weight>"
// ═══════════════════════════════════════════════════════════════════════════
void LoadInstance(const string &pathInstance);


// ═══════════════════════════════════════════════════════════════════════════
// LogParameters()
//
// In ra màn hình (stderr) tóm tắt cấu hình chạy trước khi thuật toán bắt
// đầu. Dùng stderr để không trộn lẫn với các giá trị chi phí in ra stdout
// bởi ACO_tuned().
// ═══════════════════════════════════════════════════════════════════════════
void LogParameters();
