// ═══════════════════════════════════════════════════════════════════════════
// FILE: main.cpp
//
// Điểm vào (entry point) của chương trình ACO cho bài toán MCGP.
//
// NHIỆM VỤ CHÍNH:
//   1. Đọc tham số từ command line (LoadInput)
//   2. Đọc dữ liệu bài toán từ file instance (LoadInstance)
//   3. Tạo thư mục lưu log kết quả
//   4. Khởi tạo bộ sinh số ngẫu nhiên
//   5. Chạy thuật toán ACO (ACO_tuned)
//
// CÁCH CHẠY CHƯƠNG TRÌNH:
//   ./MCGP --instance <path_to_instance>  \
//          --termination_value <seconds>  \
//          --iter_value <max_iterations>  \
//          [--seed <integer>]             \
//          [--logs <interval>]            \
//          [--debug 1]
//
// VÍ DỤ:
//   ./MCGP --instance data/instance01.txt --termination_value 300 --iter_value 5000
// ═══════════════════════════════════════════════════════════════════════════

#include "aco/ACO.h"    // Header chính: định nghĩa Instance, Parameters, ACOSolution,
                        // các hàm ACO_tuned(), is_feasible(), compute_cost(),...
                        // và các type alias vecDbl, matDbl, Tengine
#include <chrono>       // std::chrono::steady_clock — đồng hồ đo thời gian chính xác
using Clock = std::chrono::steady_clock;    // alias ngắn gọn cho đồng hồ
using namespace std;

// ─────────────────────────────────────────────────────────────────────────
// BIẾN TOÀN CỤC
//
// Được dùng chung giữa main(), LoadInput(), LoadInstance(), LogParameters().
// Khai báo ở cấp file (file scope) để tất cả hàm trong main.cpp cùng truy cập.
// ─────────────────────────────────────────────────────────────────────────

bool parallel_enabled = true;   // Cờ bật/tắt xử lý song song (OpenMP).
                                // Hiện tại luôn là true; có thể dùng để
                                // tắt song song trong chế độ debug nếu cần.

vector<Tengine> vengine;        // Mảng các bộ sinh số ngẫu nhiên, một engine
                                // cho mỗi thread OpenMP.
                                // vengine[p] = engine của thread p.
                                // Nếu không có OpenMP → chỉ có vengine[0].
                                // Tengine = mt19937_64 (Mersenne Twister 64-bit).

Instance instance;              // Đối tượng lưu toàn bộ dữ liệu bài toán:
                                // số node, cluster, trọng số, khoảng cách,...
                                // Được điền bởi LoadInstance(), dùng bởi ACO_tuned().

double termination_time = 1200.0;   // Thời gian tối đa cho phép thuật toán chạy (giây).
                                    // Mặc định 1200 giây = 20 phút.
                                    // Có thể thay đổi qua --termination_value.

int maxIter = 10000;            // Số iteration tối đa của vòng lặp ACO.
                                // Thuật toán dừng khi đạt maxIter HOẶC hết
                                // termination_time, tuỳ điều kiện nào đến trước.
                                // Có thể thay đổi qua --iter_value.

string instance_path;           // Đường dẫn đầy đủ đến file instance.
                                // Ví dụ: "data/instances/inst_100_5.txt"
                                // Được dùng bởi LoadInstance() và LogParameters().

// ─────────────────────────────────────────────────────────────────────────
// KHAI BÁO HÀM (Forward Declarations)
//
// Khai báo trước để main() có thể gọi các hàm được định nghĩa trong
// Input.cpp (LoadInput, LoadInstance, LogParameters).
// ─────────────────────────────────────────────────────────────────────────

string LoadInput(int argc, const char *argv[], unsigned &seed);
    // Đọc và phân tích tham số từ command line.
    // Trả về đường dẫn file instance.

void LoadInstance(const string &pathInstance);
    // Mở và đọc file instance, điền vào biến toàn cục `instance`.

void LogParameters();
    // In tóm tắt cấu hình chạy ra stdout (→ .log) trước khi bắt đầu thuật toán.
    // Định nghĩa nằm trong Input.cpp, dùng cout để output ra stdout.


// ═══════════════════════════════════════════════════════════════════════════
// HÀM CHÍNH: main()
//
// Điều phối toàn bộ quy trình: đọc input → chuẩn bị → chạy ACO → kết thúc.
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, const char *argv[])
{
    // ── Bước 1: Đọc tham số command line ──────────────────────────────────
    unsigned seed = 0;                              // seed = 0 → dùng random_device (không cố định)
    instance_path = LoadInput(argc, argv, seed);    // phân tích argv, điền parameters, trả về path

    // ── Bước 2: Đọc dữ liệu bài toán từ file ────────────────────────────
    LoadInstance(instance_path);    // đọc file, điền instance.nV, nK, nT, W, WL, WU, D

    // ── Bước 3: Trích tên instance từ đường dẫn ──────────────────────────
    // Ví dụ: "data/instances/inst_100_5.txt" → "inst_100_5"
    string instance_file = instance_path.substr(instance_path.find_last_of("/\\") + 1);
                                    // find_last_of("/\\") tìm '/' hoặc '\' cuối cùng
                                    // +1 để bỏ qua ký tự '/' đó
                                    // → lấy phần tên file: "inst_100_5.txt"
    string instance_name = instance_file.substr(0, instance_file.find_last_of('.'));
                                    // bỏ phần mở rộng (.txt, .dat,...) → "inst_100_5"
    if (instance_name.empty())
        instance_name = instance_file;  // fallback: nếu không có dấu chấm thì giữ nguyên

    // ── Bước 4: Tạo thư mục log ──────────────────────────────────────────
    // Cấu trúc thư mục:
    //   results/logs/<instance_name>/evolution/   ← file diễn biến hội tụ
    //   results/logs/<instance_name>/solutions/   ← file nghiệm tốt nhất
    //   results/logs/<instance_name>/objectives/  ← file cost tốt nhất
    string aco_logdir = "results/logs";
    string instDir    = aco_logdir + "/" + instance_name;
    system(("mkdir -p " + instDir + "/evolution "       // tạo thư mục evolution
                        + instDir + "/solutions "       // tạo thư mục solutions
                        + instDir + "/objectives").c_str());    // tạo thư mục objectives
    parameters.LOGdir = aco_logdir;     // lưu đường dẫn gốc vào parameters để ACO.cpp dùng

    // ── Bước 5: Khởi tạo bộ sinh số ngẫu nhiên ───────────────────────────
    // Tạo một engine cho mỗi thread OpenMP (hoặc 1 engine nếu không có OpenMP).
    // Mỗi thread có engine riêng để tránh data race khi sinh số ngẫu nhiên song song.
    random_device rd;                                   // nguồn entropy phần cứng (nếu có)
    for (unsigned p = 0; p < (unsigned)omp_get_max_threads(); p++)
    {
        vengine.emplace_back(rd());     // tạo engine p với seed ngẫu nhiên từ phần cứng
        if (seed != 0)
            vengine[p].seed((p + 1) * seed);    // nếu seed cố định → mỗi thread dùng bội số
                                                // để các thread khác nhau nhưng kết quả tái lập
        else
            vengine[p].seed(random_device{}()); // nếu seed=0 → seed ngẫu nhiên hoàn toàn
                                                // (mỗi lần chạy cho kết quả khác nhau)
    }

    // ── Bước 6: In cấu hình ra stdout → .log ────────────────────────────
    LogParameters();    // định nghĩa trong Input.cpp, dùng cout → ra .log

    // ── Bước 7: Chạy thuật toán ACO ──────────────────────────────────────
    ACOSolution best = ACO_tuned(instance,          // dữ liệu bài toán
                                 maxIter,            // số iteration tối đa
                                 termination_time,   // giới hạn thời gian (giây)
                                 instance_name);     // tên instance (cho file log)
    // Sau khi ACO_tuned() kết thúc:
    //   - best.assign   chứa nghiệm tốt nhất tìm được
    //   - best.cost     chứa giá trị hàm mục tiêu
    //   - best.feasible cho biết nghiệm có hợp lệ không
    //   - Log đã được ghi ra file tự động bởi SaveLogs() bên trong ACO_tuned()

    return 0;   // kết thúc chương trình thành công
}
