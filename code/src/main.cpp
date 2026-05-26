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
// CÁCH CHẠY CHƯƠNG TRÌNH
//
// Cú pháp tổng quát:
//
//   ./MCGP --instance <path_to_instance>      \
//          --termination_value <seconds>      \
//          --iter_value <max_iterations>      \
//          [--seed <integer>]                 \
//          [--logs <interval>]
//
//
// ───────────────────────────────────────────────────────────────────────────
// GIẢI THÍCH THAM SỐ
// ───────────────────────────────────────────────────────────────────────────
//
// 1. --instance <path>
//
//    Đường dẫn tới file dữ liệu bài toán (instance file).
//
//    Ví dụ:
//      --instance data/instance01.txt
//
//    Chương trình sẽ đọc:
//      - số node
//      - số cluster
//      - trọng số
//      - ràng buộc
//      - ma trận khoảng cách
//
//    Đây là tham số BẮT BUỘC.
//
//
// 2. --termination_value <seconds>
//
//    Giới hạn thời gian chạy tối đa (đơn vị: giây).
//
//    Ví dụ:
//      --termination_value 300
//
//    → thuật toán sẽ dừng sau 300 giây (~5 phút)
//      ngay cả khi chưa đạt max iteration.
//
//    Dùng để:
//      - giới hạn thời gian benchmark
//      - đảm bảo thực nghiệm công bằng giữa các thuật toán
//
//    Đây là tham số BẮT BUỘC.
//
//
// 3. --iter_value <max_iterations>
//
//    Số iteration tối đa của thuật toán ACO.
//
//    Ví dụ:
//      --iter_value 5000
//
//    → thuật toán dừng khi đạt 5000 iteration
//      dù vẫn còn thời gian.
//
//    Đây là tham số BẮT BUỘC.
//
//
// 4. --seed <integer>
//
//    Seed cho bộ sinh số ngẫu nhiên.
//
//    Ví dụ:
//      --seed 12345
//
//    → mọi random decision sẽ được tái lập hoàn toàn:
//         - thứ tự node
//         - roulette selection
//         - tabu diversification
//         - local search randomness
//
//    → cực kỳ quan trọng khi:
//         - debug
//         - benchmark
//         - so sánh thuật toán
//         - chạy thực nghiệm khóa luận
//
//    Nếu KHÔNG dùng --seed:
//      → chương trình dùng random_device()
//      → mỗi lần chạy cho kết quả khác nhau.
//
//    Đây là tham số TÙY CHỌN.
//
//
// 5. --logs <interval>
//
//    Chu kỳ ghi log tiến trình.
//
//    Ví dụ:
//      --logs 10
//
//    → cứ mỗi 10 iteration:
//         - lưu best cost
//         - lưu số feasible ants
//         - lưu trạng thái hội tụ
//
//    Interval nhỏ:
//      + log chi tiết hơn
//      - file lớn hơn
//
//    Interval lớn:
//      + file log nhỏ hơn
//      - ít thông tin diễn biến
//
//    Đây là tham số TÙY CHỌN.
//
// ───────────────────────────────────────────────────────────────────────────
// ĐIỀU KIỆN DỪNG THUẬT TOÁN
// ───────────────────────────────────────────────────────────────────────────
//
// Thuật toán sẽ dừng khi MỘT trong hai điều kiện xảy ra trước:
//
//   (1) Đạt max iteration
//         iter >= --iter_value
//
//   HOẶC
//
//   (2) Hết thời gian chạy
//         elapsed_time >= --termination_value
//
//
//
// ───────────────────────────────────────────────────────────────────────────
// VÍ DỤ CHẠY THỰC TẾ
// ───────────────────────────────────────────────────────────────────────────
//
// Ví dụ 1:
//   Chạy cơ bản
//
//   ./MCGP --instance data/instance01.txt \
//          --termination_value 300        \
//          --iter_value 5000
//
//   → chạy tối đa:
//        - 300 giây
//        HOẶC
//        - 5000 iteration
//
//
//
// Ví dụ 2:
//   Chạy với seed cố định để tái lập kết quả
//
//   ./MCGP --instance data/instance01.txt \
//          --termination_value 300        \
//          --iter_value 5000              \
//          --seed 12345
//
//   → mọi lần chạy đều cho cùng kết quả.
//
//
//
// Ví dụ 3:
//   Chạy benchmark có log chi tiết
//
//   ./MCGP --instance data/instance01.txt \
//          --termination_value 600        \
//          --iter_value 10000             \
//          --seed 1                       \
//          --logs 5
//
//   → ghi log mỗi 5 iteration.
//
// ═══════════════════════════════════════════════════════════════════════════

// Input.h chứa:
//   - struct Instance
//   - struct Parameters
//   - biến toàn cục instance, parameters
//   - LoadInput(), LoadInstance(), LogParameters()
//   - khai báo ACO_tuned()
#include "aco/Input.h"
#include "aco/ACO.h"

// filesystem dùng để tạo thư mục portable (Linux/Windows/macOS)
#include <filesystem>

using namespace std;

// Alias ngắn cho namespace filesystem của C++17
namespace fs = std::filesystem;


// ─────────────────────────────────────────────────────────────────────────
// BIẾN TOÀN CỤC CỦA main.cpp
// ─────────────────────────────────────────────────────────────────────────

// RNG chính của toàn bộ chương trình.
//
// Tengine = mt19937_64 (Mersenne Twister 64-bit)
// Đây là bộ sinh số ngẫu nhiên được truyền trực tiếp vào ACO_tuned().
//
// Seed được kiểm soát tập trung tại đây:
//   --seed N
//       → rng.seed(N)
//       → kết quả hoàn toàn tái lập được
//
//   không dùng --seed
//       → rng.seed(random_device{}())
//       → mỗi lần chạy cho kết quả khác nhau
Tengine rng;


// ═══════════════════════════════════════════════════════════════════════════
// HÀM CHÍNH: main()
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, const char *argv[])
{
    // ─────────────────────────────────────────────────────────────────────
    // BƯỚC 1: ĐỌC THAM SỐ COMMAND LINE
    // ─────────────────────────────────────────────────────────────────────
    //
    // LoadInput():
    //   - đọc toàn bộ argv[]
    //   - điền:
    //         parameters
    //         maxIter
    //         termination_time
    //   - trả về đường dẫn instance
    //
    // seed:
    //   = 0 → chưa chỉ định seed → dùng random seed
    //   ≠ 0 → user truyền --seed → dùng seed cố định
    // ─────────────────────────────────────────────────────────────────────

    unsigned seed = 0;

    string instance_path = LoadInput(argc, argv, seed);

    // ─────────────────────────────────────────────────────────────────────
    // BƯỚC 2: ĐỌC DỮ LIỆU INSTANCE
    // ─────────────────────────────────────────────────────────────────────
    //
    // LoadInstance():
    //   - đọc file instance
    //   - điền dữ liệu vào biến toàn cục:
    //
    //       instance.nV
    //       instance.nK
    //       instance.nT
    //       instance.W
    //       instance.WL
    //       instance.WU
    //       instance.D
    //
    // Sau bước này toàn bộ dữ liệu bài toán đã sẵn sàng.
    // ─────────────────────────────────────────────────────────────────────

    LoadInstance(instance_path);


    // ─────────────────────────────────────────────────────────────────────
    // BƯỚC 3: TRÍCH TÊN INSTANCE TỪ ĐƯỜNG DẪN
    // ─────────────────────────────────────────────────────────────────────
    //
    // Ví dụ:
    //   data/instances/inst_100_5.txt
    //
    // → instance_file = "inst_100_5.txt"
    // → instance_name = "inst_100_5"
    //
    // instance_name sẽ được dùng để:
    //   - đặt tên thư mục log
    //   - đặt tên file evolution/solution/objective
    // ─────────────────────────────────────────────────────────────────────

    string instance_file =
        instance_path.substr(instance_path.find_last_of("/\\") + 1);

    string instance_name =
        instance_file.substr(0, instance_file.find_last_of('.'));

    // fallback:
    // nếu file không có extension thì dùng luôn tên file
    if (instance_name.empty())
        instance_name = instance_file;


    // ─────────────────────────────────────────────────────────────────────
    // BƯỚC 4: TẠO THƯ MỤC LOG
    // ─────────────────────────────────────────────────────────────────────
    //
    // Cấu trúc:
    //
    // results/
    //   logs/
    //     <instance_name>/
    //       evolution/
    //       solutions/
    //       objectives/
    //
    // Dùng std::filesystem::create_directories():
    //   - portable hơn system("mkdir -p ...")
    //   - chạy được trên Windows/Linux/macOS
    //   - tự động bỏ qua nếu thư mục đã tồn tại
    // ─────────────────────────────────────────────────────────────────────

    string log_root = "results/logs";

    string inst_dir = log_root + "/" + instance_name;

    fs::create_directories(inst_dir + "/evolution");
    fs::create_directories(inst_dir + "/solutions");
    fs::create_directories(inst_dir + "/objectives");

    // parameters.LOGdir sẽ được ACO.cpp dùng để tạo filename log
    parameters.LOGdir = log_root;


    // ─────────────────────────────────────────────────────────────────────
    // BƯỚC 5: KHỞI TẠO RANDOM NUMBER GENERATOR
    // ─────────────────────────────────────────────────────────────────────
    //
    // Có 2 chế độ:
    //
    // (1) Fixed seed:
    //       --seed N
    //
    //     → kết quả tái lập được hoàn toàn
    //     → cực kỳ quan trọng khi:
    //          - debug
    //          - benchmark
    //          - so sánh thuật toán
    //          - chạy thực nghiệm khóa luận
    //
    // (2) Random seed:
    //       không truyền --seed
    //
    //     → mỗi lần chạy cho trajectory khác nhau
    //     → useful khi muốn diversify kết quả
    // ─────────────────────────────────────────────────────────────────────

    if (seed != 0)
    {
        // user chỉ định seed cố định
        rng.seed(seed);
    }
    else
    {
        // seed ngẫu nhiên từ hệ điều hành
        rng.seed(random_device{}());
    }


    // ─────────────────────────────────────────────────────────────────────
    // BƯỚC 6: IN CẤU HÌNH CHƯƠNG TRÌNH
    // ─────────────────────────────────────────────────────────────────────
    //
    // LogParameters() in:
    //   - số iteration
    //   - time limit
    //   - seed
    //   - ...
    //
    // giúp verify cấu hình trước khi chạy thuật toán.
    // ─────────────────────────────────────────────────────────────────────

    LogParameters();

    // ─────────────────────────────────────────────────────────────────────
    // BƯỚC 7: CHẠY THUẬT TOÁN ACO
    // ─────────────────────────────────────────────────────────────────────
    //
    // ACO_tuned():
    //   - xây nghiệm bằng Ant Colony Optimization
    //   - cải thiện bằng Local Search + Tabu Search
    //   - cập nhật pheromone
    //   - ghi log evolution/objective/solution
    //
    // rng được truyền theo reference:
    //   Tengine &rng
    //
    // → toàn bộ random behavior của thuật toán
    //   đều được kiểm soát bởi main.cpp
    //
    // Điều này rất quan trọng cho reproducibility.
    // ─────────────────────────────────────────────────────────────────────

    ACOSolution best =
        ACO_tuned(instance,
                  rng,
                  maxIter,
                  termination_time,
                  instance_name);


    // ─────────────────────────────────────────────────────────────────────
    // THÔNG TIN NGHIỆM TỐT NHẤT
    // ─────────────────────────────────────────────────────────────────────
    //
    // best.assign
    //     = cluster của từng node
    //
    // best.cost
    //     = giá trị hàm mục tiêu
    //
    // best.feasible
    //     = có thỏa toàn bộ ràng buộc hay không
    //
    // File log đã được SaveLogs() ghi tự động bên trong ACO_tuned().
    // ─────────────────────────────────────────────────────────────────────

    return 0;
}
