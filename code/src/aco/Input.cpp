// ═══════════════════════════════════════════════════════════════════════════
// FILE: Input.cpp
//
// Tầng INPUT: Đọc tham số dòng lệnh (Command Line) và đọc file dữ liệu (Instance).
// Tầng này KHÔNG chứa bất kỳ logic thuật toán tối ưu hóa nào.
//
// LƯU Ý THIẾT KIẾN TRÚC:
//   - Instance: Struct chứa ma trận khoảng cách (D), trọng số (W), và cận (WL, WU).
//   - Parameters: Struct chứa các config chạy thuật toán (thời gian, log, thư mục).
// ═══════════════════════════════════════════════════════════════════════════

#include "Input.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace std;

// ─────────────────────────────────────────────────────────────────────────
// BIẾN TOÀN CỤC (được khai báo extern trong Input.h, định nghĩa ở đây)
// ─────────────────────────────────────────────────────────────────────────
Instance   instance;           // Chứa toàn bộ dữ liệu bài toán đọc từ file
Parameters parameters;        // Chứa các tham số cấu hình chạy thuật toán
int        maxIter          = 10000;  // Số vòng lặp tối đa mặc định
double     termination_time = 1200.0;  // Thời gian chạy tối đa mặc định (giây)
string     instance_path;              // Đường dẫn file dữ liệu vừa đọc được


// ═══════════════════════════════════════════════════════════════════════════
// HÀM 1: LoadInput - Phân tích cú pháp dòng lệnh (Command Line Parser)
//
// Đầu vào ví dụ: ./MCGP --instance data.txt --seed 42 --termination_value 60.5
// Đầu ra: Trả về chuỗi "data.txt" và điền các giá trị vào biến toàn cục.
// ═══════════════════════════════════════════════════════════════════════════
string LoadInput(int argc, const char *argv[], unsigned &seed)
{
    string pathInstance;

    // Duyệt qua các argument. Bước nhảy i += 2 vì cú pháp là --key <value>
    // Bắt đầu từ i=1 vì argv[0] luôn là tên chương trình ("./MCGP")
    for (int i = 1; i < argc; i += 2)
    {
        bool matched = false; // Cờ đánh dấu xem argument hiện tại có hợp lệ không

        // Kiểm tra xem có tồn tại key (bắt đầu bằng '-') và đủ giá trị value hay không
        if (argv[i][0] == '-' && i + 1 < argc)
        {
            string key   = argv[i];     // VD: "--instance"
            string value = argv[i + 1]; // VD: "data.txt"

            if (key == "--instance") {
                matched = true;
                pathInstance = value; // Lưu đường dẫn file vào biến tạm thời
            }
            else if (key == "--seed") {
                matched = true;
                seed = stoi(value); // Chuyển chuỗi "42" thành số nguyên 42
            }
            else if (key == "--iter_value") {
                matched = true;
                maxIter = stoi(value); // Gán giới hạn vòng lặp (VD: 5000)
            }
            else if (key == "--termination_value") {
                matched = true;
                termination_time = stod(value);   // Gán thời gian tối đa (VD: 60 giây)
            }
            else if (key == "--logs") {
                matched = true;
                parameters.ALGlg = stoi(value);   // Tần suất ghi log (VD: mỗi 100 vòng)
            }
        }

        // Nếu người dùng gõ sai cú pháp (VD: thiếu value, hoặc key sai)
        if (!matched) {
            cerr << "\n[ERROR] Unknown or malformed argument: " << argv[i] << "\n";
            exit(1); // Dừng chương trình ngay lập tức với mã lỗi 1
        }
    }

    // --instance là tham số BẮT BUỘC. Nếu không có, không thể chạy
    if (pathInstance.empty()) {
        cerr << "\n[ERROR] Missing required argument: --instance <path>\n";
        exit(1);
    }

    return pathInstance; // Trả về đường dẫn đã chuẩn hóa
}


// ═══════════════════════════════════════════════════════════════════════════
// HÀM 2: LoadInstance - Đọc file dữ liệu và khởi tạo ma trận
//
// Hỗ trợ 3 định dạng file phổ biến trong bài toán Clustering:
//   "p" (Planar): Ma trận khoảng cách cho sẵn trực tiếp.
//   "t" (TSP-like): Cho tọa độ (x,y), tự tính khoảng cách Euclid.
//   "h" (Handover): Mạng không dây, tính khoảng cách từ cạnh đồ thị.
// ═══════════════════════════════════════════════════════════════════════════
void LoadInstance(const string &pathInstance)
{
    ifstream file(pathInstance);
    if (!file) {
        cerr << "\n[ERROR] Cannot open instance file: " << pathInstance << "\n";
        exit(1);
    }

    // Đọc ký tự loại file ở dòng đầu tiên
    file >> instance.type;

    // ─────────────────────────────────────────────────────────────────────
    // XỬ LÝ "p" (PLANAR) VÀ "t" (TỌA ĐỘ)
    // Cấu trúc file giống nhau ở phần header và trọng số, chỉ khác phần khoảng cách
    // ─────────────────────────────────────────────────────────────────────
    if (instance.type == "p" || instance.type == "t")
    {
        // Đọc header: nV (số đỉnh), nK (số cluster), nT (số chiều trọng số)
        file >> instance.nV >> instance.nK >> instance.nT;

        // Cấp phát các ma trận 2D (vector của vector) với kích thước tương ứng
        // Khởi tạo tất cả bằng 0.0
        instance.W  = matDbl(instance.nV, vecDbl(instance.nT, 0.0)); // Trọng số đỉnh (nV x nT)
        instance.D  = matDbl(instance.nV, vecDbl(instance.nV, 0.0)); // Khoảng cách (nV x nV)
        instance.WL = matDbl(instance.nK, vecDbl(instance.nT, 0.0)); // Cận dưới cluster (nK x nT)
        instance.WU = matDbl(instance.nK, vecDbl(instance.nT, 0.0)); // Cận trên cluster (nK x nT)

        // Ma trận tạm thời để chứa tọa độ (chỉ dùng cho loại "t")
        matDbl C(instance.nV, vecDbl(2, 0.0));

        // Vòng lặp đọc thông tin từng đỉnh
        for (int i = 0; i < instance.nV; ++i)
        {
            string ignore;
            file >> ignore; // Bỏ qua tên label (VD: "Node1", "V2") vì ta dùng index

            // Đọc nT trọng số cho đỉnh i
            for (int t = 0; t < instance.nT; ++t) {
                file >> instance.W[i][t];
            }

            if (instance.type == "p")
            {
                // LOẠI "p": Đọc trực tiếp ma trận khoảng cách từ file
                for (int j = 0; j < instance.nV; ++j) {
                    file >> instance.D[i][j];
                    
                    // TẠI SAO PHẢI CHO 2?
                    // Trong bài toán clustering, Intra-distance của 1 cluster được tính:
                    // Sum(D[i][j]) với i thuộc cluster, j thuộc cluster.
                    // Vì ma trận đối xứng (D[i][j] == D[j][i]), nếu ta đi qua tất cả cặp (i,j),
                    // khoảng cách giữa 2 đỉnh sẽ bị cộng 2 lần (1 lần từ i->j, 1 lần từ j->i).
                    // Chia 2 ngay lúc đọc giúp chuẩn hóa giá trị, tránh tính sai hàm mục tiêu.
                    instance.D[i][j] /= 2.0; 
                }
            }
            else // type == "t"
            {
                // LOẠI "t": Đọc tọa độ Euclidean (x, y)
                file >> C[i][0] >> C[i][1];
            }
        }

        // Nếu là loại "t", phải tự tính toán ma trận D từ tọa độ
        if (instance.type == "t")
        {
            const double factor = 1e6; // Hệ số dùng cho trick chuẩn hóa số thực
            for (int i = 0; i < instance.nV; ++i)
            {
                // Chỉ cần tính nửa tam giác trên (j > i) vì D là ma trận đối xứng
                for (int j = i + 1; j < instance.nV; ++j)
                {
                    // Tính khoảng cách Euclid: sqrt( (x1-x2)^2 + (y1-y2)^2 )
                    // TẠI SAO CÓ round(... / 2 * 1e6) / 1e6?
                    // Đây là "Floating-point precision trick" kinh điển:
                    // 1. Nhân với 1e6: Đẩy giá trị lên nguyên lớn để hàm round() hoạt động chuẩn.
                    // 2. Chia cho 2: Áp dụng luật chống đếm đôi (giống loại "p").
                    // 3. round(): Làm tròn số thực dư thập ở vị trí thập phân thứ 6.
                    // 4. Chia cho 1e6: Trả về thang đo ban đầu.
                    // -> Kết quả: Mất đi các lỗi nhiễu nhỏ dư thừa của IEEE 754 (vd: 5.00000000000001 -> 5.0)
                    double d = round(
                        sqrt(pow(C[i][0] - C[j][0], 2) + pow(C[i][1] - C[j][1], 2))
                        / 2.0 * factor) / factor;
                        
                    // Gán cho cả 2 chiều của ma trận đối xứng
                    instance.D[i][j] = d;
                    instance.D[j][i] = d;
                }
            }
        }

        // Đọc ràng buộc (bounds) cho từng cluster
        // Format: K dòng, mỗi dòng có nT số Lower Bound, rồi nT số Upper Bound
        for (int k = 0; k < instance.nK; ++k)
        {
            for (int t = 0; t < instance.nT; ++t) file >> instance.WL[k][t]; // Lower bound
            for (int t = 0; t < instance.nT; ++t) file >> instance.WU[k][t]; // Upper bound
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // XỬ LÝ "h" (HANDOVER - MẠNG KHÔNG DÂY)
    // Cấu trúc file hoàn toàn khác, giống định dạng đồ thị cạnh (Graph Edges)
    // ─────────────────────────────────────────────────────────────────────
    else if (instance.type == "h")
    {
        double handover_value;
        // Header: nV (đỉnh), nK (cluster), handover_value (bỏ qua vì không dùng trong code này)
        file >> instance.nV >> instance.nK >> handover_value;
        
        instance.nT = 1; // Bài toán handover luôn có chính xác 1 chiều trọng số (bandwidth)

        // Cấp phát ma trận (tương tự trên, kích thước nT=1)
        instance.W  = matDbl(instance.nV, vecDbl(instance.nT, 0.0));
        instance.D  = matDbl(instance.nV, vecDbl(instance.nV, 0.0));
        instance.WL = matDbl(instance.nK, vecDbl(instance.nT, 0.0)); // Mặc định bằng 0
        instance.WU = matDbl(instance.nK, vecDbl(instance.nT, 0.0));

        string ignore;

        // Đọc K dòng giới hạn trọng số cluster. Format: <label_src> <label_dst> <WU_value>
        // (Bài toán handover thường không có Lower Bound, hoặc mặc định bằng 0)
        for (int k = 0; k < instance.nK; ++k) {
            file >> ignore >> ignore >> instance.WU[k][0];
        }

        // Đọc nV dòng trọng số đỉnh. Format: <label_node> <label_ignore> <W_value>
        for (int i = 0; i < instance.nV; ++i) {
            file >> ignore >> ignore >> instance.W[i][0];
        }

        // Đọc các dòng định nghĩa cạnh mạng (các dòng còn lại trong file)
        string line;
        getline(file, line); // Bắt buộc phải tiêu thụ 1 ký tự newline ('\n') thừa ở cuối dòng trước đó

        while (getline(file, line))
        {
            // Tách dòng thành các từ (token) dựa trên khoảng trắng
            vector<string> tok = SplitString(line, ' '); // Hàm utility tách chuỗi (giả định có sẵn)
            if (tok.size() < 4) continue; // Bỏ qua dòng trống hoặc bị lỗi format

            int src = stoi(tok[1]) - 1; // Lấy index đỉnh nguồn (trừ 1 để chuyển về 0-indexed)
            int dst = stoi(tok[2]) - 1; // Lấy index đỉnh đích
            double weight = stod(tok[3]); // Lấy trọng số cạnh

            // TẠI SAO GÁN DẤU ÂM (-weight)?
            // Mục tiêu gốc của bài toán Handover/Wireless là MAXIMIZE tổng trọng số đường truyền.
            // Tuy nhiên, toàn bộ bộ máy ACO/Local Search/Tabu Search của bạn được thiết kế để MINIMIZE.
            // Bằng cách gán `-weight` vào ma trận D, bài toán Maximize được biến đổi về bài toán Minimize
            // một cách hoàn hảo về mặt toán học, mà không cần sửa lại một dòng code thuật toán nào bên dưới.
            instance.D[src][dst] = -weight;
        }
    }

    // Xử lý trường hợp file có type không hợp lệ
    else {
        cerr << "\n[ERROR] Unknown instance type: \"" << instance.type << "\"\n";
        exit(1);
    }

    file.close(); // Đóng file để giải phóng bộ nhớ
}


// ═══════════════════════════════════════════════════════════════════════════
// HÀM 3: LogParameters - In cấu hình ra màn hình
//
// Mục đích: In ra terminal các tham số bài toán TRƯỚC khi chạy.
// Nhờ hàm này, trong file log sẽ có rõ ràng bài toán nào đang được giải.
// ═══════════════════════════════════════════════════════════════════════════
void LogParameters()
{
    const string sep = "\n  "; // Ký tự tạo khoảng cách thụt vị cho đẹp

    cout << "\nAlgorithm: Ant Colony Optimization (ACO) for the MCGP problem.";
    cout << "\n  Each iteration, a colony of ants constructs independent solutions";
    cout << "\n  guided by a pheromone matrix. Each solution is refined by";
    cout << "\n  Local Search and Tabu Search before pheromone update.";
    cout << "\n  The global best solution is tracked continuously and saved to disk.";

    cout << "\n\nRun configuration:";
    cout << sep << "Termination time limit : " << termination_time << " seconds";
    cout << sep << "Max iterations       : " << maxIter;
    cout << sep << "Log frequency        : every " << parameters.ALGlg << " iteration(s)";
    cout << sep << "Output directory     : " << parameters.LOGdir;

    cout << "\n\nInstance:";
    cout << sep << "Path     : " << instance_path;
    cout << sep << "Type     : " << instance.type << "  (";
    
    // Giải thích bằng chữ rõ nghĩa của từng loại file
    if      (instance.type == "p") cout << "planar — distance matrix given directly";
    else if (instance.type == "t") cout << "tsp-like — Euclidean coordinates";
    else                           cout << "handover — wireless network edge weights";
    
    cout << ")";
    cout << sep << "Nodes    : " << instance.nV;
    cout << sep << "Clusters : " << instance.nK;
    cout << sep << "Weights  : " << instance.nT << " dimension(s)";

    cout << "\n" << endl; // Xuống 2 dòng trống để tách biệt với output của thuật toán phía sau
}
