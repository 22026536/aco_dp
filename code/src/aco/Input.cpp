// ═══════════════════════════════════════════════════════════════════════════
// FILE: input.cpp
//
// Triển khai các hàm khai báo trong input.h:
//   LoadInput      – phân tích tham số dòng lệnh
//   LoadInstance   – đọc file dữ liệu bài toán
//   LogParameters  – in cấu hình chạy ra stderr
// ═══════════════════════════════════════════════════════════════════════════

#include "input.h"


// ═══════════════════════════════════════════════════════════════════════════
// LoadInput()
// ═══════════════════════════════════════════════════════════════════════════

string LoadInput(int argc, const char *argv[], unsigned &seed)
{
    string pathInstance;

    for (int i = 1; i < argc; i += 2)
    {
        bool matched = false;

        if (argv[i][0] == '-' && i + 1 < argc)
        {
            string key   = argv[i];
            string value = argv[i + 1];

            // Đường dẫn file dữ liệu
            if (key == "--instance")
            {
                matched = true;
                pathInstance = value;
            }
            // Kiểu điều kiện dừng: "time" hoặc "iter"
            else if (key == "--termination_criteria")
            {
                matched = true;
                parameters.ALGtc = value;
            }
            // Seed cho bộ sinh số ngẫu nhiên (0 = ngẫu nhiên)
            else if (key == "--seed")
            {
                matched = true;
                seed = stoi(value);
            }
            // Số vòng lặp ACO tối đa
            else if (key == "--iter_value")
            {
                matched = true;
                maxIter = stoi(value);
            }
            // Giới hạn thời gian chạy (giây)
            else if (key == "--termination_value")
            {
                matched = true;
                termination_time = stod(value);
                parameters.ALGtv = stod(value);
            }
            // Nhãn sơ đồ xây dựng lời giải (giữ lại để tương thích script)
            else if (key == "--schema")
            {
                matched = true;
                parameters.CONm = value;
            }
            // Tần suất in log (sau mỗi N vòng lặp)
            else if (key == "--logs")
            {
                matched = true;
                parameters.ALGlg = stoi(value);
            }
            // ── Tham số GRASP cũ (giữ lại để tương thích script) ─────────
            else if (key == "--version")  { matched = true; parameters.GRASPv = value; }
            else if (key == "--alpha")    { matched = true; parameters.GRASPa = stod(value); }
            else if (key == "--m")        { matched = true; parameters.GRASPm = stoi(value); }
            else if (key == "--block")    { matched = true; parameters.GRASPb = stoi(value); }
            else if (key == "--delta")    { matched = true; parameters.GRASPd = stoi(value); }
            // ── Tham số Local Search cũ (giữ lại để tương thích script) ──
            else if (key == "--move")       { matched = true; parameters.LSm = value; }
            else if (key == "--efficient")  { matched = true; parameters.LSe = (stoi(value) != 0); }
            else if (key == "--exploration")
            {
                matched = true;
                parameters.LSs = value;
                if (value != "best" && value != "hybrid" && value != "first")
                {
                    cerr << "\nUnknown exploration strategy: " << value
                         << "  (valid: best | first | hybrid)\n";
                    exit(1);
                }
            }
            // Chế độ debug
            else if (key == "--debug")
            {
                matched = true;
                if      (value == "1" || value == "true")  parameters.DEBUG = true;
                else if (value == "0" || value == "false") parameters.DEBUG = false;
                else
                {
                    cerr << "\nInvalid --debug value: " << value
                         << "  (valid: 1 | 0 | true | false)\n";
                    exit(1);
                }
            }
        }

        if (!matched)
        {
            cerr << "\nUnknown argument: " << argv[i] << "\n";
            exit(1);
        }
    }

    if (pathInstance.empty())
    {
        cerr << "\nMissing required argument: --instance <path>\n";
        exit(1);
    }

    return pathInstance;
}


// ═══════════════════════════════════════════════════════════════════════════
// LoadInstance()
// ═══════════════════════════════════════════════════════════════════════════

void LoadInstance(const string &pathInstance)
{
    ifstream file(pathInstance);
    if (!file)
    {
        cerr << "\nCannot open instance file: " << pathInstance << "\n";
        exit(1);
    }

    file >> instance.type;

    // ── Định dạng "p" và "t" ─────────────────────────────────────────────
    if (instance.type == "p" || instance.type == "t")
    {
        file >> instance.nV >> instance.nK >> instance.nT;

        instance.W  = matDbl(instance.nV, vecDbl(instance.nT, 0.0));
        instance.D  = matDbl(instance.nV, vecDbl(instance.nV, 0.0));
        instance.WL = matDbl(instance.nK, vecDbl(instance.nT, 0.0));
        instance.WU = matDbl(instance.nK, vecDbl(instance.nT, 0.0));

        matDbl C(instance.nV, vecDbl(2, 0.0));   // tọa độ (chỉ dùng cho loại "t")

        for (int i = 0; i < instance.nV; i++)
        {
            string ignore;
            file >> ignore;                       // bỏ qua nhãn nút

            for (int t = 0; t < instance.nT; t++)
                file >> instance.W[i][t];

            if (instance.type == "p")
            {
                // File lưu 2*D, nên chia đôi khi đọc vào.
                for (int j = 0; j < instance.nV; j++)
                {
                    file >> instance.D[i][j];
                    instance.D[i][j] /= 2.0;
                }
            }
            else
            {
                file >> C[i][0] >> C[i][1];
            }
        }

        // Tính khoảng cách Euclid cho loại "t".
        if (instance.type == "t")
        {
            const double factor = 1e6;            // làm tròn đến 6 chữ số thập phân
            for (int i = 0; i < instance.nV; i++)
                for (int j = i + 1; j < instance.nV; j++)
                {
                    // Chia đôi vì chi phí tính tổng D[i][j]+D[j][i] cho mỗi cặp.
                    instance.D[i][j] = round(
                        sqrt(pow(C[i][0]-C[j][0], 2) + pow(C[i][1]-C[j][1], 2))
                        / 2.0 * factor) / factor;
                    instance.D[j][i] = instance.D[i][j];
                }
        }

        // Đọc giới hạn trọng số cho từng cụm.
        for (int k = 0; k < instance.nK; k++)
        {
            for (int t = 0; t < instance.nT; t++) file >> instance.WL[k][t];
            for (int t = 0; t < instance.nT; t++) file >> instance.WU[k][t];
        }
    }
    // ── Định dạng "h" ────────────────────────────────────────────────────
    else if (instance.type == "h")
    {
        double handover;
        file >> instance.nV >> instance.nK >> handover;
        instance.nT = 1;

        instance.W  = matDbl(instance.nV, vecDbl(instance.nT, 0.0));
        instance.D  = matDbl(instance.nV, vecDbl(instance.nV, 0.0));
        instance.WL = matDbl(instance.nK, vecDbl(instance.nT, 0.0));
        instance.WU = matDbl(instance.nK, vecDbl(instance.nT, 0.0));

        string ignore;

        // Giới hạn trên dung lượng cụm (WL mặc định bằng 0).
        for (int k = 0; k < instance.nK; k++)
            file >> ignore >> ignore >> instance.WU[k][0];

        // Trọng số từng nút.
        for (int i = 0; i < instance.nV; i++)
            file >> ignore >> ignore >> instance.W[i][0];

        // Các cạnh: "edge <src> <dst> <weight>"
        // D bị đảo dấu để các cặp lưu lượng cao hút nhau vào cùng một cụm.
        string line;
        getline(file, line);   // tiêu thụ phần còn lại của dòng hiện tại
        while (getline(file, line))
        {
            vector<string> tok = SplitString(line, ' ');
            if (tok.size() < 4) continue;
            instance.D[stoi(tok[1])-1][stoi(tok[2])-1] = -stod(tok[3]);
        }
    }
    else
    {
        cerr << "\nUnknown instance type: \"" << instance.type << "\"\n";
        exit(1);
    }

    file.close();
}


// ═══════════════════════════════════════════════════════════════════════════
// LogParameters()
// ═══════════════════════════════════════════════════════════════════════════

void LogParameters()
{
    // ── Mô tả thuật toán ──────────────────────────────────────────────────
    cout << "\n Structure: Ant Colony Optimization (ACO) for the MCGP problem.";
    cout << "\n   Each iteration, a colony of ants constructs independent solutions";
    cout << "\n   guided by a pheromone matrix. Each solution is then refined by";
    cout << "\n   Local Search and Tabu Search before pheromone update.";
    cout << "\n   The global best solution is tracked continuously and saved to disk.";

    // ── Tham số thuật toán ────────────────────────────────────────────────
    cout << "\nAlgorithm parameters:";
    cout << "\n  - Search method        : ACO (Ant Colony Optimization)";
    cout << "\n  - Termination criteria : " << parameters.ALGtc << " = " << parameters.ALGtv;
    cout << "\n  - Max iterations       : " << maxIter;
    cout << "\n  - Log frequency        : every " << parameters.ALGlg << " iteration(s)";

    // ── Tham số dữ liệu bài toán ──────────────────────────────────────────
    cout << "\nInstance parameters:";
    cout << "\n  - Path     : " << instance_path;
    cout << "\n  - Type     : " << instance.type
         << "  ("
         << (instance.type == "p" ? "planar — distance matrix given directly"
           : instance.type == "t" ? "tsp-like — Euclidean coordinates"
                                  : "handover — wireless network edge weights")
         << ")";
    cout << "\n  - Nodes    : " << instance.nV;
    cout << "\n  - Clusters : " << instance.nK;
    cout << "\n  - Weights  : " << instance.nT << " dimension(s)";

    // ── Đầu ra ────────────────────────────────────────────────────────────
    cout << "\nOutput:";
    cout << "\n  - Log dir  : " << parameters.LOGdir;

    cout << "\n" << endl;
}
