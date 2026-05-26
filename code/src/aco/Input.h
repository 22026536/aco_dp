#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// FILE: Input.h
//
// Header của tầng INPUT.
// Chứa: struct Instance, struct Parameters, SplitString,
//        khai báo các biến toàn cục liên quan đến input/config,
//        và khai báo hàm LoadInput / LoadInstance / LogParameters.
//
// Tầng Input KHÔNG biết gì về nội tại của thuật toán ACO.
// Tầng ACO include Input.h để lấy Instance khi cần.
// ═══════════════════════════════════════════════════════════════════════════

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

// Type aliases
using vecDbl  = vector<double>;
using matDbl  = vector<vector<double>>;

// ═══════════════════════════════════════════════════════════════════════════
// Parameters — cấu hình chạy (đọc từ command line / argv)
//
// LOGdir: thư mục gốc lưu kết quả
// ALGlg : tần suất ghi log (mỗi ALGlg iteration)
// ═══════════════════════════════════════════════════════════════════════════
struct Parameters {
    string LOGdir = "results/logs";
    int    ALGlg  = 10;
};

// ═══════════════════════════════════════════════════════════════════════════
// Instance — dữ liệu bài toán (đọc từ file instance)
//
// type : loại instance — "p" (planar, ma trận dist cho sẵn)
//                        "t" (TSP-like, tọa độ Euclid)
//                        "h" (handover, ma trận cạnh mạng không dây)
// nV   : số node (vertices)
// nK   : số cluster (partitions)
// nT   : số chiều trọng số (resource dimensions)
// W    : W[i][t]   = trọng số node i chiều t          (nV × nT)
// WL   : WL[k][t]  = lower bound cluster k chiều t    (nK × nT)
// WU   : WU[k][t]  = upper bound cluster k chiều t    (nK × nT)
// D    : D[i][j]   = khoảng cách từ node i đến j      (nV × nV)
// ═══════════════════════════════════════════════════════════════════════════
struct Instance {
    string type = "p";
    int nV = 0, nK = 0, nT = 0;
    matDbl W, WL, WU, D;
};

// ─────────────────────────────────────────────────────────────────────────
// SplitString — tách chuỗi theo delimiter
// Chỉ dùng trong Input.cpp (LoadInstance, parse file type "h").
// ─────────────────────────────────────────────────────────────────────────
inline vector<string> SplitString(const string &str, char delimiter)
{
    vector<string> tokens;
    stringstream   ss(str);
    string         token;
    while (getline(ss, token, delimiter))
        tokens.push_back(token);
    return tokens;
}

// ─────────────────────────────────────────────────────────────────────────
// Biến toàn cục của tầng Input
// (định nghĩa trong Input.cpp, dùng bởi main.cpp và LogParameters)
// ─────────────────────────────────────────────────────────────────────────
extern Instance   instance;         // dữ liệu bài toán (điền bởi LoadInstance)
extern Parameters parameters;       // cấu hình chạy   (điền bởi LoadInput)
extern int        maxIter;          // số iteration tối đa
extern double     termination_time; // giới hạn thời gian (giây) - ĐỒNG TÊN TỪ `parameters.ALGtv`
extern string     instance_path;    // đường dẫn file instance

// ─────────────────────────────────────────────────────────────────────────
// Hàm công khai của Input.cpp
// ─────────────────────────────────────────────────────────────────────────

// Đọc và phân tích tham số từ argv; trả về đường dẫn file instance
string LoadInput(int argc, const char *argv[], unsigned &seed);

// Mở và đọc file instance; điền vào biến toàn cục `instance`
void LoadInstance(const string &pathInstance);

// In tóm tắt cấu hình ra stdout trước khi thuật toán bắt đầu
void LogParameters();
