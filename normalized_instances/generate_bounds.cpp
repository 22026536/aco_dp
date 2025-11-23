// Đỉnh cao vibe code

#include <bits/stdc++.h>
#include <filesystem>
using namespace std;
namespace fs = std::filesystem;

using vecDbl = vector<double>;
using matDbl = vector<vecDbl>;

struct Instance {
    string type;
    unsigned nV, nK, nT;
    matDbl W, D, WL, WU;
};

// === Đọc dữ liệu cơ bản (p, t) ===
Instance LoadBaseInstance(const string& path) {
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "❌ Cannot open file: " << path << endl;
        exit(1);
    }

    Instance inst;
    string ignore;
    file >> inst.type;

    if (inst.type == "p") {
        file >> inst.nV >> inst.nK >> inst.nT;
        inst.W  = matDbl(inst.nV, vecDbl(inst.nT, 0.0));
        inst.D  = matDbl(inst.nV, vecDbl(inst.nV, 0.0));
        for (unsigned i = 0; i < inst.nV; i++) {
            file >> ignore;
            for (unsigned t = 0; t < inst.nT; t++) file >> inst.W[i][t];
            for (unsigned j = 0; j < inst.nV; j++) {
                file >> inst.D[i][j];
                inst.D[i][j] /= 2.0;
            }
        }
    }
    else if (inst.type == "t") {
        file >> inst.nV >> inst.nK >> inst.nT;
        inst.W  = matDbl(inst.nV, vecDbl(inst.nT, 0.0));
        inst.D  = matDbl(inst.nV, vecDbl(inst.nV, 0.0));
        matDbl C(inst.nV, vecDbl(2, 0.0));

        for (unsigned i = 0; i < inst.nV; i++) {
            file >> ignore;
            for (unsigned t = 0; t < inst.nT; t++) file >> inst.W[i][t];
            file >> C[i][0] >> C[i][1];
        }

        unsigned decimals = 6;
        double factor = pow(10, decimals);
        for (unsigned i = 0; i < inst.nV; i++) {
            for (unsigned j = i + 1; j < inst.nV; j++) {
                inst.D[i][j] = round(
                    sqrt(pow(C[i][0] - C[j][0], 2) + pow(C[i][1] - C[j][1], 2))
                    / 2.0 * factor) / factor;
                inst.D[j][i] = inst.D[i][j];
            }
        }
    } else {
        cerr << "⚠️ Unsupported type in file: " << path << endl;
    }

    file.close();
    return inst;
}

// === Tính WL và WU ===
void ComputeWLWU(Instance &inst) {
    unsigned decimals = 2;
    double factor = pow(10, decimals);
    double mean, stddev;
    inst.WL = matDbl(inst.nK, vecDbl(inst.nT, 0.0));
    inst.WU = matDbl(inst.nK, vecDbl(inst.nT, 0.0));

    for (unsigned t = 0; t < inst.nT; t++) {
        double sum = 0.0;
        for (unsigned i = 0; i < inst.nV; i++) sum += inst.W[i][t];
        mean = sum / inst.nV;

        stddev = 0.0;
        for (unsigned i = 0; i < inst.nV; i++)
            stddev += pow(inst.W[i][t] - mean, 2);
        stddev = sqrt(stddev / inst.nV);

        for (unsigned k = 0; k < inst.nK; k++) {
            inst.WL[k][t] = round((mean * ((double)inst.nV / inst.nK) - 2.0 * stddev) * factor) / factor;
            inst.WU[k][t] = round((mean * ((double)inst.nV / inst.nK) + 2.0 * stddev) * factor) / factor;
        }
    }
}

// === Ghi append vào cuối file gốc ===
void AppendBounds(const string& path, const Instance& inst) {
    ofstream out(path, ios::app); // mở ở chế độ append
    if (!out.is_open()) {
        cerr << "❌ Cannot write to file: " << path << endl;
        return;
    }

    for (unsigned k = 0; k < inst.nK; k++) {
        for (unsigned t = 0; t < inst.nT; t++)
            out << inst.WL[k][t] << " ";
        out << "\n";

        for (unsigned t = 0; t < inst.nT; t++)
            out << inst.WU[k][t] << " ";
        out << "\n";
    }
    out.close();
}

// === Main ===
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    vector<string> folders = {"./pollster", "./tsplib"};
    unsigned count = 0;

    for (auto& folder : folders) {
        if (!fs::exists(folder)) {
            cerr << "⚠️ Folder not found: " << folder << endl;
            continue;
        }

        for (auto& entry : fs::directory_iterator(folder)) {
            if (entry.path().extension() == ".txt") {
                string path = entry.path().string();
                cout << "Processing: " << path << endl;

                try {
                    Instance inst = LoadBaseInstance(path);
                    ComputeWLWU(inst);
                    AppendBounds(path, inst);
                    count++;
                } catch (exception &e) {
                    cerr << "❌ Error on file " << path << ": " << e.what() << endl;
                }
            }
        }
    }

    cout << "\n✅ Finished " << count << " files.\n";
    return 0;
}
