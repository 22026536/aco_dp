# aco_dp

# Thuật toán chính
HÀM ACO_tuned(instance, maxIter, timeLimit):
    WHILE iter < maxIter VÀ thời gian < timeLimit:
        iter += 1

        TẠO m con kiến (n/2 hoặc 40 vì nếu quá nhiều kiến khiến thuật toán chạy chậm)
        CHO mỗi con kiến:
            XÁO TRỘN danh sách node
                kiến xây dựng lời giải
            END FOR

            TÍNH chi phí của con kiến (intra-cluster distance + penalty vi phạm)
            KIỂM TRA feasibility

        END FOR

        SẮP XẾP các con kiến theo chi phí
        CHO top repairTop con kiến:
            local search

        CẬP NHẬT best_solution nếu con kiến mới tốt hơn về chi phí hoặc feasibility

        CẬP NHẬT pheromone phi:

        NẾU không cải thiện trong nhiều vòng:
            RESET pheromone về giá trị ban đầu
    END WHILE

    TRẢ VỀ best_solution
END HÀM


# thông tin heuristic
Ở đây, thông tin heuristic được tính bằng 1 / [1 + (Delta thay đổi trọng số nếu gắn nút i vào cụm k) + (Pelnaty trọng số)]

Với Pelnaty trọng số = Hệ số phạt x Độ vi phạm từng trọng số nếu gắn i vào cụm K
                      + Hệ số khuyến khích x Độ thiếu hụt trọng số từng cụm nếu gắn i vào cụm K

Pelnaty trọng số sẽ giảm khả năng chọn cụm nếu quá tải nhưng nếu lựa chọn này đủ tốt thì vẫn có thể vi phạm được, đồng thời khuyến khích các nút chọn các cụm đang thiếu hụt trọng số dựa trên độ phù hợp về mặt trọng số của nút đó

Cả hệt số phạt và hệ số khuyến khích đều được tự điều chỉnh dựa trên Tổng chi phí, số lượng cặp nút, tổng trọng số, số lượng trọng số...
Công thức hiện tại: 50.0 * (Tổng tất cả cặp chi phí / Tổng số cặp nút) / [Tổng trọng số tất cả nút / (Số lượng nút * Số chiều trọng số)]

# Sự lựa chọn của kiến
weight[i][k] = pheromone[i][k]^α.heuristic[i][k]^β

tham số Q0:
Với q < Q0: kiến tham lam chọn cụm có lựa chọn tốt nhất
Với q > Q0: kiến chọn cụm theo tỉ lệ dựa trên weight[i][k]

Điều chỉnh: Ban đầu Q0 = 0.85. Với mỗi t0 vòng không cải thiện, giảm Q0 đi 0,05. Nếu tìm được lời giải thay thế best, đặt lại Q0 = 0.85. Điều này sẽ giúp nếu khi tìm được lời giải tốt thì sẽ tập trung khai thác lời giải đó, ngược lại nếu stagnation thì sẽ giảm Q0 để kiến khám phá

# Các tham số khác.
tham số bay hơi rho: 0,3 : mức bay hơi vừa để khiến kiến ít stagnation hơn
T_max = 0.2, T_min = 0.03 : T_max thấp để kiến ít đi theo lối mòn kiến trước để lại hơn.
