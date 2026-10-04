# Failure Analysis - Lab 18: Production RAG

**Họ và tên:** Đỗ Thanh Tùng  
**MSSV:** 2A202602845  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 05/10/2026

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.8769 | 0.8650 | -0.0119 |
| Answer Relevancy | 0.8926 | 0.8742 | -0.0184 |
| Context Precision | 0.8833 | 0.9250 | +0.0417 |
| Context Recall | 0.9250 | 0.9750 | +0.0500 |

Bộ test có 20 câu. Câu trả lời sinh bằng `qwen/qwen3-30b-a3b-instruct-2507`, judge là `qwen/qwen3-235b-a22b-2507` (qua OpenRouter), embedding dùng bge-m3. Số liệu Production lấy từ `reports/ragas_report.json`.

Production cải thiện rõ ở phần retrieval: Precision tăng 4 điểm, Recall tăng 5 điểm, nhờ hybrid search, rerank và trả về parent chunk. Faithfulness và Relevancy nhích xuống một chút, chủ yếu do hai câu trả lời quá ngắn (#2 và #3 bên dưới), không phải do lấy sai tài liệu.

Có hai điểm cần lưu ý khi đọc bảng:

- **Faithfulness của câu tạm ứng (#2) không đo được.** Câu trả lời chỉ là "50.000 VNĐ", RAGAS không tách được mệnh đề nào để kiểm chứng nên trả về NaN. Báo cáo tính ô này là 0, vì vậy 0.8650 là con số thận trọng. Nếu bỏ ô này, trung bình 19 câu còn lại là 0.9105.
- **Baseline chỉ còn điểm tổng.** Điểm tổng lấy từ bảng so sánh mà `main.py` in ra ở lần chạy hoàn chỉnh. Điểm từng câu của baseline đã bị ghi đè khi các lần chạy lại sau đó hết credit API, nên mình không dựng lại.

## Bottom-5 Failures

Xếp theo trung bình bốn metric, từ thấp lên.

### #1 - Câu hỏi hai ý: phép năm và lương Senior (TB 0.519)

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + 3 = 18 ngày phép (v2024); lương Senior (P3-P4) 20-35 triệu/tháng.
- **Got:** 18 ngày phép năm có lương (15 cơ bản + 3 thâm niên). Không nhắc gì đến lương.
- **Worst metric:** context_precision = 0.00 (recall 0.50, faithfulness 0.67).
- **Context:** nghi_phep_nam_v2024.md, nghi_phep_nam_v2023.md, nghi_phep_khong_luong.md.
- **Error Tree:** Output thiếu ý lương → Context có bảng lương không? Không → Query OK? Một query chứa hai ý, cả ba slot đều bị tài liệu nghỉ phép chiếm → **lỗi retrieval**.
- **Root cause:** Phần "nghỉ phép" khớp mạnh ở cả BM25 lẫn dense nên chiếm hết top-3. Với câu hỏi này, bang_luong_2024.md chỉ liên quan qua chữ "Senior" nên không cạnh tranh nổi.
- **Fix:** Tách câu hỏi thành hai sub-query, mỗi ý giữ ít nhất một nguồn, gộp lại rồi mới rerank. Bỏ bản v2023 khỏi top-k khi đã có bản mới hơn.

### #2 - Tạm ứng quá hạn (TB 0.607)

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Hạn 15 ngày, quá 5 ngày; phí 2%/tháng, tính pro-rata khoảng 50.000 VNĐ.
- **Got:** 50.000 VNĐ
- **Worst metric:** faithfulness = NaN (tính là 0); answer_relevancy 0.43.
- **Context:** tam_ung.md, chi_phi_expense.md, bao_hiem_suc_khoe.md.
- **Error Tree:** Output đúng số → Context đúng (precision và recall đều 1.0) → lỗi nằm ở **cách viết câu trả lời**.
- **Root cause:** Kết quả tính toán đúng (calculator ra `15000000 * 2/100 * (20-15)/30 = 50000`), nhưng model chỉ trả về con số trần. RAGAS không có mệnh đề nào để đối chiếu với context, và câu hỏi sinh ngược từ một con số cũng không khớp câu gốc, nên relevancy thấp.
- **Fix:** Bắt buộc prompt trả lời thành câu đầy đủ kèm căn cứ, ví dụ "Quá hạn 5 ngày, phí 2%/tháng trên 15 triệu, tính theo ngày là 50.000 VNĐ".

### #3 - Hoàn trả chi phí đào tạo (TB 0.708)

- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết tối thiểu 1 năm; nghỉ sau 8 tháng phải hoàn trả 100%, tức 25.000.000 VNĐ.
- **Got:** Nhân viên phải hoàn trả 25.000.000 VNĐ.
- **Worst metric:** faithfulness = 0.00.
- **Context:** hoan_chi_dao_tao.md, tam_ung.md, thuong_tet.md.
- **Error Tree:** Output đúng đáp số → Context đúng (precision và recall đều 1.0) → câu trả lời **bỏ mất bước suy luận**.
- **Root cause:** Con số 25 triệu đến từ câu hỏi chứ không có trong tài liệu. Câu trả lời cũng không nhắc quy tắc "dưới 12 tháng thì hoàn trả 100%", nên judge không tìm thấy căn cứ trong context và chấm 0.
- **Fix:** Yêu cầu model nêu quy tắc trước rồi mới áp vào số liệu: "8 tháng < 12 tháng cam kết nên hoàn trả 100% × 25 triệu".

### #4 - Số ngày phép năm (TB 0.829)

- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** 15 ngày theo v2024; bản v2023 (12 ngày) đã bị thay thế.
- **Got:** 15 ngày theo phiên bản 2.0 (2024), kèm giải thích thêm về cộng thâm niên.
- **Worst metric:** context_precision = 0.50.
- **Context:** nghi_phep_nam_v2023.md, nghi_phep_nam_v2024.md, nghi_phep_dac_biet.md.
- **Error Tree:** Output đúng → Context đủ nhưng **bản cũ v2023 xếp trên bản mới** → lỗi thứ tự rerank.
- **Root cause:** Hai phiên bản gần như giống hệt nhau về nội dung nên cross-encoder không phân biệt được bản nào hiện hành. Model vẫn chọn đúng nhờ đọc ngày hiệu lực.
- **Fix:** Lưu `version` và `effective_date` vào metadata, ưu tiên bản mới nhất khi người dùng không hỏi về lịch sử chính sách.

### #5 - Mua laptop 30 triệu (TB 0.913)

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Director phê duyệt (khoảng 5-50 triệu); cần CNTT xác nhận cấu hình; cần ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Director phê duyệt; CNTT xác nhận cấu hình. Thiếu ý 3 báo giá.
- **Worst metric:** answer_relevancy = 0.82.
- **Context:** mua_sam.md, phan_loai_du_lieu.md, dao_tao_noi_bo.md.
- **Error Tree:** Output thiếu một điều kiện → Context có đủ (recall 1.0) → lỗi ở **bước generation**.
- **Root cause:** Model trả lời đúng hai điều được hỏi trực tiếp và bỏ qua điều kiện đi kèm trong cùng quy trình.
- **Fix:** Thêm vào prompt yêu cầu liệt kê mọi điều kiện áp dụng cho mức giá đó: thẩm quyền duyệt, xác nhận kỹ thuật, số báo giá.

## Case Study: câu Senior (#1)

Mình chọn câu này vì nó là lỗi retrieval thật duy nhất trong bottom-5. Các câu còn lại đều đã có context đúng.

**Error Tree walkthrough:**

1. **Output đúng?** Đúng một nửa: 18 ngày phép là đúng, nhưng thiếu khoảng lương 20-35 triệu.
2. **Context đúng?** Không. Cả ba context đều là tài liệu nghỉ phép, kể cả bản v2023 đã cũ. bang_luong_2024.md không lọt vào top-3.
3. **Query rewrite OK?** Không có bước rewrite. Pipeline đưa nguyên câu hai ý vào search, và ý "nghỉ phép" lấn át ý "lương".
4. **Fix ở bước:** Retrieval, cụ thể là query decomposition. Sửa prompt ở đây không giúp được vì model không thể trả lời phần lương khi không có tài liệu.

**Nếu có thêm 1 giờ:**

- Thêm bước tách câu hỏi nhiều ý thành sub-query (khoảng 20 phút), mỗi sub-query lấy riêng top-k rồi gộp lại.
- Lọc phiên bản cũ bằng metadata `effective_date` (khoảng 15 phút), xử lý luôn câu #4.
- Sửa prompt để luôn trả lời bằng câu đầy đủ, nêu quy tắc trước rồi mới tính (khoảng 10 phút), xử lý câu #2, #3 và #5.
- Chạy lại cả 20 câu với cùng judge để so sánh (khoảng 15 phút).

Bảng latency từng bước nằm ở [reports/latency_report.md](../reports/latency_report.md).
