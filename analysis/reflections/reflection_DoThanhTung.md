# Individual Reflection - Lab 18: Production RAG

**Họ và tên:** Đỗ Thanh Tùng  
**MSSV:** 2A202602845  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 05/10/2026

---

## Phần 1: Mapping bài giảng

| Lecture Concept | Module | Hàm cụ thể | Observation |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Với threshold 0.85, toàn bộ corpus tách thành 208 chunk, trong khi basic chỉ có 51. Nhiều chunk rất ngắn (ngắn nhất 6 ký tự) vì ngưỡng cao làm câu nào cũng thành một nhóm riêng. Vì vậy mình không dùng nó cho pipeline. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Đây là strategy pipeline dùng: 26 parent, 104 child. Retrieve bằng child 256 ký tự cho chính xác, rồi đưa parent 2048 ký tự cho LLM để đủ ngữ cảnh. Context Recall tăng từ 0.925 lên 0.975, và việc trả về parent là một phần lý do. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | BM25 (sau khi tách từ bằng underthesea) hợp với từ khóa chính xác như "MFA", "PVI"; bge-m3 hợp với câu hỏi diễn đạt khác tài liệu. RRF (k=60) gộp hai danh sách theo thứ hạng nên không cần chuẩn hóa điểm. Search chỉ mất khoảng 314 ms mỗi câu. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Context Precision tăng từ 0.883 lên 0.925 so với baseline. Đổi lại, rerank tốn khoảng 17.6 s mỗi câu trên CPU (chấm khoảng 11 parent mỗi câu), chiếm 87% thời gian truy vấn. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Faithfulness thấp nhất (0.865). Nguyên nhân không phải hallucination mà là hai câu trả lời quá ngắn, chỉ có con số. Một câu RAGAS không tách được mệnh đề nên trả về NaN, câu còn lại bị chấm 0 vì con số đến từ câu hỏi chứ không có trong context. |
| Contextual embeddings | M5 | `_enrich_single_call()` | Mỗi chunk chỉ cần một lượt gọi API để trả về cả summary, câu hỏi giả định, câu ngữ cảnh và metadata. Cả 104 chunk được làm giàu thành công. Kết quả được cache theo hash nên lần chạy sau không tốn thêm lượt gọi nào. |

---

## Phần 2: Khó khăn và cách giải quyết

**1. Reranker không load được**

```text
OSError: BAAI/bge-reranker-v2-m3 does not appear to have a file named pytorch_model.bin or model.safetensors.
```

Mạng chập chờn khiến HuggingFace tải dở model, và log báo `peer closed connection without sending complete message body`. Cache chỉ có config và tokenizer nên 5 test của M3 fail. Mình mở thư mục snapshot thì thấy thiếu file trọng số, nên tải lại riêng `model.safetensors`, chỉ lấy đúng các file cần thiết và có retry khi mạng đứt. Sau đó 37/37 test pass.

**2. Hết credit giữa lúc chấm RAGAS**

```text
This request requires more credits, or fewer max_tokens. You requested up to 16384 tokens, but can only afford 12057.
```

Mình không đặt `max_tokens` cho judge, nên OpenRouter giữ chỗ tối đa 16K token cho mỗi request và tài khoản hết tiền rất nhanh. Mình sửa bằng cách đặt `max_tokens=1024` và chỉ chạy một worker. Tuy vậy, credit vẫn hết ở các lần chạy thử thêm. Bài học là các lần chạy hỏng sẽ ghi đè lên kết quả tốt. Mình mất điểm từng câu của baseline vì vậy, chỉ còn giữ được điểm tổng từ log.

**3. Thử chuyển sang provider miễn phí**

```text
Error code: 400 - Multiple candidates is not enabled for this model
Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit: 20
```

Mình thử Gemini thì gặp hai vấn đề. Thứ nhất, metric `answer_relevancy` của RAGAS gửi `n=3`, nhưng Gemini không hỗ trợ. Mình viết một lớp bọc để gửi 3 request riêng lẻ thay vì một request `n=3`. Thứ hai, free tier chỉ cho 20 request mỗi ngày cho mỗi model, trong khi một lần chạy đầy đủ cần khoảng 420 request. Groq cũng không đủ vì giới hạn 200K token mỗi ngày. Cuối cùng mình giữ kết quả đo bằng Qwen, nhưng code giờ đã có retry với backoff, giới hạn số request mỗi phút và cấu hình provider riêng cho enrichment.

**Kiến thức còn thiếu và cách bổ sung:**

- RAGAS chỉ đo việc câu trả lời có bám context hay không, không đo đúng sai. Một đáp số đúng nhưng viết cụt vẫn bị điểm thấp. Mình đọc source các metric của ragas 0.1 để hiểu từng metric gọi LLM như thế nào.
- Với lab chạy bằng API, cần ước lượng số request và token trước khi chạy. Lần sau mình sẽ đo thử trên 1-2 câu, nhân lên cho cả bộ test rồi mới chạy thật.
- Hai file PDF dạng scan bị bỏ qua vì chưa OCR, nên corpus thật chỉ có 26 trên 28 tài liệu.

---

## Phần 3: Action Plan cho project

### Project: Trợ lý tra cứu chính sách nội bộ

#### Hiện tại

- **Pipeline:** hierarchical chunking, BM25 + bge-m3 + RRF, cross-encoder rerank parent, enrichment một lượt gọi mỗi chunk, có cache.
- **Known issues:** câu hỏi nhiều ý bị thiếu nguồn (câu Senior); bản chính sách cũ xếp trên bản mới; rerank chậm trên CPU; câu trả lời số học viết quá ngắn; PDF scan chưa đọc được.

#### Plan áp dụng

1. [ ] **Chunking:** giữ hierarchical vì Recall đang tốt; thêm OCR cho PDF scan và lưu `version`, `effective_date` vào metadata.
2. [ ] **Search:** giữ hybrid + RRF; thêm query decomposition cho câu hỏi nhiều ý, mỗi ý giữ ít nhất một nguồn.
3. [ ] **Reranking:** vẫn dùng bge-reranker-v2-m3, nhưng rerank child trước rồi mới mở parent cho top-3, mục tiêu giảm từ 17 s xuống dưới 3 s mỗi câu.
4. [ ] **Evaluation:** RAGAS 4 metric làm thước đo chính, thêm kiểm tra đáp số cho câu số học, và ước lượng chi phí API trước mỗi lần chạy.
5. [ ] **Enrichment:** giữ combined mode một lượt gọi mỗi chunk vì rẻ và đã có cache; prompt sinh câu trả lời phải nêu quy tắc trước khi tính.

#### Timeline

- **Tuần 1 (06-12/10):** query decomposition, lọc bản cũ theo metadata, sửa prompt trả lời đầy đủ câu.
- **Tuần 2 (13-19/10):** tối ưu rerank (child trước, parent sau), đo lại latency.
- **Tuần 3 (20-26/10):** OCR hai PDF scan, mở rộng bộ test lên khoảng 40 câu, chạy lại RAGAS với cùng judge để so sánh.
