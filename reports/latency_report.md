# Latency Breakdown - Production Pipeline

Số liệu đo trong lần chạy `python src/pipeline.py` dùng cho `ragas_report.json` (20 câu hỏi, top_k=3, chạy CPU, không có GPU). Nguồn: trường `telemetry.latency_ms` trong báo cáo.

## Giai đoạn build (chạy một lần)

| Bước | Thời gian | Ghi chú |
|---|---:|---|
| Chunking (M1) | 0.15 s | 26 tài liệu thành 26 parent và 104 child |
| Enrichment (M5) | 190.0 s | 104 lượt gọi API, mỗi chunk một lượt, 2 worker |
| Indexing (M2) | 70.4 s | BM25 + embed bge-m3 + ghi Qdrant |
| Load reranker (M3) | 7.6 s | bge-reranker-v2-m3 |

## Giai đoạn truy vấn (trung bình mỗi câu)

| Bước | Tổng 20 câu | TB mỗi câu | Tỷ trọng |
|---|---:|---:|---:|
| Hybrid search (M2) | 6.3 s | 314 ms | 1.5% |
| Rerank (M3) | 352.4 s | 17,621 ms | 86.6% |
| LLM generation | 48.3 s | 2,413 ms | 11.9% |
| **Tổng** | **407.0 s** | **20,348 ms** | 100% |

Chấm RAGAS (M4) cho 20 câu mất thêm 231.2 s.

## Nhận xét

- **Rerank là nút cổ chai.** Mỗi câu hỏi phải chấm khoảng 11 parent chunk (mỗi chunk tối đa 2048 ký tự) bằng một cross-encoder 568M tham số trên CPU. Cách nhanh nhất để giảm là rerank child chunk (256 ký tự) trước rồi mới mở parent cho top-3, hoặc giới hạn số candidate trước khi rerank.
- **Search rất nhẹ**, chỉ khoảng 0.3 s mỗi câu, nên không cần tối ưu thêm ở bước này.
- **Enrichment tốn thời gian nhưng chỉ chạy một lần** và đã được cache theo hash của chunk và model, nên các lần chạy sau gần như bằng 0.
- Baseline (dense-only, không rerank) có chunking 0.29 s và indexing 67.5 s, tức chi phí build tương đương. Khác biệt lớn nằm ở bước rerank lúc truy vấn.
