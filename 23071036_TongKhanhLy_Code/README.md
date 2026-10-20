# INS315501 – Midterm Project: Traditional Image Processing
Filtering · Stereo 3D Reconstruction · Image Stitching. Chỉ dùng phương pháp cổ điển (OpenCV/NumPy), không deep learning.

## Cách chạy
Mở `TongKhanhLy_23071036_Midterm_code.ipynb` (Jupyter / VS Code / Colab) và chạy lần lượt từng ô. Notebook đã chạy sẵn, kết quả nằm ngay trong file.

Hoặc chạy bằng file .py (cần Python 3.9+):
```bash
pip install -r requirements.txt
python run_all.py                    # chạy cả 3 phần, kết quả trong results/
```
Dữ liệu Middlebury và ảnh tự chụp đã có sẵn trong `data/`. Nếu thiếu thì chạy `python download_middlebury.py`.
Chạy từng phần: `python part_a_filtering.py [--image ...]`, `python part_b_stereo.py [--scenes Venus Teddy]`, `python part_c_stitching.py [--images a.jpg b.jpg c.jpg d.jpg]`.

## Xem hình 3D xoay được (Part B)
| Cách | Làm gì |
|---|---|
| **Trình duyệt (dễ nhất)** | Mở `results/part_b/venus/B2_point_cloud_sgbm_interactive.html` → kéo chuột để xoay, cuộn để zoom (cần internet để tải plotly.js) |
| **MeshLab** | File → Import Mesh → `results/part_b/venus/point_cloud_sgbm.ply` |
| **Open3D** | `python view_point_cloud.py results/part_b/venus/point_cloud_sgbm.ply` |

Trong file nộp chỉ giữ point cloud `sgbm` của mỗi scene cho nhẹ. Chạy lại code sẽ tạo thêm `bm` và `groundtruth` (dựng từ disparity chuẩn của Middlebury) để so sánh.

## Cấu trúc
| File | Nội dung |
|---|---|
| `metrics.py` | **Tất cả hàm tính chỉ số**: MSE, PSNR, SSIM, EPI, Edge-F1, sharpness · density, LR-consistency, bad-pixel, EPE, RMSE, sai số epipolar, góc epipolar · inlier ratio, reprojection error, overlap RMSE, black-border %, straightness |
| `part_a_filtering.py` | Mean / Gaussian / Median (kernel 3, 5, 7) / Laplacian sharpening, 2 loại nhiễu (Gaussian σ=20, salt & pepper 5%), thêm σ = 0.5 và 2.0. Chạy trên ảnh tự chụp `data/own_photos/part_a/IMG_3115.jpg` → `results/part_a/own_photo/` |
| `part_b_stereo.py` | Middlebury Venus + Teddy: BM vs SGBM so với ground truth, khảo sát blockSize và numDisparities, point cloud (HTML + PLY), F: 8-point vs RANSAC vs LMedS (sai số epipolar + Sampson) |
| `part_c_stitching.py` | 4 ảnh tự chụp trong `data/own_photos/part_c/` (**tự sắp thứ tự trái → phải**): SIFT vs ORB vs AKAZE, KNN + ratio test 0.75, RANSAC homography (ngưỡng 1/3/5/10 px), so sánh least squares / RANSAC / LMedS, planar vs cylindrical (tiêu cự lấy từ EXIF), crop không viền đen, blending overwrite / feather / multi-band |
| `download_middlebury.py` | Tải `other-color-twoframes.zip`, `other-gt-flow.zip`, `eval-color-twoframes.zip` từ link của thầy |
| `data/` | `own_photos/part_a/` (A: 1 ảnh tự chụp) · `middlebury/` (B: Venus, Teddy từ link của thầy + ground truth) · `own_photos/part_c/` (C: 4 ảnh tự chụp, giữ nguyên EXIF) |

## Ghi chú về dữ liệu Part B
- Link thầy gửi là bộ **optical flow**. Trong đó có 2 scene thật ra là **cặp stereo**: **Venus** (có ground-truth flow) và **Teddy** (ground truth bị ẩn). Với cặp stereo, `frame10` là ảnh trái, `frame11` là ảnh phải, và thành phần ngang của flow = −disparity. Vì vậy code đọc file `.flo` để lấy disparity chuẩn.
- Dữ liệu đã có sẵn trong `data/middlebury/` (lấy từ link của thầy): `other-data/Venus`, `eval-data/Teddy`, `other-gt-flow/Venus/flow10.flo`.
- Hai ảnh `frame10`/`frame11` bị cắt lệch nhau theo chiều ngang (Venus 10 px, Teddy 26 px), nên disparity có thể **âm**. Code tự ước lượng khoảng tìm disparity từ các cặp điểm SIFT, và tính độ sâu bằng Z = f·B/(d + doffs).
- Ground truth của Teddy bị ẩn trong bộ flow, nên dùng `stereo-gt/Teddy/disp2.png` (ground truth công khai của bộ Middlebury stereo), cắt về đúng khung hình.
- Không có calibration nên độ sâu `Z = f·B/d` dùng f = 0.8·width, B = 1. Hình dạng 3D đúng nhưng tỉ lệ là tương đối.

## Ý nghĩa các chỉ số chính
- **Bad1.0 %**: % pixel có |d − d_gt| > 1 px, pixel không ước lượng được cũng tính là sai (chuẩn Middlebury). **Bad1.0_on_estimated %**: chỉ xét pixel có ước lượng. Hai cột này tách được "BM chính xác nhưng thưa" với "SGBM dày nhưng nhiễu hơn".
- **EPE**: sai số tuyệt đối trung bình của disparity (px).
- **Straightness %**: độ cong/lệch của mép trên và mép dưới panorama so với chiều cao. Càng nhỏ càng phẳng.
- **Area kept %**: phần diện tích giữ lại sau khi cắt bỏ viền đen.
- **Tiêu cự (Part C)**: lấy từ EXIF của ảnh (iPhone 12, 26 mm tương đương 35 mm) → f = 26/36 × 1200 = 866.7 px. Tiêu cự suy ra từ homography (722–878 px) chỉ để so sánh.
- **Inlier %**: tỉ lệ match qua RANSAC trên số match sau ratio test. **RMSE (px)**: sai số reprojection ‖x' − Hx‖ của các inlier.

## Ghi chú về file nộp
- Thư mục `results/` trong file nộp là bản nén nhẹ: hình lưu dạng JPG thay cho PNG, bỏ các hình trung gian (match từng cặp, panorama chưa cắt) và point cloud `bm`/`groundtruth`. Chạy `python run_all.py` (khoảng 1–2 phút) sẽ tạo lại đầy đủ.
- Notebook `TongKhanhLy_23071036_Midterm_code.ipynb` đã chạy sẵn, có đầy đủ hình, bảng và point cloud xoay được ngay trong notebook.
- Báo cáo: `../23071036_TongKhanhLy_Report.pdf`.
