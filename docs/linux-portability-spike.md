# G0-spike-1 — Tính khả thi trên Linux / Google Colab

Ngày: **2026-10-17**
Máy kiểm: Windows + WSL Ubuntu 22.04 (kernel 4.4.0-19041-Microsoft), x86_64, `mono 6.8.0.105`.

## Kết luận một dòng

| Câu hỏi | Kết quả |
|---|---|
| Chương trình `.tkv` đã biên dịch **chạy được** trên Linux? | ✅ **CÓ** |
| Compiler **khởi động** được trên Linux? | ⚠️ `--help` chạy được |
| Compiler **biên dịch được** trên Linux? | ❌ **KHÔNG** |

---

## 1. Đã xác minh chạy được (đường khả thi cho Colab)

Chương trình biên dịch trên Windows rồi chạy trên Ubuntu dưới mono → chạy đúng:

```
# Windows
tkvc spike_hello.tkv spike_hello.exe main   -> da dich
./spike_hello.exe                         -> TokenVector chay tren Linux - OK

# WSL / Ubuntu
mono /mnt/c/.../spike_hello.exe           -> TokenVector chay tren Linux - OK
```

**Kết luận:** đường "biên dịch ở Windows → chạy ở Linux bằng mono" đã thông.

Hệ quả cho kế hoạch: Colab chỉ cần `mono`, không cần compiler. Nhưng mỗi lần
sửa thuật toán vẫn phải biên dịch lại ở máy Windows rồi mới upload lên Colab.
Chấp nhận được với chu kỳ phát triển kiểu roadmap này.

## 2. Trở ngại 1 — IL của chính compiler không được mono JIT chấp nhận

```
$ mono tkvc-tv.exe --help
dung: tkvc <nguon.tkv> <ra.exe> [ten_ham_vao] [--verify-all] | tkvc @ds.txt     <- chay duoc

$ mono tkvc-tv.exe spike_hello.tkv /tmp/out.exe main
[tkvc] loi: Invalid IL code in TKVApp:_parse_program_ast (string,int,int):
           IL_080e: stloc.s   63
```

Đây **không** phải máy chạy thiếu thư viện — mà là IL mà compiler sinh ra cho
chính nó không hợp lệ theo tiêu chuẩn mà mono 6.8 dùng (JIT của .NET Framework
chấp nhận, mono thì không).

Đã thử `MONO_VERIFIER_MODE=skip` và `=none`: **không thay đổi** → không phải máy
chỉ siêng kiểm tra, mà là IL thực sự sai.

> `stloc.s 63` nghĩa là ghi vào biến cục bộ số 63. Nếu `.locals` khai báo ít hơn
> 64 biến thì đây là chỉ số vượt biên — .NET Framework có vẻ chấp nhận, mono thì
> từ chối. Cần người viết compiler xác nhận.

**Đề nghị gửi compiler team:** sửa để `_parse_program_ast` sinh IL đúng `.locals`,
hoặc báo rõ các method nào chỉ chạy được trên .NET Framework.

## 3. Trở ngại 2 — chưa có assembler trên Linux

Compiler gọi `ilasm.exe` (Windows). Trên Ubuntu đã cài `mono-runtime` nhưng
**không có `ilasm`** (`mono-ilasm`/`ilasm` đều không tồn tại).

Cần `apt-get install mono-complete` (có `ilasm`) hoặc .NET SDK.
Chưa thử vì môi trường hiện tại thiếu mạng/cài đặt — **để mở, xem mục 5**.

## 4. Câu hỏi còn mở liên quan tới GPU

Chưa kiểm chứng được (cần môi trường GPU thật):

- Mono trên Colab có gọi được CUDA không: `DllImport("cudart")` trong chương
  trình chạy dưới mono có phân giải được `libcudart.so` không.
- Nếu được, một `.tkv` train được có thể đẩy phần nặng sang GPU mà phần điều
  phối vẫn chạy trên mono. Đây là tiền đề cho kế hoạch GPU của G5/G6.

## 5. Việc cần làm tiếp

| # | Việc | Chặn gì |
|---|---|---|
| 1 | Cài `mono-complete` trong Ubuntu rồi thử biên dịch thật | xem có qua được trở ngại 2 không |
| 2 | Sửa IL `_parse_program_ast` (compiler team) | trở ngại 1 |
| 3 | Trên Colab: `apt-get install mono-complete`, chạy thử 1 `.tkv` đã biên dịch | xác nhận đường chạy thật |
| 4 | Kiểm tra `DllImport` CUDA dưới mono | mở hay không mở đường GPU |

## 6. Đánh giá rủi ro

Trở ngại 1 là rủi ro **cao** vì nó nằm ở compiler chứ không phải ở repo này và
không tự sửa được từ phía inference. Nhưng nó **không chặn roadmap**: chu kỳ
phát triển vẫn biên dịch trên Windows, và Colab chỉ giữ vai trò nơi chạy thử
khi đã có sẵn file `.exe`.
