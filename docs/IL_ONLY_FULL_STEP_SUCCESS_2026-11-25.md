# IL-only GPU roadmap - mot buoc train xac nhan (2026-11-25)

## Ket qua
- Kaggle CPU, .NET 10.0.404, HEAD `TokenVector.CudaBackend@7299bce`, nhap `Inference@8d68240`
- Notebook: `demo/KAGGLE_IL_ONLY_FULL_STEP_V2_2026-11-25.ipynb`
- `.tkv -> IL -> .NET PE -> dotnet`, KHONG C++/CUDA source, KHONG torch, KHONG mono
- LOSS IL 2.1086578 vs ref 2.108657656669579 -> err 1.43e-7
- W1 update maxerr 7.52e-8, W2 7.26e-8
- dotnet rc=0, stderr empty

## Kernel da xac nhan (deu MAXERR ~1e-7 hoac tot hon)
Forward: SAXPY, SGEMM, SCALE, GELU, ADD, INDEX_SELECT, LayerNorm, Softmax, SDPA, log-softmax/gather/loss (11)
Backward: dGELU, dLN, dMM, dAttn (4)
Optimizer: AdamW
Full step: 2-layer forward + backward + AdamW trong mot .tkv

## Gioi han trung thuc
- Buoc train hien chay tren HOST (T=4,D=8,V=8), chua scale va chua do toc do thuc.
- Bridge C++ sinh (`2.36x` vs torch) van la baseline duyduy nhat co ket qua train GPT2-medium.
- De thay baseline can: suce lai P/Invoke CUDA duoi .NET 10 (cuLaunchKernel crash) va do loss/toc do tren GPT2-medium.

## Bai hoc
- mono 6.8 crash voi IL lon -> chuyen .NET 10 (chay PE mscorlib cu duoc)
- .NET 10 khong tim thay cuda.dll neu runtime CPU -> bo moi P/Invoke CUDA khi khong can
- Kaggle: Save Version -> Output -> download ipynb; hoac File -> Download -> Notebook