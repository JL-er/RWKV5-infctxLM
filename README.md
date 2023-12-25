# RWKV5-infctxLM
```
python train.py --my_testing r3r4 --load_model /home/asd/model/RWKV-5-World-1B5-v2-20231025-ctx4096.pth --proj_dir /home/asd/model --data_file ttt_text_document --data_type binidx --vocab_size 65536 --epoch_steps 1 --epoch_count 100 --epoch_begin 0 --epoch_save 5 --micro_bsz 1 --n_layer 24 --n_embd 2048 --pre_ffn 0 --head_qk 0 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1 --real_len 100 --ctx_len 200
```
ctx_len 为你想要的训练长度  4096
real_len 受显存限制为实际训练长度 1024
ttt 测试文件

# 项目说明/Project Description Translation:

The project has successfully achieved infinite-length training for rwkv5. However, due to a lack of understanding of some design aspects in bo v5backward, there are issues with gradient fallback during breakpoint backpropagation, resulting in gradient drops. Improvement is needed in the backward process.\
本项目实现了rwkv5的无限长度训练，但由于没有理解bo v5backward的一些设计所以目前断点回传梯度时fallback会出现掉点，需改进backward\
wanicca版本的https://github.com/xiaol/Train-infctx-RWKV.git  和Blealtan的https://github.com/RWKV/RWKV-infctx-trainer.git  
我只是修改了rwkv5算子部分，以及移植代码，非常感谢Blealtan老师的耐心指导和交流。
