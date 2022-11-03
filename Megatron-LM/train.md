<div align='center'><font size='20'> Train </font></div>

`megatron/training.py`

# pretrain
Interface of training
* Call `initialize_megatron` to initialize `torch.distributed.process_group` and [`mpu`](mpu.md)
* Initialize jit
* Build model and optimizer via `setup_model_and_optimizer`