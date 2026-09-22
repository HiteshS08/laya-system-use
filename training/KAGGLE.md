# Training on Kaggle (2x T4)

Needs your Kaggle account. The items contain only Mind2Web **train** data (CC-BY-4.0). Never upload `data/` or any
`test_*` file.

1. On the Mac, after Task 7 Step 1: `training/out/` holds `train_items.pt`, `train_items.meta.json`,
   `dev_items.pt`, `dev_items.meta.json`.
2. Kaggle > Datasets > New Dataset (**Private**), name `laya-mind2web-items`. Upload those four files plus
   `training/train_ddp.py`.
3. New Notebook > Settings > Accelerator **GPU T4 x2**, Internet **On**. Add the dataset. Run:

   Cell 1
   ```
   !pip install -q "laya>=0.3.4" "transformers>=4.48" safetensors huggingface_hub peft
   from huggingface_hub import snapshot_download
   from laya.agent import _fix_tokenizer_config
   model_dir = snapshot_download("convaiinnovations/laya")
   _fix_tokenizer_config(model_dir)
   ```
   Cell 2
   ```
   D = "/kaggle/input/laya-mind2web-items"
   !torchrun --standalone --nproc_per_node=2 {D}/train_ddp.py {model_dir} {D}/train_items.pt {D}/dev_items.pt /kaggle/working/laya_browser_mind2web
   ```
   The script reads `train_items.meta.json` next to `train_items.pt`, so keep both in the same folder.
4. Watch the `reward` and `loss` lines. If the session hits its limit, rerun with `EPOCHS=2` (set with
   `%env EPOCHS=2` before Cell 2). Other overridable env vars: `LR_ENCODER`, `LR_HEAD`, `SIGMA_START`,
   `SIGMA_END`, `GROUP_SIZE`, and `USE_LORA=1` (with `LORA_R`, `LORA_ALPHA`) to freeze the base encoder and
   train LoRA adapters instead — the saved checkpoint is merged back to plain dense weights either way.
5. Download `/kaggle/working/laya_browser_mind2web` (Output tab) to `checkpoints/laya_browser_mind2web` on the Mac.
   It must contain `model.safetensors`, `encoder/`, `tokenizer/`, `rl_agent_config.json`.
