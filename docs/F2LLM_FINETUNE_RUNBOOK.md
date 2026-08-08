# F2LLM-v2-80M × EnterpriseRAG-5K finetuning — full runbook

Everything needed to redo this finetuning from scratch. Written 2026-08-08 during the
second run (first instance died mid-run; rebuilt on a new box).

## 1. Goal

Finetune `codefuse-ai/F2LLM-v2-80M` (final stage-2 release checkpoint, Qwen3Model
8 layers / hidden 320 / EOS-pooled L2-normed 320-dim embeddings) on 25,000 synthetic
question→doc pairs (5 per doc) from `shanaka95/enterprise-rag-questions`, closed over
the 5,000-doc EnterpriseRAG-Bench corpus subset.

- **Training pairs**: 5 synthetic questions per corpus doc = 25,000 (query, positive).
- **Hard negatives**: 24 per query, self-mined with the *base* F2LLM against precomputed
  5K doc embeddings (the docs the base model confuses with the gold).
- **Evaluation**: the original 470 answerable EnterpriseRAG-Bench questions — **eval-only,
  never in training**.
- **Targets**: beat base hit@10 = 77.45%, hit@1 = 58.09% (n=470, exact float32 cosine over
  the 5K corpus). Stretch ≥85% (jina-embeddings-v5 reference: 88.72% @10).

## 2. Repository layout

Local working copy lives in the RAG project (`/data/projects/rag/f2llm_finetune/`);
this fork receives the synced result. Files:

| File | Purpose |
|---|---|
| `model.py` | **patched** — fp32 master weights, `F2LLM_ATTN` env switch (sdpa / flash_attention_2) |
| `utils.py` | **patched** — `enterprise` added to `RETRIEVAL_DATASETS`, grad-clip 1.0, validate() 0-dim gather fix |
| `run.py` | **patched** — deepspeed-plugin guard for single-GPU accelerate |
| `build_ft_data.py` | builds corpus.parquet + enterprise_query.parquet on the remote (deterministic, SEED=0) |
| `eval_ft_ckpt.py` | 470-question benchmark eval for any checkpoint (identical protocol to baseline) |
| `sweep_eval.py` | evaluates every step/epoch checkpoint + base, prints comparison table |
| `configs/enterprise_rag/config.json` | full-run config (bs8, ctx2048, lr1e-5, 3 epochs) |
| `configs/enterprise_rag/config_smoke.json` | 25-step smoke config |
| `configs/enterprise_rag/accelerate_config.yaml` | single-GPU, bf16 autocast, **no** deepspeed block |

Upstream files (`arguments.py`, `tokenize_data_qwen.py`, `requirements.txt`, READMEs) unchanged.
`configs/config.json` + `configs/accelerate_config.yaml` at the repo root are the
**original upstream** multi-GPU configs — do not overwrite them.

## 3. Patches vs upstream (what and why)

1. **`model.py`** — `self.dtype = torch.float32` (fp32 master weights; bf16 comes from
   accelerate autocast, keeps optimizer stable without deepspeed);
   `attn_implementation=os.environ.get('F2LLM_ATTN', 'sdpa')`;
   `trust_remote_code=True`, `use_cache=False`.
2. **`utils.py`** — `'enterprise'` appended to `RETRIEVAL_DATASETS` (enables the in-batch
   loss leg for our dataset); `accelerator.clip_grad_norm_(model.parameters(), 1.0)` before
   `optimizer.step()`; in `validate()`, `accelerator.gather(loss).float().view(-1)` on both
   loss lists — **required**: accelerate ≥1.x `gather` returns 0-dim scalars on 1 process and
   `torch.cat` crashes on them.
3. **`run.py`** — `if AcceleratorState().deepspeed_plugin is not None:` guard around the
   deepspeed-only setup so plain accelerate works.

## 4. Instance requirements

- Any NVIDIA GPU ≥16GB (run used RTX 4060 Ti then RTX A4000, both 16GB).
- Vast.ai pytorch image with `/venv/main`; torch ≥2.7 (run used 2.12.0+cu130).
- ~32GB disk: HF cache ~5GB (EnterpriseRAG-Bench documents split is the big one),
  checkpoints ~6GB (18 × ~320MB fp32), data ~1GB.
- `/workspace` persists across stop/start but **not** recycle/destroy — pull the best
  checkpoint local as soon as it's selected.

## 5. Setup on a fresh instance

```bash
# from local
SSH="ssh -p <PORT> root@<HOST>"
$SSH 'mkdir -p /workspace/rag5k/{train_data,output,cache,eval_out}'
rsync -az -e "ssh -p <PORT>" --exclude '__pycache__' \
  /data/projects/rag/f2llm_finetune/ root@<HOST>:/workspace/rag5k/f2llm_finetune/
rsync -az -e "ssh -p <PORT>" \
  /data/projects/rag/data/corpus_5k_doc_ids.json \
  /data/projects/rag/data/questions.jsonl \
  root@<HOST>:/workspace/rag5k/
rsync -az -e "ssh -p <PORT>" \
  /data/projects/rag/data/embeddings/f2llm_v2_80m_5k.npz root@<HOST>:/workspace/rag5k/

# on the instance: deps
source /venv/main/bin/activate
uv pip install transformers accelerate datasets pandas pyarrow tensorboard
```

### flash-attn (worth it: 2.6× over SDPA for this workload)

No prebuilt wheel exists for torch 2.12/cu13 — build from source (~28 min on 16 cores).
Critical env vars:

```bash
FLASH_ATTN_CUDA_ARCHS=80   # A4000(sm_86)/4060 Ti(sm_89) both run sm_80 SASS.
                           # Default "80;90;100;120" wastes ~4x compile time.
                           # TORCH_CUDA_ARCH_LIST is IGNORED by FA2 setup.py.
MAX_JOBS=4 NVCC_THREADS=2  # more than this OOM-killed nvcc on the 16-core box
FLASH_ATTENTION_FORCE_BUILD=TRUE
uv pip install flash-attn --no-build-isolation   # (use `nice -n 15 env ... & nohup`)
python3 -c "import flash_attn"                   # verify
```

If the build is impractical, set `F2LLM_ATTN=sdpa` — everything else is identical, just
~2.6× slower per step.

## 6. Build training data (deterministic)

On the instance (needs GPU + the uploads above):

```bash
cd /workspace/rag5k/f2llm_finetune && source /venv/main/bin/activate
# 1) synthetic questions (221MB; downloads 2.56M rows from HF)
python3 -c "
from datasets import load_dataset
ds = load_dataset('shanaka95/enterprise-rag-questions', split='train')
ds.to_parquet('/workspace/rag5k/enterprise_rag_questions.parquet')
print('rows:', len(ds))"                          # expect 2560220
# 2) build (downloads EnterpriseRAG-Bench documents split ~4GB cache first time)
python3 build_ft_data.py | tee /workspace/rag5k/build_data.log   # ~4 min after cache
```

Produces `train_data/corpus.parquet` (5000 tokenized docs, EOS-appended, raw content) and
`train_data/enterprise_query.parquet` (25000 rows: prompt-prepended query token ids,
positive doc-id, 24 hard-negative doc-ids).

**Verify** against `train_data/build_info.json` — with SEED=0 these are the reference
values (any deviation means something changed upstream):

```
num_pairs=25000  num_neg=24
base-model mining diagnostics: hit@1=0.21584 hit@5=0.3548 hit@10=0.41976 hit@100=0.66448
pos_rank_mean≈292.3  pos_rank_median=22
corpus_token_len_mean≈1306  corpus_token_len_max=4096
```

Conventions (match upstream `tokenize_data_qwen.py`): truncate then append EOS
(double-EOS, as the model was trained on); queries get the prompt
`Instruct: Given a question, retrieve passages that can help answer the question.\nQuery: `;
negative columns hold doc-id **strings** resolved against corpus.parquet by run.py.

## 7. Configs (`configs/enterprise_rag/`)

`config.json` — full run:

```json
{
  "model_path": "codefuse-ai/F2LLM-v2-80M",
  "experiment_id": "ft5k_lr1e-5_bs8x1_ctx2048_3ep",
  "train_data_path": "/workspace/rag5k/train_data",
  "output_dir": "/workspace/rag5k/output",
  "tb_dir": "/workspace/rag5k/output/tb",
  "cache_dir": "/workspace/rag5k/cache",
  "train_batch_size": 8,
  "checkpointing_steps": 625,
  "validation_steps": 625,
  "max_seq_length": 2048,
  "learning_rate": 1e-5,
  "min_lr": 1e-7,
  "weight_decay": 0.01,
  "warmup_steps": 200,
  "train_epochs": 3,
  "train_steps": -1,
  "log_interval": 20,
  "num_hard_neg": 7,
  "num_hard_neg_clustering": 9,
  "scheduler_type": "cosine",
  "use_mrl": false
}
```

Rationale: `max_seq_length 2048` = upstream F2LLM tokenization convention (2047+EOS);
15.4% of docs truncate (p90≈2242) — acceptable, eval still encodes at 8192.
bs8 ≈ same tokens/step as bs4/4096 but doubles in-batch negatives. `num_hard_neg: 7`
sampled per step from the 24 mined. 9282 steps total (3094/epoch).
`config_smoke.json` = same but `experiment_id "smoke"`, `output_smoke`,
`train_steps 25`, `train_epochs 1`, `warmup_steps 10`, `log_interval 5`.
`accelerate_config.yaml`: `distributed_type: NO`, `mixed_precision: bf16`,
`num_processes: 1`, no deepspeed section.

## 8. Smoke test (always run first)

```bash
cd /workspace/rag5k/f2llm_finetune && source /venv/main/bin/activate
rm -rf /workspace/rag5k/output_smoke
F2LLM_ATTN=flash_attention_2 accelerate launch \
  --config_file configs/accelerate_config.yaml run.py \
  --config configs/enterprise_rag/config_smoke.json 2>&1 | tee /workspace/rag5k/smoke.log
```

Pass criteria (observed values in parentheses):
- no OOM (VRAM ≈14.1/16GB at bs8/2048),
- `enterprise/training_loss_hard` starts ~2.7–3.0 (> ln8≈2.08 because negatives are mined)
  and trends down; `training_loss_in_batch` ≪ ln(8)≈2.08 (was ~0.5),
- epoch-end `[Validation] Step = 25` prints **without** the 0-dim torch.cat crash,
- checkpoints `step_5/` and `epoch_1/` each contain `config.json` + `model.safetensors`
  + tokenizer files (directly loadable by `AutoModel.from_pretrained`).
- Throughput reference: **FA2 ≈ 5.4 s/it** on A4000, SDPA ≈ 14 s/it; 4060 Ti SDPA was 12.9.

## 9. Full training

```bash
tmux new-session -d -s f2llm_full \
  "cd /workspace/rag5k/f2llm_finetune && source /venv/main/bin/activate && \
   F2LLM_ATTN=flash_attention_2 accelerate launch \
   --config_file configs/accelerate_config.yaml run.py \
   --config configs/enterprise_rag/config.json 2>&1 | tee /workspace/rag5k/full_train.log"
```

- 9282 steps @ ~5.4 s/it (FA2) ≈ **14h**; VRAM ~14.4GB; checkpoints every 625 steps
  (~15 step ckpts + 3 epoch ckpts, ~320MB fp32 each ≈ 6GB).
- **No resume-from-checkpoint support** in run.py — a crash means starting over.
  Monitor accordingly (below).
- Losses to watch: hard loss should grind below the 2.08 random baseline; validation
  hard loss (held-out 250 pairs, every 625 steps) must not turn upward.

### Monitoring (Claude Code session cron, every 3h)

Check: `tr "\r" "\n" < /workspace/rag5k/full_train.log | tail -4`,
`pgrep -f "config.jso[n]" | wc -l` (bracket trick — see pitfalls),
`ls /workspace/rag5k/output/ft5k_lr1e-5_bs8x1_ctx2048_3ep/`, `df -h /workspace`.
If process died: relaunch the exact tmux command above (restart-from-zero caveat).
If `epoch_3` exists: run the eval sweep (§10).

## 10. Evaluation sweep + selection

```bash
# after training finishes (GPU must be free)
tmux new-session -d -s sweep "source /venv/main/bin/activate && \
  python3 /workspace/rag5k/f2llm_finetune/sweep_eval.py --base 2>&1 | \
  tee /workspace/rag5k/sweep.log"
```

`sweep_eval.py` runs `eval_ft_ckpt.py` per checkpoint (skips existing summaries) plus
`--base` re-evaluates `codefuse-ai/F2LLM-v2-80M` under the same schema, then prints a
table + writes `/workspace/rag5k/eval_out/sweep_table.csv`. ~12 min per checkpoint
(~3.5h total). Protocol (must stay identical to baseline): docs raw @ max_length 8192,
queries prompted, bf16 encode → float32 exact cosine, full ranking, hit@K for
K∈{1,5,10,20,50,75,100,200,500,600,1000,2000}, n=470 answerable, per-type breakdown.
Select best by hit@10. Baseline: MRR=0.6499, hit@1=58.09%, hit@10=77.45%, hit@100≈90%.

## 11. Export best checkpoint

```bash
# remote -> local (best ckpt dir name from the sweep table)
rsync -az -e "ssh -p <PORT>" \
  root@<HOST>:/workspace/rag5k/output/ft5k_lr1e-5_bs8x1_ctx2048_3ep/<BEST>/ \
  /data/projects/rag/data/models/f2llm_v2_80m_ft_enterprise/
```

Then write `data/exp_f2llm_ft_5k/REPORT.md` (base vs finetuned vs jina-v5, per-type,
chosen-ckpt rationale), update `docs/DATA_MAP.md` and Claude memory.

## 12. Known pitfalls (all cost real time — read before redoing)

1. **validate() 0-dim crash** — `accelerator.gather` returns 0-dim scalars on
   single-process; `torch.cat` fails. Fixed via `.view(-1)` in utils.py (patch above).
   Crashes at the first validation (step 625) if unpatched.
2. **flash-attn build OOM** — default ninja `-j 8` × `--threads 4` gets OOM-killed on
   16-core/limited-RAM containers (`Killed` + "ninja: build stopped"). Use
   `MAX_JOBS=4 NVCC_THREADS=2`.
3. **flash-attn arch selection** — `TORCH_CUDA_ARCH_LIST` is ignored; use
   `FLASH_ATTN_CUDA_ARCHS=80` (default builds 4 archs ≈ 4× compile time).
4. **`pkill -f` self-match over ssh** — if the pattern string appears in your own ssh
   command, pkill kills your own remote shell (connection drops, exit 255). Use the
   bracket trick: `pkill -f "config.jso[n]"`.
5. **ssh with `nohup ... &`** — ssh can hang on the open channel; add
   `< /dev/null` or launch inside `tmux new-session -d`.
6. **`python` not on PATH** in non-interactive vast.ai shells — use
   `source /venv/main/bin/activate` or `python3`.
7. **TensorBoard `lr` shows 0.0** — 4-decimal display rounding of ~1e-6 warmup values,
   not a bug.
8. **Huge HF downloads** — first `load_dataset` of EnterpriseRAG-Bench documents split
   (~4GB) is slow unauthenticated; an `HF_TOKEN` env var speeds it up.
9. **Don't run two flash-attn builds** (or two of anything) — duplicate launches
   happened twice in this project; each doubled wall-time or caused OOM.

## 13. Run history

| Date | Instance | Outcome |
|---|---|---|
| 2026-08-07/08 | vast.ai RTX 4060 Ti, SDPA | data built, smoke fixed+passed, full run died at step 40/9282 when instance vanished |
| 2026-08-08 | vast.ai RTX A4000 (202.122.49.242:62236), FA2 | data rebuilt identical (build_info matches), FA2 5.37 s/it vs SDPA 14.0 s/it, full run launched 12:15 UTC, ETA ≈ 2026-08-09 02:00 UTC |

Results table — fill in after the sweep:

| ckpt | hit@1 | hit@5 | hit@10 | hit@20 | hit@100 | mrr |
|---|---|---|---|---|---|---|
| base | 58.09 | 71.70 | 77.45 | — | ~90 | 64.99 |
| best ft | | | | | | |
