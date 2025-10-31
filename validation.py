import nemo.collections.asr as nemo_asr
import lightning.pytorch as pl
import torch
import copy
torch.set_float32_matmul_precision("high")
from omegaconf import OmegaConf, open_dict

# ✅ 학습된 모델 경로 (파일명을 본인이 저장한 경로로 수정)
model_path = "/workspace/facl/experiments/nemo_fairaudio/simclr_supconGRL_1e-1_diffspace/2025-02-19_21-43-27/checkpoints/simclr_supconGRL_1e-1_diffspace.nemo"

# ✅ 여러 개의 Validation Manifest 파일 리스트
val_manifests = [
    "/workspace/facl/metadata/age/test_manifest_18.json",
    "/workspace/facl/metadata/age/test_manifest_23.json",
    "/workspace/facl/metadata/age/test_manifest_31.json",
    "/workspace/facl/metadata/age/test_manifest_46.json",
    "/workspace/facl/metadata/ethnicity/test_manifest_Asian.json",
    "/workspace/facl/metadata/ethnicity/test_manifest_Black.json",
    "/workspace/facl/metadata/ethnicity/test_manifest_Hispanic.json",
    "/workspace/facl/metadata/ethnicity/test_manifest_Middle.json",
    "/workspace/facl/metadata/ethnicity/test_manifest_NativeA.json",
    "/workspace/facl/metadata/ethnicity/test_manifest_NativeH.json",
    "/workspace/facl/metadata/ethnicity/test_manifest_White.json",
    "/workspace/facl/metadata/first_language/test_manifest_english.json",
    "/workspace/facl/metadata/first_language/test_manifest_non_english.json",
    "/workspace/facl/metadata/gender/test_manifest_female.json",
    "/workspace/facl/metadata/gender/test_manifest_male.json",
    "/workspace/facl/metadata/socioeconomic_bkgd/test_manifest_Affluent.json",
    "/workspace/facl/metadata/socioeconomic_bkgd/test_manifest_Low.json",
    "/workspace/facl/metadata/socioeconomic_bkgd/test_manifest_Medium.json",
]
wer_results=[]

# ✅ 학습된 모델 불러오기
asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(model_path)
cfg = copy.deepcopy(asr_model.cfg)
with open_dict(cfg):
    cfg.validation_ds = OmegaConf.create({})  
    cfg.validation_ds.sample_rate = 16000  
    cfg.validation_ds.labels = cfg.labels
    cfg.validation_ds.batch_size =64
    cfg.validation_ds.num_workers = 8
    cfg.validation_ds.pin_memory = True
    cfg.validation_ds.trim_silence = True
    cfg.validation_ds.shuffle = False

trainer = pl.Trainer(
    devices=1,
    accelerator="gpu",
    precision=16,
    logger=False
)

# ✅ 여러 개의 Validation Set을 사용하여 성능 평가 수행
for val_manifest in val_manifests:
    print(f"\n🔥 Evaluating on: {val_manifest}")
    with open_dict(cfg):
        cfg.validation_ds.manifest_filepath = val_manifest
    
    # ✅ 개별 Validation 파일을 적용
    asr_model.setup_validation_data(val_data_config=cfg.validation_ds)
    
    # ✅ Validation 수행 (WER 계산)
    results = trainer.validate(asr_model)
    
    # ✅ WER 값 출력
    wer = results[0]["val_wer"]
    print(f"✅ WER on {val_manifest}: {wer:.4f}\n")
    wer_results.append(wer)
    
for ds, wer in zip(val_manifests,wer_results):
    print(f"WER on {ds}: {wer:.4f}")