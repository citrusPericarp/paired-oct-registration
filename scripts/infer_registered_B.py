"""Frozen v10 inference on registered B; never compares transformed priors as truth."""
from pathlib import Path
import argparse,csv,json,os,subprocess,sys,shutil
import numpy as np
from PIL import Image

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--release-root",type=Path,required=True,help="Retinal segmentation v10 release with source and checkpoints")
    parser.add_argument("--output",type=Path,required=True,help="Independent inference records directory")
    parser.add_argument("--sample-id",action="append",default=[])
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    rows=list(csv.DictReader((args.dataset/"manifests/pairs.csv").open(encoding="utf-8-sig")))
    if args.sample_id:
        selected=set(args.sample_id);rows=[r for r in rows if r["sample_id"] in selected]
    if not rows: parser.error("No selected rows")
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=args.output/"registered_B_inputs.csv"
    with manifest.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=["sample_id","side","image_path","input_role"])
        w.writeheader()
        for row in rows:
            w.writerow(dict(sample_id=row["sample_id"],side="clean",image_path=str((args.dataset/row["aligned_B"]).resolve()),input_role="registered_B"))
    release=args.release_root
    cmd=[sys.executable,str(release/"source/scripts/infer_structured_release.py"),"--checkpoint",str(release/"checkpoints/best_reviewed_terminal_trajectory.pt"),"--decoder-config",str(release/"source/configs/decoder_v10_terminal_connected_candidate_20260910.json"),"--manifest",str(manifest),"--out-dir",str(args.output),"--device",args.device,"--batch-size",str(args.batch_size)]
    env=os.environ.copy();env["OMP_NUM_THREADS"]="1";env["MKL_NUM_THREADS"]="1"
    subprocess.run(cmd,check=True,env=env)
    predictions=list(csv.DictReader((args.output/"predictions.csv").open(encoding="utf-8-sig")))
    mapping={row["sample_id"]:row for row in rows}
    qc=[]
    for pred in predictions:
        row=mapping[pred["sample_id"]]
        mask=np.asarray(Image.open(pred["mask_path"]))>0
        assert mask.shape==(496,768)
        with np.load(pred["boundary_path"],allow_pickle=False) as f:
            upper,lower,valid=f["upper_y"],f["lower_y"],f["valid_columns"].astype(bool)
            assert upper.shape==lower.shape==valid.shape==(768,)
            assert np.isfinite(upper).all() and np.isfinite(lower).all()
            assert np.all(lower[valid]>=upper[valid])
        # v10 outputs ordered bands; count disjoint occupied runs per column.
        starts=(mask & ~np.vstack([np.zeros((1,768),bool),mask[:-1]])).sum(axis=0)
        assert np.all(starts<=1)
        for key,source in [("resegmented_mask_B",pred["mask_path"]),("resegmented_boundary_B",pred["boundary_path"])]:
            relative=row.get(key)
            if not relative:
                relative=(Path(row["aligned_B"]).parent/(key+(".png" if key.endswith("mask_B") else ".npz"))).as_posix()
            destination=args.dataset/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_name(destination.name+".tmp")
            shutil.copy2(source,temporary)
            temporary.replace(destination)
            view=args.dataset/relative.replace("all_dataset/",row["label"]+"_dataset/",1)
            view.parent.mkdir(parents=True,exist_ok=True)
            temporary_view=view.with_name(view.name+".tmp")
            if temporary_view.exists():temporary_view.unlink()
            os.link(destination,temporary_view)
            temporary_view.replace(view)
        pred["input_role"]="registered_B";pred["data_type"]="registered_B"
        qc.append(dict(sample_id=row["sample_id"],status="ok",valid_columns=int(valid.sum()),mask_pixels=int(mask.sum()),band_contiguity_passed=True,input_role="registered_B"))
    with (args.output/"predictions.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
    provenance=json.loads((args.output/"inference_provenance.json").read_text(encoding="utf-8"))
    provenance.update(input_role="registered_B",domain_mapping={"clean":"registered_B"},domain_counts={"registered_B":len(predictions)},structural_prior_is_truth=False,registration_feedback=False)
    (args.output/"inference_provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    (args.output/"segmentation_qc.json").write_text(json.dumps(dict(status="complete",count=len(qc),records=qc),indent=2),encoding="utf-8")
if __name__=="__main__":main()
