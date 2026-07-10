#!/usr/bin/env python3
import os
import sys
import json
import re

# Sorting key for chip generations
def chip_sort_key(slug):
    slug_lower = slug.lower()
    if 'm1' in slug_lower:
        val = 1
    elif 'm2' in slug_lower:
        val = 2
    elif 'm3' in slug_lower:
        val = 3
    elif 'm4' in slug_lower:
        val = 4
    else:
        val = 9
    
    if 'pro' in slug_lower:
        val += 0.1
    elif 'max' in slug_lower:
        val += 0.2
    elif 'ultra' in slug_lower:
        val += 0.3
    return val

def format_latency(val):
    if val is None or val == 0:
        return "TBD"
    return f"{val:.2f} ms"

def format_speedup(val):
    if val is None or val == 0:
        return "TBD"
    return f"{val:.2f}\\times"

def format_mean_std(mean, std):
    if mean is None or mean == 0:
        return "TBD"
    return f"${mean:.2f} \\pm {std:.2f}$"

def main():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    print(f"Scanning for results in: {results_dir}")
    
    chips_found = []

    # Scan directory
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.') and item != 'reproducibility' and item != 'ablation':
                chips_found.append(item)

    # Dedup: legacy short-form slugs (e.g. "Apple_M1", from before the hw_slug format
    # included memory/core/ANE counts) refer to the same physical machine as a longer
    # slug that starts with the same prefix (e.g. "Apple_M1_8GB_8CPU_7GPU_16ANE"). Keep
    # only the long-form slug so the same machine doesn't appear as two chip rows.
    long_slugs = [c for c in chips_found if any(
        c != other and c.startswith(other) for other in chips_found
    )]
    superseded = {other for c in long_slugs for other in chips_found
                  if c != other and c.startswith(other)}
    chips_found = [c for c in chips_found if c not in superseded]

    chips_found.sort(key=chip_sort_key)
    print(f"Found chip directories: {chips_found}")
    
    # Datastores
    swift_data = {}
    python_data = {}
    mlx_fp16_data = {}
    training_fp16_data = {}
    prefill_data = {}
    
    # Helper to clean up chip name display.
    # New slugs: "Apple_M1_8GB_8CPU_7GPU_16ANE" → "Apple M1 8GB"
    # Old slugs: "Apple_M1" → "Apple M1"
    def clean_chip_name(slug):
        parts = slug.replace("_", " ").split()
        result = []
        for p in parts:
            result.append(p)
            if p.endswith("GB") or p.endswith("TB"):
                break
        return " ".join(result)

    # Load all data
    for chip in chips_found:
        chip_dir = os.path.join(results_dir, chip)
        
        # Load swift results
        swift_path = os.path.join(chip_dir, "swift_benchmark.json")
        if os.path.exists(swift_path):
            try:
                with open(swift_path, 'r') as f:
                    swift_data[chip] = json.load(f)
                print(f"  ✓ Loaded Swift results for {chip}")
            except Exception as e:
                print(f"  ⚠️ Error loading Swift results for {chip}: {e}")
                
        # Load python results
        python_path = os.path.join(chip_dir, "model_comparison.json")
        if os.path.exists(python_path):
            try:
                with open(python_path, 'r') as f:
                    python_data[chip] = json.load(f)
                print(f"  ✓ Loaded Python model comparison for {chip}")
            except Exception as e:
                print(f"  ⚠️ Error loading Python results for {chip}: {e}")

        # Load MLX-FP16 baseline (the fair, precision-matched comparison --
        # MLX-FP32 alone overstates FusionML's advantage since FusionML runs FP16)
        mlx_fp16_path = os.path.join(chip_dir, "mlx_fp16_baseline_test.json")
        if os.path.exists(mlx_fp16_path):
            try:
                with open(mlx_fp16_path, 'r') as f:
                    mlx_fp16_data[chip] = json.load(f)
                print(f"  ✓ Loaded MLX-FP16 baseline for {chip}")
            except Exception as e:
                print(f"  ⚠️ Error loading MLX-FP16 baseline for {chip}: {e}")

        # Load MLX-FP16 TRAINING baseline (same-session, thermal-matched
        # FusionML re-run included in the file)
        training_fp16_path = os.path.join(chip_dir, "mlx_fp16_training_baseline.json")
        if os.path.exists(training_fp16_path):
            try:
                with open(training_fp16_path, 'r') as f:
                    training_fp16_data[chip] = json.load(f)
                print(f"  ✓ Loaded MLX-FP16 training baseline for {chip}")
            except Exception as e:
                print(f"  ⚠️ Error loading MLX-FP16 training baseline for {chip}: {e}")

        # Load prefill-scale benchmark (per-layer eager-boundary stream split)
        prefill_path = os.path.join(chip_dir, "prefill_scale_benchmark.json")
        if os.path.exists(prefill_path):
            try:
                with open(prefill_path, 'r') as f:
                    prefill_data[chip] = json.load(f)
                print(f"  ✓ Loaded prefill-scale benchmark for {chip}")
            except Exception as e:
                print(f"  ⚠️ Error loading prefill-scale benchmark for {chip}: {e}")

    # Ensure we always have at least M1 as a baseline or show template
    if not swift_data and not python_data:
        print("❌ No benchmark results found! Make sure to run benchmarks first.")
        sys.exit(1)

    valid_chips = [c for c in chips_found if c in swift_data or c in python_data]
    if not valid_chips:
        valid_chips = ["Apple_M1"]

    # -------------------------------------------------------------------------
    # 1. Generate Swift LaTeX Table
    # -------------------------------------------------------------------------
    swift_workloads = [
        ("Llama-3-8B Inference", "llama_inference"),
        ("Llama-3-8B Training", "llama_training"),
        ("GPT-2 XL Inference", "gpt2_inference"),
        ("GPT-2 XL Training", "gpt2_training"),
        ("MLP Inference", "mlp_inference"),
        ("MLP Training", "mlp_training")
    ]
    
    swift_latex = []
    swift_latex.append(r"\begin{tabular}{llccccr}")
    swift_latex.append(r"\toprule")
    swift_latex.append(r"Model Block & Chip & CPU-Only & GPU-Only & Smart Split & Winner & Speedup \\")
    swift_latex.append(r"\midrule")
    
    for i, (w_name, w_key) in enumerate(swift_workloads):
        first_row = True
        swift_latex.append(f"\\textbf{{{w_name}}} & & & & & & \\\\")
        
        # We populate for all found chips
        for chip in valid_chips:
            chip_name = clean_chip_name(chip)
            
            cpu_val = None
            gpu_val = None
            smart_val = None
            speedup = None
            winner = "TBD"
            
            if chip in swift_data:
                results = swift_data[chip].get("results", {})
                w_data = results.get(w_key, {})
                cpu_val = w_data.get("cpu_ms")
                gpu_val = w_data.get("gpu_ms")
                smart_val = w_data.get("smart_ms")
                speedup = w_data.get("speedup")
                
                if smart_val is not None and gpu_val is not None and cpu_val is not None:
                    if smart_val < gpu_val and smart_val < cpu_val:
                        winner = "Smart Split"
                    elif gpu_val < cpu_val:
                        winner = "GPU-Only"
                    else:
                        winner = "CPU-Only"
            
            cpu_str = format_latency(cpu_val)
            gpu_str = format_latency(gpu_val)
            
            if smart_val is not None:
                smart_str = f"\\textbf{{{smart_val:.2f} ms}}"
            else:
                smart_str = "TBD"
                
            speedup_str = format_speedup(speedup)
            
            swift_latex.append(f" & {chip_name} & {cpu_str} & {gpu_str} & {smart_str} & {winner} & {speedup_str} \\\\")
            
        if i < len(swift_workloads) - 1:
            swift_latex.append(r"\midrule")
            
    swift_latex.append(r"\bottomrule")
    swift_latex.append(r"\end{tabular}")
    swift_latex_str = "\n".join(swift_latex)

    # -------------------------------------------------------------------------
    # 2. Generate Python LaTeX Table
    # -------------------------------------------------------------------------
    # w_key: model_comparison.json workload key.
    # fp16_key: (model, dtype-)key into mlx_fp16_baseline_test.json, or None where
    # no precision-matched baseline has been measured yet (training, MLP).
    python_workloads = [
        ("Llama-3-8B Inference", "Llama-3-8B Inference", "llama"),
        ("Llama-3-8B Training", "Llama-3-8B Training", None),
        ("GPT-2 XL Inference", "GPT-2 XL Inference", "gpt2"),
        ("GPT-2 XL Training", "GPT-2 XL Training", None),
        ("Deep MLP Inference", "MLP Inference", None),
        ("Deep MLP Training", "MLP Training", None)
    ]
    
    # seq_len for tokens/sec computation (same for all workloads: 1024)
    _SEQ_LEN = 1024

    import math as _math

    def format_ci(mean, ci95):
        if mean is None or ci95 is None:
            return "TBD"
        return f"${mean:.2f} \\pm {ci95:.2f}$"

    def format_mem(val):
        if val is None or val == 0:
            return "---"
        return f"{val:.0f}"

    python_latex = []
    python_latex.append(r"\begin{tabular}{llccccrrr}")
    python_latex.append(r"\toprule")
    python_latex.append(
        r"Model Block / Mode & Chip & MLX (FP32) & MLX (FP16) & PyTorch (MPS) & \textbf{FusionML (Ours)} "
        r"& Speedup vs.\ MLX-FP16 & FusionML (tok/s) & Mem (MB) \\"
    )
    python_latex.append(r"\midrule")

    gpt2_inf_note = False
    gpt2_inf_note_vals = None
    fp16_slower_note = False
    fp16_slower_vals = []

    for i, (w_name, w_key, fp16_key) in enumerate(python_workloads):
        python_latex.append(f"\\textbf{{{w_name}}} & & & & & & & & \\\\")

        for chip in valid_chips:
            chip_name = clean_chip_name(chip)

            mlx_mean = mlx_std = mlx_med = mlx_ci95 = None
            pt_mean  = pt_std  = pt_med  = pt_ci95  = None
            fml_mean = fml_std = fml_med = fml_ci95 = None
            fml_mem  = None
            speedup_fp32 = None
            mlx_fp16_med = mlx_fp16_ci95 = None
            speedup_fp16 = None

            if chip in python_data:
                results = python_data[chip].get("results", {})
                w_data  = results.get(w_key, {})

                mlx     = w_data.get("MLX", {})
                mlx_mean, mlx_std, mlx_med = mlx.get("mean"), mlx.get("std"), mlx.get("median")
                n_mlx   = mlx.get("n_runs", 20)
                mlx_ci95 = mlx.get("ci95") or (
                    1.96 * mlx_std / _math.sqrt(n_mlx) if mlx_std else None
                )

                pt      = w_data.get("PyTorch (MPS)", {})
                pt_mean, pt_std, pt_med = pt.get("mean"), pt.get("std"), pt.get("median")
                n_pt    = pt.get("n_runs", 20)
                pt_ci95 = pt.get("ci95") or (
                    1.96 * pt_std / _math.sqrt(n_pt) if pt_std else None
                )

                fml     = w_data.get("FusionML", {})
                fml_mean, fml_std, fml_med = fml.get("mean"), fml.get("std"), fml.get("median")
                n_fml   = fml.get("n_runs", 20)
                fml_ci95 = fml.get("ci95") or (
                    1.96 * fml_std / _math.sqrt(n_fml) if fml_std else None
                )
                fml_tps  = fml.get("tokens_per_sec") or (
                    _SEQ_LEN * 1000.0 / fml_mean if fml_mean and fml_mean > 0 else None
                )
                fml_mem  = fml.get("peak_mem_mb")

                if fml_med is not None and mlx_med is not None and fml_med > 0:
                    speedup_fp32 = mlx_med / fml_med

            # Fair, precision-matched comparison: MLX run natively in FP16, not FP32.
            # This is the number that should drive any "FusionML beats MLX" claim.
            if fp16_key and chip in mlx_fp16_data:
                fp16_res = mlx_fp16_data[chip].get("results", {}).get(fp16_key, {}).get("fp16", {})
                mlx_fp16_med  = fp16_res.get("median")
                mlx_fp16_ci95 = fp16_res.get("ci95")
                if mlx_fp16_med is not None and fml_med is not None and fml_med > 0:
                    speedup_fp16 = mlx_fp16_med / fml_med
                    if speedup_fp16 < 1.0:
                        fp16_slower_note = True
                        fp16_slower_vals.append((w_name, fml_med, mlx_fp16_med, speedup_fp16))

            # Training rows: precision-matched baseline lives in
            # mlx_fp16_training_baseline.json. Its speedup is computed against
            # the same-session (thermal-matched) FusionML re-run in that file,
            # not the model_comparison.json number from a different session.
            elif "Training" in w_key and chip in training_fp16_data:
                model_key = "llama" if "Llama" in w_name else ("gpt2" if "GPT-2" in w_name else None)
                tr = training_fp16_data[chip].get("results", {}).get(model_key, {}) if model_key else {}
                fp16_res = tr.get("mlx_fp16") or {}
                fml_fresh = tr.get("fusionml") or {}
                mlx_fp16_med  = fp16_res.get("median")
                mlx_fp16_ci95 = fp16_res.get("ci95")
                fml_fresh_med = fml_fresh.get("median")
                if mlx_fp16_med is not None and fml_fresh_med:
                    speedup_fp16 = mlx_fp16_med / fml_fresh_med
                    if speedup_fp16 < 1.0:
                        fp16_slower_note = True
                        fp16_slower_vals.append((w_name, fml_fresh_med, mlx_fp16_med, speedup_fp16))

            mlx_str  = format_ci(mlx_mean, mlx_ci95)
            mlx_fp16_str = format_ci(mlx_fp16_med, mlx_fp16_ci95) if mlx_fp16_med is not None else "not measured"
            pt_str   = format_ci(pt_mean,  pt_ci95)
            fml_str  = (
                f"$\\mathbf{{{fml_mean:.2f} \\pm {fml_ci95:.2f}}}$"
                if fml_mean is not None and fml_ci95 is not None
                else ("TBD" if fml_mean is None else
                      f"$\\mathbf{{{fml_mean:.2f} \\pm {fml_std:.2f}}}$")
            )

            # Report the FP16-vs-FP16 speedup where measured; fall back to the
            # (weaker, precision-mismatched) FP32 comparison only where FP16
            # hasn't been benchmarked yet, and mark it as such.
            if speedup_fp16 is not None:
                speedup_str = format_speedup(speedup_fp16)
            elif speedup_fp32 is not None:
                # \textsuperscript is text-mode safe, unlike nesting another $...$
                # math toggle inside the unwrapped \mathbf{...} cell used below.
                speedup_str = format_speedup(speedup_fp32) + r"\textsuperscript{\ddagger}"
            else:
                speedup_str = "TBD"

            tps_str = (
                f"{fml_tps:,.0f}".replace(",", "{,}") if fml_tps is not None else "TBD"
            )
            mem_str = format_mem(fml_mem)

            row_note = ""
            if w_key == "GPT-2 XL Inference" and fml_med is not None and pt_med is not None and fml_med > pt_med:
                row_note = r"$^{\dagger}$"
                gpt2_inf_note = True
                if gpt2_inf_note_vals is None:
                    gpt2_inf_note_vals = (fml_med, pt_med)

            python_latex.append(
                f" & {chip_name} & {mlx_str} & {mlx_fp16_str} & {pt_str} & {fml_str}{row_note}"
                f" & \\mathbf{{{speedup_str}}} & {tps_str} & {mem_str} \\\\"
            )

        if i < len(python_workloads) - 1:
            python_latex.append(r"\midrule")

    python_latex.append(r"\bottomrule")
    footnotes = []
    if gpt2_inf_note and gpt2_inf_note_vals:
        fml_v, pt_v = gpt2_inf_note_vals
        pct = (fml_v / pt_v - 1.0) * 100.0
        footnotes.append(
            r"\footnotesize $^{\dagger}$FusionML is "
            f"{pct:.1f}\\% slower than PyTorch MPS on GPT-2 XL inference "
            f"({fml_v:.1f}\\,ms vs.\\ {pt_v:.1f}\\,ms); speedup is measured against MLX baseline."
        )
    footnotes.append(
        r"\footnotesize $^{\ddagger}$No precision-matched MLX-FP16 baseline measured for this "
        r"row yet; speedup shown is vs.\ MLX-FP32, which overstates FusionML's advantage since "
        r"FusionML computes in FP16. Run \texttt{mlx\_fp16\_baseline\_test.py} for this workload "
        r"before citing this number."
    )
    if fp16_slower_note:
        detail = "; ".join(
            f"{name}: FusionML {fml:.1f}\\,ms vs.\\ MLX-FP16 {mlx16:.1f}\\,ms ({sp:.2f}\\times)"
            for name, fml, mlx16, sp in fp16_slower_vals
        )
        footnotes.append(
            r"\footnotesize \textbf{Caution:} FusionML is \emph{slower} than MLX run natively "
            f"in FP16 for: {detail}. The precision-matched comparison does not currently "
            r"support a ``FusionML beats MLX'' claim for these workloads."
        )
    for fn in footnotes:
        python_latex.append(r"\multicolumn{9}{l}{" + fn + r"} \\")
    python_latex.append(r"\end{tabular}")
    python_latex_str = "\n".join(python_latex)

    # -------------------------------------------------------------------------
    # 2b. Prefill-scale table: per-layer eager-boundary stream split vs MLX-FP16
    # -------------------------------------------------------------------------
    prefill_latex = []
    prefill_latex.append(r"\begin{tabular}{llcccrr}")
    prefill_latex.append(r"\toprule")
    prefill_latex.append(
        r"Model Block & Seq.\ Len.\ & MLX-FP16 & FusionML (no split) & \textbf{FusionML (split)} "
        r"& Speedup & Max.\ Rel.\ Err.\ \\"
    )
    prefill_latex.append(r"\midrule")
    for chip in valid_chips:
        if chip not in prefill_data:
            continue
        pres = prefill_data[chip].get("results", {})
        for model_key, model_name in [("llama", "Llama-3-8B"), ("gpt2", "GPT-2 XL")]:
            seqs = pres.get(model_key, {})
            for L in sorted(seqs, key=int):
                arms = seqs[L]
                mlx16 = arms.get("mlx_fp16") or {}
                nosp  = arms.get("fusion_nosplit") or {}
                spl   = arms.get("fusion_split") or {}
                mlx16_str = format_ci(mlx16.get("mean"), mlx16.get("ci95"))
                nosp_str  = format_ci(nosp.get("mean"), nosp.get("ci95"))
                spl_str   = (
                    f"$\\mathbf{{{spl['mean']:.2f} \\pm {spl['ci95']:.2f}}}$"
                    if spl.get("mean") is not None else "TBD"
                )
                sp = (mlx16.get("median") / spl.get("median")
                      if mlx16.get("median") and spl.get("median") else None)
                sp_str = format_speedup(sp)
                err = spl.get("max_rel_err_vs_gpu_only")
                if err is not None:
                    mant, expo = f"{err:.1e}".split("e")
                    err_str = f"${mant} \\times 10^{{{int(expo)}}}$"
                else:
                    err_str = "---"
                prefill_latex.append(
                    f"{model_name} & {L} & {mlx16_str} & {nosp_str} & {spl_str} & {sp_str} & {err_str} \\\\"
                )
            prefill_latex.append(r"\midrule")
    if prefill_latex[-1] == r"\midrule":
        prefill_latex.pop()
    prefill_latex.append(r"\bottomrule")
    prefill_latex.append(
        r"\multicolumn{7}{l}{\footnotesize Correctness: max relative error of the split forward "
        r"vs.\ the GPU-only forward of the same FP16 block (FP16 noise level).} \\"
    )
    prefill_latex.append(r"\end{tabular}")
    prefill_latex_str = "\n".join(prefill_latex)

    # -------------------------------------------------------------------------
    # Print out results for verification
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("GENERATED SWIFT SMART SPLIT LATENCY TABLE (LaTeX)")
    print("="*80)
    print(swift_latex_str)
    
    print("\n" + "="*80)
    print("GENERATED PYTHON-TO-PYTHON COMPARISON TABLE (LaTeX)")
    print("="*80)
    print(python_latex_str)

    print("\n" + "="*80)
    print("GENERATED PREFILL-SCALE SPLIT TABLE (LaTeX)")
    print("="*80)
    print(prefill_latex_str)
    
    # -------------------------------------------------------------------------
    # 3. Update the LaTeX file
    # -------------------------------------------------------------------------
    tex_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../fusionml_neural_scheduling_neurips2026.tex"))
    if os.path.exists(tex_path):
        print(f"\nUpdating LaTeX paper at: {tex_path}")
        with open(tex_path, 'r') as f:
            tex_content = f.read()
            
        # Update Table 4 (Swift smart split benchmarks)
        # Search for \label{tab:native_swift_benchmarks} and replace the next \begin{tabular} ... \end{tabular} block
        pattern_swift = r"(\\label\{tab:native_swift_benchmarks\}\s*\n*)(\\begin\{tabular\}.*?\\end\{tabular\})"
        match_swift = re.search(pattern_swift, tex_content, re.DOTALL)
        if match_swift:
            tex_content = tex_content.replace(match_swift.group(2), swift_latex_str)
            print("  ✓ Successfully updated Swift Smart Split table in LaTeX paper.")
        else:
            print("  ❌ Could not find native Swift table environment pattern in LaTeX paper!")
            
        # Update Table 5 (Python benchmarks)
        # Search for \label{tab:python_benchmarks} and replace the next \begin{tabular} ... \end{tabular} block
        pattern_py = r"(\\label\{tab:python_benchmarks\}\s*\n*)(\\begin\{tabular\}.*?\\end\{tabular\})"
        match_py = re.search(pattern_py, tex_content, re.DOTALL)
        if match_py:
            tex_content = tex_content.replace(match_py.group(2), python_latex_str)
            print("  ✓ Successfully updated Python comparison table in LaTeX paper.")
        else:
            print("  ❌ Could not find Python comparison table environment pattern in LaTeX paper!")
            
        # Update prefill-scale table (per-layer eager-boundary split)
        pattern_prefill = r"(\\label\{tab:prefill_scale\}\s*\n*)(\\begin\{tabular\}.*?\\end\{tabular\})"
        match_prefill = re.search(pattern_prefill, tex_content, re.DOTALL)
        if match_prefill:
            tex_content = tex_content.replace(match_prefill.group(2), prefill_latex_str)
            print("  ✓ Successfully updated prefill-scale table in LaTeX paper.")
        else:
            print("  ❌ Could not find prefill-scale table (\\label{tab:prefill_scale}) in LaTeX paper — add the table environment manually first.")

        # NOTE: a prior version of this script unconditionally overwrote the Limitations
        # paragraph with a hardcoded "0.80--0.91x of MLX" claim, regardless of what the
        # actual current data showed. That is not safe -- the correct comparison baseline
        # (MLX-FP16, not MLX-FP32) and the actual measured ratios must be computed from
        # mlx_fp16_baseline_test.json before any such claim is written. This block
        # intentionally does not auto-rewrite that paragraph; update it manually once the
        # fair-comparison numbers are confirmed for the target hardware generation.

        # Write back updated tex file
        with open(tex_path, 'w') as f:
            f.write(tex_content)
        print("💾 Updated LaTeX paper written successfully.")
    else:
        print(f"\n⚠️ LaTeX paper not found at {tex_path}")

if __name__ == "__main__":
    main()
