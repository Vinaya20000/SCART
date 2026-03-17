import GEOparse
import os
import scanpy as sc
import anndata as ad
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class SampleAnnotator:

    def __init__(self, *inputs):

        self.inputs = list(inputs)
        self.base_dir = "GSE_data"

        os.makedirs(self.base_dir, exist_ok=True)

        self.gse_ids = []
        self.h5ad_inputs = []

        for item in self.inputs:
            if isinstance(item, str) and item.lower().endswith(".h5ad"):
                self.h5ad_inputs.append(item)
            else:
                self.gse_ids.append(item)


    def run(self):

        normal = []
        tumor = []
        unspecified = []
        annotation_info = {}
        cancer_type = None

        tumor_adatas = []
        results = {}

        for gse_id in self.gse_ids:

            n, t, u, ann, ct = self._process_gse(gse_id)

            normal.extend(n)
            tumor.extend(t)
            unspecified.extend(u)

            annotation_info.update(ann)

            if ct and cancer_type is None:
                cancer_type = ct

            adata = self._build_h5ad(
                gse_id,
                t,
                save_single=(len(self.gse_ids) == 1 and len(self.h5ad_inputs) == 0)
            )

            if adata is not None:
                tumor_adatas.append(adata)

            results[gse_id] = (n, t, u, ann, None, ct)

        for file in self.h5ad_inputs:

            print("\n========== Reading h5ad file ==========")

            adata = sc.read_h5ad(file)

            adata.obs_names_make_unique()

            adata.layers["counts"] = adata.X.copy()
            adata.raw = adata

            tumor_adatas.append(adata)

            results[file] = ([], [], [], {}, None, None)

        query_h5ad = None
        total_inputs = len(self.gse_ids) + len(self.h5ad_inputs)

        if total_inputs == 1:

            if len(self.gse_ids) == 1:

                query_h5ad = f"{self.gse_ids[0]}_tumor.h5ad"

                results[self.gse_ids[0]] = (
                    normal, tumor, unspecified,
                    annotation_info, query_h5ad, cancer_type
                )

            elif len(self.h5ad_inputs) == 1:

                adata = tumor_adatas[0]

                filename = "input_tumor.h5ad"
                adata.write(filename)

                print("\n========== h5ad created ==========")
                print(f"{filename} is created successfully")

                query_h5ad = filename

                key = self.h5ad_inputs[0]

                results[key] = ([], [], [], {}, query_h5ad, None)

        elif total_inputs > 1 and len(tumor_adatas) > 0:

            combined = ad.concat(tumor_adatas, join="outer")

            combined.obs_names_make_unique()
            combined.layers["counts"] = combined.X.copy()
            combined.raw = combined

            combined.write("combined_tumor.h5ad")

            print("\n========== h5ad created ==========")
            print("combined_tumor.h5ad is created successfully")

            query_h5ad = "combined_tumor.h5ad"

            for key in results:
                n, t, u, ann, _, ct = results[key]
                results[key] = (n, t, u, ann, query_h5ad, ct)

        return normal, tumor, unspecified, annotation_info, query_h5ad, cancer_type, results


    def _process_gse(self, gse_id):

        gse_dir = os.path.join(self.base_dir, gse_id)
        os.makedirs(gse_dir, exist_ok=True)

        gse = GEOparse.get_GEO(
            geo=gse_id,
            destdir=gse_dir,
            how="full"
        )

        # ✅ download supplementary safely
        try:
            gse.download_supplementary_files()
        except Exception as e:
            print(f"Warning: Supplementary download issue: {e}")

        normal, tumor, unspecified, annotation_info = [], [], [], {}

        cancer_type = self._predict_cancer_type(gse)

        tumor_keywords = ["tumor", "tumour", "cancer", "carcinoma", "adenocarcinoma", "malignant", "metastatic"]
        normal_keywords = ["normal", "healthy", "control", "adjacent normal"]

        for gsm_id, gsm in gse.gsms.items():

            text = " ".join([str(v) for v in gsm.metadata.values()]).lower()

            label = "unspecified"

            if any(k in text for k in tumor_keywords):
                tumor.append(gsm_id)
                label = "tumor"

            elif any(k in text for k in normal_keywords):
                normal.append(gsm_id)
                label = "normal"

            else:
                unspecified.append(gsm_id)

            annotation_info[gsm_id] = label

        print(f"\n========== SAMPLE SUMMARY: {gse_id} ==========")
        print(f"Cancer type: {cancer_type}")
        print("Normal samples:", ", ".join(normal) if normal else "None")
        print("Tumor samples:", ", ".join(tumor) if tumor else "None")
        print("Unspecified samples:", ", ".join(unspecified) if unspecified else "None")

        return normal, tumor, unspecified, annotation_info, cancer_type


    def _predict_cancer_type(self, gse):

        text = (gse.metadata.get("title", [""])[0] + " " +
                gse.metadata.get("summary", [""])[0]).lower()

        cancers = {
            "ovarian": "ovarian_cancer",
            "breast": "breast_cancer",
            "lung": "lung_cancer",
            "colon": "colon_cancer",
            "colorectal": "colorectal_cancer",
            "prostate": "prostate_cancer",
            "pancreatic": "pancreatic_cancer",
            "liver": "liver_cancer",
            "hepatocellular": "liver_cancer",
            "brain": "brain_cancer",
            "glioblastoma": "glioblastoma",
            "leukemia": "leukemia",
            "lymphoma": "lymphoma",
            "melanoma": "melanoma",
            "kidney": "kidney_cancer",
            "renal": "kidney_cancer",
            "bladder": "bladder_cancer",
            "gastric": "gastric_cancer",
            "stomach": "gastric_cancer",
            "esophageal": "esophageal_cancer",
            "thyroid": "thyroid_cancer",
            "cervical": "cervical_cancer",
            "endometrial": "endometrial_cancer"
        }

        for k, v in cancers.items():
            if k in text:
                return v

        return None


    def _build_h5ad(self, gse_id, tumor_samples, save_single=False):

        if len(tumor_samples) == 0:
            return None

        print("\n========== Reading Tumor Samples ==========")

        adatas = []

        # ✅ use supplementary folder if exists
        supp_dir = f"{gse_id}_Supp"
        if os.path.exists(supp_dir):
            base_dir = supp_dir
        else:
            base_dir = os.path.join(self.base_dir, gse_id)

        for root, dirs, files in os.walk(base_dir):

            for gsm_id in tumor_samples:

                if gsm_id not in root:
                    continue

                try:

                    if any(f.endswith(".mtx.gz") for f in files):
                        print(f"Reading MTX for {gsm_id}")

                        adata = sc.read_10x_mtx(root, var_names="gene_symbols")

                    elif any(f.endswith((".txt", ".csv", ".tsv", ".txt.gz")) for f in files):
                        print(f"Reading TEXT for {gsm_id}")

                        file = [f for f in files if f.endswith((".txt", ".csv", ".tsv", ".txt.gz"))][0]

                        df = pd.read_csv(os.path.join(root, file), sep=None, engine="python", index_col=0)

                        adata = ad.AnnData(df.T)

                    else:
                        continue

                    adata.obs["gsm_id"] = gsm_id
                    adata.obs["gse_id"] = gse_id

                    adata.layers["counts"] = adata.X.copy()
                    adata.raw = adata

                    adata.obs_names_make_unique()

                    adatas.append(adata)

                except Exception as e:
                    print(f"Skipping {gsm_id} due to error: {e}")
                    continue

        if len(adatas) == 0:
            print("No valid tumor data found.")
            return None

        combined = ad.concat(adatas, join="outer")
        combined.obs_names_make_unique()

        if save_single:
            filename = f"{gse_id}_tumor.h5ad"
            combined.write(filename)

            print("\n========== h5ad created ==========")
            print(f"{filename} is created successfully")

        return combined
