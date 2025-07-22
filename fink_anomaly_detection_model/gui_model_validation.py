import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import webbrowser
from PIL import Image, ImageTk
import numpy as np
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import pandas as pd
from collections import defaultdict
import os
from coniferest.aadforest import AADForest, Label

def find_or_download_file(filename, save_dir='.'):
    file_path = os.path.join(save_dir, filename)
    if os.path.isfile(file_path):
        print(f"[INFO] File '{filename}' already exists.")
        return file_path
    print(f"[INFO] File '{filename}' not found. Starting download default dataset...")
    download_url = f'https://file.cosmos.msu.ru/files/{filename}'
    try:
        # Send a GET request with streaming
        response = requests.get(download_url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        # Get file size from headers
        total_size = int(response.headers.get('Content-Length', 0))
        downloaded_size = 0
        chunk_size = 1024
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)

                    # Display progress
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r[Download] {progress:.2f}% ({downloaded_size}/{total_size} bytes)", end='', flush=True)
                    else:
                        print(f"\r[Download] {downloaded_size} bytes", end='', flush=True)

        print(f"\n[INFO] File successfully downloaded and saved as '{filename}'.")
        return file_path

    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Error downloading file: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)  # Remove incomplete file
        return None


def get_cutout(ztf_id):
    """Load cutout image via Fink API
    Parameters
    ----------
        ztf_id : str
            unique identifier for this object

    Returns
    -------
        out : BytesIO stream
            cutout image in png format

    """
    # transfer cutout data
    r = requests.post(
        "https://api.fink-portal.org/api/v1/cutouts",
        json={"objectId": ztf_id, "kind": "Science", "output-format": "array"},
    )
    if not status_check(r, "get cutouts"):
        return io.BytesIO()
    data = np.log(np.array(r.json()["b:cutoutScience_stampData"], dtype=float))
    plt.axis("off")
    plt.imshow(data, cmap="PuBu_r")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return buf

def get_curve(ztf_id, last_days=None):
    """Load light curve image via Fink API and optionally plot only the latest data points (optional).

    Parameters
    ----------
        ztf_id : str
            unique identifier for this object
        last_days : int or None
            if set, include only the latest N days of observations.
            if None, include all available data (default)

    Returns
    -------
        out : BytesIO stream
            light curve picture
    """
    r = requests.post(
        "https://api.fink-portal.org/api/v1/objects",
        json={"objectId": ztf_id, "withupperlim": "True"},
    )
    if not status_check(r, "getting curve"):
        return None

    # Format output in a DataFrame
    pdf = pd.read_json(io.BytesIO(r.content))

    # Convert JD to MJD
    pdf["mjd"] = pdf["i:jd"].apply(lambda x: x - 2400000.5)

    # Optionally filter by last N days
    if last_days is not None:
        latest_mjd = pdf["mjd"].max()
        pdf_filtered = pdf[pdf["mjd"] >= (latest_mjd - last_days)]
    else:
        pdf_filtered = pdf  # use all data

    plt.figure(figsize=(15, 6))

    colordic = {1: "C0", 2: "C1"}
    filter_dict = {1: "g band", 2: "r band"}

    for filt in np.unique(pdf_filtered["i:fid"]):
        if filt == 3:
            continue
        maskFilt = pdf_filtered["i:fid"] == filt

        # Valid points
        maskValid = pdf_filtered["d:tag"] == "valid"
        plt.errorbar(
            pdf_filtered[maskValid & maskFilt]["mjd"],
            pdf_filtered[maskValid & maskFilt]["i:magpsf"],
            pdf_filtered[maskValid & maskFilt]["i:sigmapsf"],
            ls="",
            marker="o",
            color=colordic[filt],
            label=filter_dict[filt],
        )

        # Upper limits
        maskUpper = pdf_filtered["d:tag"] == "upperlim"
        plt.plot(
            pdf_filtered[maskUpper & maskFilt]["mjd"],
            pdf_filtered[maskUpper & maskFilt]["i:diffmaglim"],
            ls="",
            marker="^",
            color=colordic[filt],
            markerfacecolor="none",
        )

        # Bad quality
        maskBadquality = pdf_filtered["d:tag"] == "badquality"
        plt.errorbar(
            pdf_filtered[maskBadquality & maskFilt]["mjd"],
            pdf_filtered[maskBadquality & maskFilt]["i:magpsf"],
            pdf_filtered[maskBadquality & maskFilt]["i:sigmapsf"],
            ls="",
            marker="v",
            color=colordic[filt],
        )

    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel("Modified Julian Date")
    plt.ylabel("Difference magnitude")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf


SUPPLEMENTARY_DATASETS = {
    "July_21-22_2025": "https://file.cosmos.msu.ru/files/jul_21_22_07_25.parquet",
}


COMMON_REMS_GUI = [
    'percent_amplitude', 'linear_fit_reduced_chi2', 'inter_percentile_range_10',
    'mean_variance', 'linear_trend', 'standard_deviation',
    'weighted_mean', 'mean'
]


def extract_all(data) -> pd.Series:
    """
    Function for extracting data from lc_features
    :param data: dict
                lc_features dict
    :param key: str
                Name of the extracted filter
    :return: pd.DataFrame
                Dataframe with a specific filter
    """
    series = pd.Series(data, dtype=float)
    return series

def extract_one(data, key) -> pd.Series:
    """
    Function for extracting data from lc_features
    :param data: dict
                lc_features dict
    :param key: str
                Name of the extracted filter
    :return: pd.DataFrame
                Dataframe with a specific filter
    """
    series = pd.Series(data[key], dtype=float)
    return series

def status_check(r, error_msg):
    if r.status_code != 200:
        print(f"Failed {error_msg}: Status Code {r.status_code}")
        return False
    return True

class ResultsWindow:

    def __init__(self, parent, top_anomalies_df):
        self.top_anomalies = top_anomalies_df
        self.win = tk.Toplevel(parent)
        self.win.title("Top 10 Anomalous Objects")
        self.win.geometry("1000x800")

        main_frame = tk.Frame(self.win)
        main_frame.pack(fill=tk.BOTH, expand=1)

        self.canvas = tk.Canvas(main_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.populate_results()
        self.win.after_idle(lambda: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

    def populate_results(self):
        for index, row in self.top_anomalies.iterrows():
            object_id = row['objectId']
            score = row['score']
            item_frame = ttk.Labelframe(self.scrollable_frame, text=f"Rank {index + 1}: {object_id}", padding=10)
            item_frame.pack(pady=10, padx=10, fill=tk.X, expand=True)

            info_frame = ttk.Frame(item_frame)
            info_frame.pack(fill=tk.X)
            ttk.Label(info_frame, text=f"Anomaly Score: {score:.4f}").pack(side=tk.LEFT)
            link_label = tk.Label(info_frame, text="Open in Fink Portal", fg="blue", cursor="hand2")
            link_label.pack(side=tk.RIGHT)
            link_label.bind("<Button-1>",
                            lambda e, url=f"https://fink-portal.org/{object_id}": webbrowser.open_new_tab(url))
            images_frame = ttk.Frame(item_frame)
            images_frame.pack(fill=tk.X, pady=5)
            curve_label = ttk.Label(images_frame, text="Loading light curve...")
            curve_label.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.BOTH)
            cutout_label = ttk.Label(images_frame, text="Loading cutout...")
            cutout_label.pack(side=tk.RIGHT, padx=5, expand=True, fill=tk.BOTH)
            threading.Thread(
                target=self.load_images_for_item,
                args=(object_id, curve_label, cutout_label),
                daemon=True
            ).start()

    def load_images_for_item(self, object_id, curve_label, cutout_label):
        def update_curve_label(photo_image=None, error_msg=None):
            if photo_image:
                curve_label.config(image=photo_image, text="")
                curve_label.image = photo_image
            elif error_msg:
                curve_label.config(text=f"Error loading curve:\n{error_msg}", image='')

        def update_cutout_label(photo_image=None, error_msg=None):
            if photo_image:
                cutout_label.config(image=photo_image, text="")
                cutout_label.image = photo_image
            elif error_msg:
                cutout_label.config(text=f"Error loading cutout:\n{error_msg}", image='')

        try:
            curve_data = get_curve(object_id)
            if curve_data:
                img_curve = Image.open(curve_data).resize((450, 180), Image.LANCZOS)
                photo_curve = ImageTk.PhotoImage(img_curve)
                self.win.after(0, lambda p=photo_curve: update_curve_label(photo_image=p))
            else:
                 self.win.after(0, lambda: update_curve_label(error_msg="No data returned"))
        except Exception as e:
            self.win.after(0, lambda err=str(e): update_curve_label(error_msg=err))

        # try:
        #     cutout_data = get_cutout(object_id)
        #     if cutout_data and cutout_data.getbuffer().nbytes > 0:
        #         img_cutout = Image.open(cutout_data).resize((180, 180), Image.LANCZOS)
        #         photo_cutout = ImageTk.PhotoImage(img_cutout)
        #         self.win.after(0, lambda p=photo_cutout: update_cutout_label(photo_image=p))
        #     else:
        #          self.win.after(0, lambda: update_cutout_label(error_msg="No data returned"))
        # except Exception as e:
        #     self.win.after(0, lambda err=str(e): update_cutout_label(error_msg=err))


class FinkAnalyzerApp:

    def __init__(self, root, base_dataset, reactions_datasets, reactions):
        self.root = root
        self.root.title("Fink Anomaly Detection Model Trainer")
        self.base_dataset = base_dataset
        self.reactions_datasets = reactions_datasets
        self.reactions = reactions
        self.suppl_preprocessed = False
        self.supp_data, self.object_ids = None, None

        self.DEFAULT_PARAMS = {
            'n_trees': tk.StringVar(value='150'),
            'max_depth': tk.StringVar(value='28'),
            'n_subsamples': tk.StringVar(value='26568'),
            'C_a': tk.StringVar(value='1000'),
            'budget': tk.StringVar(value='100'),
            'random_seed': tk.StringVar(value='42'),
        }

        self.setup_widgets()

    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def setup_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        params_frame = ttk.Labelframe(main_frame, text="Model Hyperparameters", padding="10")
        params_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        row = 0
        for name, var in self.DEFAULT_PARAMS.items():
            ttk.Label(params_frame, text=name).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Entry(params_frame, textvariable=var, width=15).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            row += 1

        data_frame = ttk.Labelframe(main_frame, text="Target dataset", padding="10")
        data_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(data_frame, text="Supplementary Dataset:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.dataset_selector = ttk.Combobox(data_frame, values=list(SUPPLEMENTARY_DATASETS.keys()), width=40)
        self.dataset_selector.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        if SUPPLEMENTARY_DATASETS:
            self.dataset_selector.current(0)

        self.start_button = ttk.Button(main_frame, text="Train and Evaluate", command=self.start_analysis_thread)
        self.start_button.grid(row=2, column=0, columnspan=2, pady=10)

        log_frame = ttk.Labelframe(main_frame, text="Log", padding="10")
        log_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.log_text = scrolledtext.ScrolledText(log_frame, width=80, height=15, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.root.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def start_analysis_thread(self):
        self.start_button.config(state=tk.DISABLED)
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _preprocess_data(self, df, is_supplementary=False):
        self.log(f"Preprocessing data... Shape before: {df.shape}")

        object_ids = df['objectId'] if is_supplementary else None

        if "lc_features" in df.columns:
            features_r = df["lc_features"].apply(lambda data: extract_one(data, "1")).add_suffix("_r")
            features_g = df["lc_features"].apply(lambda data: extract_one(data, "2")).add_suffix("_g")
        elif "lc_features_r" in df.columns:
            features_r = df["lc_features_r"].apply(extract_all).add_suffix("_r")
            features_g = df["lc_features_g"].apply(extract_all).add_suffix("_g")
        else:
            raise ValueError("Feature columns (lc_features or lc_features_r/g) not found!")

        processed_df = pd.concat([features_r, features_g], axis=1).dropna(axis=0)
        if object_ids is not None:
            object_ids = object_ids.loc[processed_df.index]

        self.log(f"Shape after dropna: {processed_df.shape}")

        data_by_filter = defaultdict(lambda: defaultdict(list))
        for _, row in processed_df.iterrows():
            for passband in ['_r', '_g']:
                new_data = data_by_filter[passband]
                for col, val in zip(processed_df.columns, row):
                    if col.endswith(passband):
                        new_data[col[:-2]].append(val)

        final_data = {}
        for passband, data_dict in data_by_filter.items():
            df_passband = pd.DataFrame(data=data_dict).astype('float64')
            df_passband = df_passband.drop(labels=COMMON_REMS_GUI, axis=1, errors='ignore')
            final_data[passband] = df_passband.values.copy(order='C')

        # self.log(f"Final feature count for _r: {final_data['_r'].shape[1]}")
        # self.log(f"Final feature count for _g: {final_data['_g'].shape[1]}")

        return final_data, object_ids

    def _run_analysis(self):
        try:
            self.log("=" * 30)
            self.log("Analysis started.")

            try:
                params = {name: int(var.get()) for name, var in self.DEFAULT_PARAMS.items()}
                # n_jobs=None для AADForest
                params['n_jobs'] = None
                self.log(f"Using parameters: {params}")
            except ValueError:
                messagebox.showerror("Error", "All hyperparameters must be integers.")
                return

            base_data= self.base_dataset


            selected_dataset_name = self.dataset_selector.get()
            if not selected_dataset_name:
                messagebox.showerror("Error", "Please select a supplementary dataset.")
                return

            url = SUPPLEMENTARY_DATASETS[selected_dataset_name]
            filename = os.path.basename(url)
            self.log(f"Downloading supplementary dataset '{filename}'...")
            supp_path = find_or_download_file(filename)
            if not supp_path:
                self.log("Failed to download supplementary dataset.")
                return
            if not self.suppl_preprocessed:
                supp_df = pd.read_parquet(supp_path)
                if 'objectId' not in supp_df.columns and 'i:objectId' in supp_df.columns:
                    supp_df.rename(columns={'i:objectId': 'objectId'}, inplace=True)

                supp_data, object_ids = self._preprocess_data(supp_df, is_supplementary=True)
                self.supp_data, self.object_ids = supp_data, object_ids
                self.suppl_preprocessed = True
            else:
                supp_data, object_ids = self.supp_data, self.object_ids

            total_scores = np.zeros(supp_data['_r'].shape[0])

            for key in ['_r', '_g']:
                self.log(f"\n--- Processing filter: {key} ---")
                self.log("Training AADForest on base data...")
                forest = AADForest(**params).fit_known(
                    base_data[key],
                    known_data=self.reactions_datasets[key],
                    known_labels=self.reactions
                )
                self.log("Applying model to supplementary data...")
                scores = forest.score_samples(supp_data[key])
                total_scores += scores

            self.log("\nSearching for top 10 anomalies...")
            results_df = pd.DataFrame({'objectId': object_ids, 'score': total_scores}).drop_duplicates(subset=['objectId'])
            top_anomalies = results_df.nsmallest(10, 'score').reset_index(drop=True)

            self.log("Top 10 most anomalous objects found:")
            for i, row in top_anomalies.iterrows():
                self.log(f"  {i + 1}. ID: {row['objectId']}, Score: {row['score']:.4f}")
            self.root.after(0, lambda: ResultsWindow(self.root, top_anomalies))

        except Exception as e:
            self.log(f"\nAN ERROR OCCURRED: {e}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("Runtime Error", str(e))
        finally:
            self.log("=" * 30)
            self.log("Analysis finished.")
            self.start_button.config(state=tk.NORMAL)


def launch_gui_analyzer(base_dataset, reactions_datasets, reactions):
    root = tk.Tk()
    app = FinkAnalyzerApp(root, base_dataset, reactions_datasets, reactions)
    root.mainloop()
