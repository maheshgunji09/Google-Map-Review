# GeoReview India

GeoReview India groups Indian venues by the distribution, volume and timing of their reviews. The local dashboard lets a user choose a city, service and experience pattern, then inspect the evidence for an individual business.

**The bundled run uses fictional data.** Its [analysis](outputs/demo/analysis.md) demonstrates the research workflow. India-specific counts and findings from Google Local Reviews remain open until an authorized archive is processed.

## Open the dashboard

From the project directory, use Python 3.11 and the tested package versions.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard opens the completed research run in `outputs/research` when available and otherwise opens `outputs/demo`. The explorer offers a lower confidence bound on the positive-review share, rating consistency and review volume as sorting options. Select a row to see its star distribution and sample size. The other views show PCA, segment profiles, clustering metrics and data limitations.

The overview uses [ybouane/LiquidGlass](https://github.com/ybouane/liquidglass), pinned to version 1.0.3 and loaded from jsDelivr in a Streamlit v2 component. A single WebGL instance renders the overview panel and is destroyed on component cleanup. The rest of the dashboard uses coordinated translucent surfaces and native analytical controls. Reduced-motion or increased-contrast preferences use the static panel. A static fallback also remains available when the CDN or WebGL is unavailable. Inter fonts load from Google Fonts with a system-font fallback.

## Reproduce the demonstration

```powershell
python -m georeview demo --seed 42 --output outputs/demo-rebuilt
$env:GEOREVIEW_RUN = "outputs/demo-rebuilt"
python -m streamlit run app.py
```

Every run creates a fresh directory. The default seed reproduces fictional venues across six cities and four service categories, along with invalid locations and duplicate reviews to exercise the audit. The generator creates raw records locally and feeds them through the same ingestion, modeling and reporting code used for research. The source hashes, seed, requested cluster counts and library versions are recorded in `manifest.json`.

## Work with the research archive

The target is the **Google Local Reviews (2018)** places and reviews release. The publisher lists 3,116,785 businesses and 11,453,845 reviews across the **entire archive**. These totals provide no India-specific count. The separate 2021 release covers the United States. [UCSD 2018 catalogue](https://mcauleylab.ucsd.edu/public_datasets/data/datasets.html#google_local) · [UCSD 2021 scope](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/)

The 2018 release page requests citation of the following papers. I found no explicit reuse licence or redistribution grant on that page. The Canadian Urban Data Catalogue also records the licence as unspecified. For a research run, use local archives covered by terms or permission applicable to the intended analysis. Record that basis in the run manifest. A citation alone does not resolve permission to redistribute the data. [Publisher’s terms and citations](https://mcauleylab.ucsd.edu/public_datasets/data/datasets.html#google_local) · [Catalogue licence record](https://data.urbandatacentre.ca/en/catalogue/google-local-reviews-2018)

- Rajiv Pasricha and Julian McAuley. *Translation-based factorization machines for sequential recommendation*. RecSys, 2018.
- Ruining He, Wang-Cheng Kang and Julian McAuley. *Translation-based recommendation*. RecSys, 2017.

Place authorized `places.clean.json.gz` and `reviews.clean.json.gz` files under `data/raw/`, or supply their actual paths. The operator identifies their origin. SHA-256 fingerprints identify the exact local inputs used on subsequent runs. The reader uses the archive’s `gPlusPlaceId` as the join key. It accepts gzip-compressed or plain newline-delimited records, including the publisher’s legacy Python-literal dictionaries. The user-profile archive is unnecessary. This project reads local files and has no Maps extraction or scraping step.

```powershell
$reuseBasis = Read-Host "Enter the permission or terms reference that covers this analysis"
python -m georeview research --places data/raw/places.clean.json.gz --reviews data/raw/reviews.clean.json.gz --reuse-note "$reuseBasis"
```

Use `--prepare-only` to export the validated dataset and audit independently of model fitting. This also works when the filtered cohort is too small or too uniform to cluster. Use `--output` to set a fresh run directory, `--k 2 3 4 5 6` to select the grid, and `--max-fit 2000` to set the maximum comparison cohort. A larger fit cap increases the memory required for pairwise distances quadratically. Set `GEOREVIEW_RUN` as in the demo example to explore a specific completed run.

## Analytical design

Geographic filtering requires an explicit India address and coordinates inside the bundled Natural Earth polygon, with a 0.01-degree exclusion near its edges. This can discard valid coastal and border records. The current city recognizer corroborates ten named cities using address aliases and proximity. Other geographically eligible businesses appear under Unknown city. Review frequency is unavailable for observation spans under 30 days. Text sentiment is currently unavailable pending a validated language-specific processor, and requesting `--sentiment` reports that dependency. [Boundary provenance and limitations](georeview/data/SOURCE.md)

1. **Validate and join.** Validate places first, then retain eligible reviews through the place ID join. The location and cleaning audit records exclusions. Only a review date, business identifier and star rating are needed in the cleaned review export. Direct reviewer identifiers, names, profile data and raw review text are excluded from final outputs.
2. **Describe each venue.** Count reviews, compute the five rating shares, mean and variance, and separate four- and five-star feedback from one- and two-star feedback. Count dated observations for activity rates. A single rating provides insufficient evidence of consistency. Missing sentiment is treated as unobserved.
3. **Select and scale features.** The fit cohort sets medians for missing values and the parameters for standardization. Model inputs include rating level, dispersion, three-star share and log-transformed volume and frequency when they vary and meet coverage requirements. The model can accept a future validated sentiment feature only with adequate coverage. Sparse one-hot service categories produce the segment service mix. City and service serve as discovery filters, so geography and catalogue labels cannot determine the fitted review patterns.
4. **Compare models.** Every candidate sees the same deterministic cohort. K-Means uses Euclidean distance and ten starts per fit. Hierarchical candidates use Ward/Euclidean and average or complete linkage under both Euclidean and Manhattan distances. A normalized chord estimates a bend in the K-Means WCSS curve. Silhouette uses the candidate distance. Davies–Bouldin uses Euclidean centroid geometry throughout.
5. **Measure stability and choose.** Three additional K-Means seeds measure initialization sensitivity. Three 80% subsamples refit the preprocessing and clustering, then compare labels on shared businesses using Adjusted Rand Index (ARI). Selection favors the lowest average rank across silhouette, Davies–Bouldin and subsample ARI among candidates meeting minimum cluster support when possible. Ties prefer fewer clusters and then K-Means. The elbow and profile descriptions provide separate interpretive checks. Matched-linkage ARIs quantify changes caused by switching distance.
6. **Interpret.** Fit PCA on the model cohort for the two-dimensional view. For cohorts above the cap, assign other businesses to the closest centroid for K-Means or the closest fitted cluster medoid for hierarchical clustering. The assignment column identifies these approximations. Fit metrics apply to the common cohort. Segment profiles describe the full assigned population.

## Run outputs

| File | What it contains |
| --- | --- |
| `reviews.csv` | Cleaned, geographically eligible review observations |
| `businesses.csv` | One row per eligible reviewed venue, including star shares and evidence counts |
| `audit.json`, `manifest.json` | Retention and exclusion counts, input provenance and run settings |
| `metrics.csv`, `distance_agreement.csv` | Candidate scores, stability, support and Euclidean/Manhattan agreement |
| `assignments.csv`, `profiles.csv` | Segment, assignment method, PCA coordinates and descriptive profiles |
| `feature_contrasts.csv`, `category_mix.csv` | Feature differences and service composition within segments |
| `model_summary.json` | Selected model, preprocessing, fit size, PCA variance and elbow measurements |
| `analysis.md` | A run-specific interpretation addressing the six research questions |

Historical coverage, location uncertainty, underrepresented languages, skewed star ratings, review manipulation and sparse business histories limit interpretation. Recommendations support exploration of observed customer experience. Present-day quality or availability requires independent evidence. The optional tourist-destination extension is outside the scope of the core dataset.

## Verification

```powershell
python -m unittest discover -s tests -v
```
