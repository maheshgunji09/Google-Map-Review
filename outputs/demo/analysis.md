# GeoReview India analysis

**Synthetic demonstration.** Every venue and rating in this run is fictional. The results show how the methods behave on the generated inputs. Research questions about actual Google Maps businesses in India remain open.

## Coverage and segmentation

The validated sample contains **96 businesses** with **3,779 reviews**. The median business has 37 reviews. 2 businesses have a single rating. Dated reviews span 2013-01-28 through 2016-03-01.

### RQ1 · Which experience patterns emerged?

| Segment | Businesses | Mean venue rating | Median reviews per venue |
| --- | ---: | ---: | ---: |
| Consistently positive reviews (group 1) | 24 | 4.62 | 45 |
| Consistently positive reviews (group 2) | 12 | 4.53 | 4.5 |
| Moderate ratings | 12 | 3.93 | 5.5 |
| High-volume mixed reviews | 24 | 3.45 | 78 |
| Lower-rated experiences | 24 | 2.12 | 31 |

Profile means give each business equal weight. Cluster names describe observed evidence.

### RQ2 · Which measurements distinguish the segments?

- positive ratio (descriptive feature) has a between-segment variance share of 95.0%.
- mean rating (model input) has a between-segment variance share of 93.6%.
- negative ratio (descriptive feature) has a between-segment variance share of 93.1%.

These shares describe assignments produced by this run. Features derived from the same star counts are correlated, and these values do not establish causality.

### RQ3 · How do the clustering methods compare?

Within each method family, the following eligible candidate had the lowest mean rank.

| Configuration | Silhouette | Davies–Bouldin | Resampling ARI |
| --- | ---: | ---: | ---: |
| kmeans-centroid-euclidean-k5 | 0.578 | 0.754 | 0.981 |
| agglomerative-ward-euclidean-k5 | 0.578 | 0.754 | 0.981 |

Silhouette uses each candidate’s distance. Davies–Bouldin uses Euclidean centroids for a common compactness measure. The K-Means WCSS curve suggests an elbow at k = 3.

### RQ4 · How much does changing distance alter membership?

Across 10 matched linkage and cluster-count pairs, Euclidean and Manhattan partitions had ARIs from 0.030 to 1.000. The lowest agreement occurred with complete linkage at k = 2. An ARI of 1 indicates identical partitions up to label permutation.

### RQ5 · Which configuration was selected?

**kmeans-centroid-euclidean-k5** obtained a silhouette of 0.578, a Davies–Bouldin index of 0.754 and a resampling ARI of 0.981. The smallest fitted cluster contains 12 businesses. The comparison used 96 fitted businesses. The first two PCA axes capture 79.8% of the fit cohort’s standardized variation.

### RQ6 · How can a user explore local services?

Select a city and service in the dashboard, then choose the review pattern and minimum evidence that fit the decision. For a business with many high ratings, check the share of low ratings and the review count alongside the mean. Sparse feedback provides little evidence about consistency.

## Limits of interpretation

- Historical reviews cannot establish current quality, availability, or opening hours.
- Archive coverage and geographic validation can favor well-documented urban venues. Business counts do not measure the prevalence of a pattern across India.
- Self-selection, uneven star use, duplicate content, coordinated activity and missing dates can alter apparent satisfaction and review frequency. Duplicate removal cannot certify authenticity.
- Text coverage varies across languages and scripts. Missing sentiment is unknown. Review-level exports omit text and reviewer identity.
- Internal separation and resampling stability describe this sample. They do not validate service quality or establish which recommendation a customer will prefer.

The [geographic and cleaning audit](audit.json), [model settings](model_summary.json), [full configuration grid](metrics.csv), [feature contrasts](feature_contrasts.csv), [distance comparisons](distance_agreement.csv) and [service mix](category_mix.csv) contain the measurements behind this analysis.
