# Data Setup

This repository does not include the raw Olist data.

Download the **Brazilian E-Commerce Public Dataset by Olist** from Kaggle and place the following files in this folder:

```text
data/
├── olist_customers_dataset.csv
├── olist_geolocation_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_orders_dataset.csv
├── olist_products_dataset.csv
├── olist_sellers_dataset.csv
└── product_category_name_translation.csv
```

The notebook accepts customer filenames matching `olist_customers_dataset*.csv`, so minor local filename suffixes will also work.

## Why the raw files are excluded

Keeping the dataset outside the repository:
- avoids committing large files,
- keeps the repository lightweight,
- makes the project easier to review,
- preserves the original dataset distribution/source.

After placing the files here, open `../notebooks/DSML_Olist_Complete_Project.ipynb` and run all cells.
