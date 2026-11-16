# Data

Nothing in `data/` is committed. Fetch it with:

```bash
kaggle competitions download playground-series-s6e11 -p data && unzip -o data/playground-series-s6e11.zip -d data
kaggle datasets download uditjain13/credit-card-fraud-detection-2026 -p data --unzip
mv data/credit_card_fraud_2026.csv data/original.csv
```

## Files

| File | Rows | Fraud rate | Source and licence |
| --- | --- | --- | --- |
| `train.csv` | 699,960 | 1.67% (11,689 positives) | Competition data, synthetic, under the competition rules |
| `test.csv` | 299,984 | n/a | Competition data |
| `original.csv` | 20,000 | 1.69% | [Credit Card Fraud Detection 2026](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026), CC0-1.0 |

The competition data was generated from the original dataset: the column set is identical, with
`transaction_id` in the original in place of `id`. No missing values in any file.

## Columns

| Group | Columns |
| --- | --- |
| Numeric | `amount_usd`, `hours_since_last_txn`, `distance_from_home_km`, `account_balance_usd`, `velocity_score`, `merchant_risk_score` |
| Integer | `txn_count_last_24h` (12 values), `card_age_months` (49), `customer_age` (64), `cvv_retry_count` (4), `time_of_day_hour` (24), `day_of_week` (7), `prior_disputes` (5) |
| Categorical | `merchant_category` (12), `card_type` (5), `auth_method` (5), `channel` (5), `device_type` (8) |
| Boolean | `is_foreign_transaction`, `is_new_merchant`, `used_vpn`, `ip_country_mismatch`, `billing_shipping_mismatch`, `is_ai_generated_scam_attempt` |

Booleans arrive as `True`/`False` in the competition files and `TRUE`/`FALSE` in the original;
`src/common.py` maps both to 0/1.
