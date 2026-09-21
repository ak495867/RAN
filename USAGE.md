# Usage

Install the dependencies:

```bash
pip install numpy pandas scipy matplotlib torch yfinance lxml
```

### Check data

```bash
python ran_research.py --check_data
python ran_research.py --check_data --universe multiasset --start 2008-01-01
```

These only check the download/universe. No training is run.

### Run

```bash
python ran_research.py --stage dev --universe sp500 --out runs/dev_sp500

python ran_research.py --stage dev --universe multiasset \
    --start 2008-01-01 \
    --out runs/dev_multi
```

`--stage dev` keeps the run on the development period; use it to test the pipeline and fix the setup before running a final holdout evaluation.
