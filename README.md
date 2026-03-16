# Distributed ML with Dask

A demonstration repository for scalable machine learning workflows using Dask on larger-than-memory tabular datasets.

## Purpose

This repository illustrates practical machine learning engineering patterns for:

- distributed data loading
- scalable preprocessing
- larger-than-memory workflows
- structured ML pipelines
- configuration-driven experimentation

It is designed as a public demonstration repository using synthetic or public-style tabular data. Proprietary systems and internal datasets are not included.

## Repository Structure

```text
src/
    data_generation.py
    distributed_preprocessing.py
    distributed_training.py
    run_pipeline.py

experiments/
    pipeline_config.yaml

results/
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python src/run_pipeline.py
```

## Notes

This repository contains demonstration implementations illustrating scalable machine learning engineering patterns using Dask.
