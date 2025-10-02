#!/bin/bash
hyperfine -w 1 ./run_aligner_transformers_reference.sh --export-json benchmark_results_transformers_reference.json
